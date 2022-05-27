import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data import MelAudioDataset, AddTargetDataset, Dictionary
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score, accuracy_score, precision_recall_fscore_support


def compute_metrics(cfs_matrix):
    """
      Calculate common metrics based on the confusion matrix
      Args:
        - cfs_matrix: a sklearn confusion matrix 
                      :type: a ndarray of shape (n_classes, n_classes)
      Returns:
        - precision: the precision of the prediction
                     :type: float  
        - recall: the recall of the prediction
                  :type: float  
        - f1: the f1-score of the prediction
              :type: float                       
    """     
    precision = cfs_matrix[1,1] / (cfs_matrix[1,1] + cfs_matrix[0,1])
    recall = cfs_matrix[1,1] / (cfs_matrix[1,1] + cfs_matrix[1,0])
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate(ensem_preds, targets):
    """
      Evaluate the prediction by providing metrics & also the best threshold (to get the highest f1-score)
      Ex: AUC, Accurary, Precision, Recall, F1-Score.
      Then print these metrics
      Args:
        - ensem_preds: predictions for ids
                       :type: a numpy array
        - targets: the actual results of ids
                   :type: a numpy array
      Returns:
        - None
    """
    best_th = 0
    best_score = 0

    for th in np.arange(0.0, 0.6, 0.01):
        pred = (ensem_preds > th).astype(int)
        score = f1_score(targets, pred)
        if score > best_score:
            best_th = th
            best_score = score

    print(f"\nAUC score: {roc_auc_score(targets, ensem_preds):12.4f}")
    print(f"Best threshold {best_th:12.4f}")

    preds = (ensem_preds > best_th).astype(int)
    # print(classification_report(targets, preds, digits=3))

    cm1 = confusion_matrix(targets, preds)
    print('\nConfusion Matrix : \n', cm1)
    precision, recall, f1 = compute_metrics(cm1)
    
    print('\n=============')
    print (f'Precision    : {precision:12.4f}')
    
    print(f'Recall : {recall:12.4f}')
    
    print(f'F1 Score : {f1:12.4f}')
    
    total1=sum(sum(cm1))

    print('\n=============')
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print (f'Accuracy    : {accuracy1:12.4f}')


def multi_class_evaluate(ensem_preds, targets):
    onehot_preds = [np.argmax(pred) for pred in ensem_preds]
    print(f"\nAUC score: {roc_auc_score(targets, ensem_preds, multi_class='ovr'):12.4f}")

    cm1 = confusion_matrix(targets, onehot_preds)
    print('\nConfusion Matrix : \n', cm1)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, onehot_preds, average='macro')
    acc = accuracy_score(targets, onehot_preds)

    print('\n=============')
    print (f'Precision  : {precision:12.4f}')
    
    print(f'Recall      : {recall:12.4f}')
    
    print(f'F1 Score    : {f1:12.4f}')

    print(f'Accuracy    : {acc:12.4f}')


def load_dataset(X, file_path, dir_path, label, offset=4):
    X[file_path] = X[file_path].apply(lambda x: dir_path + str(x))
    if offset == 0:
        feats = list(X[file_path].apply(lambda x: np.load(x + '_mel_2048_128.npy')).values)
    else:
        feats = list(X[file_path].apply(lambda x: np.load(x[:-offset] + '_mel_2048_128.npy')).values)
    # feats = np.stack(feats, axis=0)

    return feats, X[label].values


def load_fairseq_dataset(dir_path, orig_dir, splits=['train', 'valid', 'test']):
    '''
    dir_path: directory to fairseq meta data files
    orig_dir: original directory that stored the spectrum
    '''
    feats = []
    labels = []
    for split in splits:
        with open(os.path.join(dir_path, f'{split}.tsv')) as f, open(os.path.join(dir_path, f'{split}.label')) as label:
            for line, label in zip(f.read().split('\n')[1:-1], label.read().split('\n')):
                file_path, _, _, _ = line.split('\t')
                spec = np.load(os.path.join(orig_dir, file_path))
                feats.append(spec)
                labels.append(int(label))

    return feats, labels


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].shape[0]
    features = torch.zeros((len(data), n_ftrs, max_len))
    labels = torch.tensor(labels)
    # lengths = torch.tensor(lengths)
    lengths = torch.zeros((len(data), max_len))

    for i in range(len(data)):
        j, k = data[i][0].shape[0], data[i][0].shape[1]
        # print(torch.from_numpy(data[i][0]).shape)
        # print(torch.zeros((j, max_len - k)).shape)
        features[i] = torch.cat([torch.from_numpy(data[i][0]), torch.zeros((j, max_len - k))], dim=1)
        lengths[i][:k] = 1

    return features.float(), labels.long(), lengths.long()


class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.from_numpy(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y, x.shape[1]

    def __len__(self):
        return len(self.data)


# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='audio_finetuning')
args = options.parse_args_and_arch(parser)


# Setup task
task = tasks.setup_task(args)
# Load model
print(f' | loading model from ${args.path}')
# models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/SSD/tungtk2/fairseq/data/orig_2048_128', 'w2v_path': '/media/SSD/tungtk2/fairseq/outputs/2022-03-07/08-30-20/checkpoints/checkpoint_best.pt'})
models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/SSD/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4'})
model = models[0].cuda()

print(model)

# NOTE: For COVID-19 dataset

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/output/df_5fold.csv')
# sounddr_test_set_inp, sounddr_test_set_out = load_dataset(X[X['fold'] == 4], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/', 'label_covid', offset=4)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/assets/df_maj.csv')
# X = X.loc[~X['file_path'].isin(['assets/audio/isofh_undefined-1635044334.8719654.wav', 'assets/audio/isofh_undefined-1635054324.1639278.wav', 'assets/audio/isofh_undefined-1645344442.8312197.wav', 'assets/audio/isofh_undefined-1646753418.6567717.wav', 'assets/audio/isofh_undefined-1646086594.8671155.wav', 'assets/audio/isofh_undefined-1646133546.8110027.wav', 'assets/audio/QhUkPm-1633484885.7741814.wav', 'assets/audio/isofh_undefined-1634049245.820838.wav', 'assets/audio/isofh_undefined-1647675255.3401847.wav', 'assets/audio/yoJaXY-1646410609.823617.wav', 'assets/audio/isofh_undefined-1646882061.0733764.wav', 'assets/audio/isofh_undefined-1634905167.7832456.wav', 'assets/audio/bookingcare-211025-152625-61766a3167e69-1635150385.902807.wav'])]
# aicvvn_new_test_set_inp, aicvvn_new_test_set_out = load_dataset(X[X['fold'] == 4], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/', 'label', offset=4)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coughvid/df_fold.csv')
# coughvid_test_set_inp, coughvid_test_set_out = load_dataset(X[X['fold'] == 4], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coughvid/public_dataset/', 'label_covid', offset=5)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_train/public_train_metadata_fold.csv')
# aicvvn_test_set_inp, aicvvn_test_set_out = load_dataset(X[X['fold'] == 4], 'uuid', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_train/public_train_audio_files/', 'assessment_result', offset=0)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/df_fold.csv')
# coswara_test_set_inp, coswara_test_set_out = load_dataset(X[X['fold'] == 0], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/Coswara-Data_0511/', 'label_covid', offset=4)


# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/df_maj.csv')
# sounddr_maj_inp, sounddr_maj_out = load_dataset(X[X['fold']==4], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/', 'label')

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/assets/df_maj.csv')
# X = X.loc[~X['file_path'].isin(['assets/audio/isofh_undefined-1635044334.8719654.wav', 'assets/audio/isofh_undefined-1635054324.1639278.wav', 'assets/audio/isofh_undefined-1645344442.8312197.wav', 'assets/audio/isofh_undefined-1646753418.6567717.wav', 'assets/audio/isofh_undefined-1646086594.8671155.wav', 'assets/audio/isofh_undefined-1646133546.8110027.wav', 'assets/audio/QhUkPm-1633484885.7741814.wav', 'assets/audio/isofh_undefined-1634049245.820838.wav', 'assets/audio/isofh_undefined-1647675255.3401847.wav', 'assets/audio/yoJaXY-1646410609.823617.wav', 'assets/audio/isofh_undefined-1646882061.0733764.wav', 'assets/audio/isofh_undefined-1634905167.7832456.wav', 'assets/audio/bookingcare-211025-152625-61766a3167e69-1635150385.902807.wav'])]
# aicovidvn_new_maj_inp, aicovidvn_new_maj_out = load_dataset(X[X['fold']==4], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/', 'label')


# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/df_min.csv')
# sounddr_min_inp, sounddr_min_out = load_dataset(X[X['fold']==1], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/', 'label')

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/assets/df_min.csv')
# aicovidvn_new_min_inp, aicovidvn_new_min_out = load_dataset(X[X['fold']==1], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/', 'label')



# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_test/public_test_sample_submission.csv')
# res = X.copy()
# aicvvn_test_set_inp, aicvvn_test_set_out = load_dataset(X, 'uuid', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_test/public_test_audio_files/', 'assessment_result', offset=0)


# NOTE: For ICBHI dataset
icbhi_inp, icbhi_out = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32_unnormalized_log_augmented', '/media/SSD/tungtk2/RespireNet', splits=['test'])
# icbhi_inp, icbhi_out = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32', '/media/SSD/tungtk2/RespireNet', splits=['valid'])

# test_set_inp = [*coughvid_test_set_inp, *aicvvn_test_set_inp, *coswara_test_set_inp, *sounddr_test_set_inp, *aicvvn_new_test_set_inp]
# test_set_out = np.concatenate((coughvid_test_set_out, aicvvn_test_set_out, coswara_test_set_out, sounddr_test_set_out, aicvvn_new_test_set_out))

# test_set_inp = [*sounddr_maj_inp, *aicovidvn_new_maj_inp]
# test_set_out = np.concatenate((sounddr_maj_out, aicovidvn_new_maj_out))

# test_set_inp = [*sounddr_min_inp, *aicovidvn_new_min_inp]
# test_set_out = np.concatenate((sounddr_min_out, aicovidvn_new_min_out))


test_set_inp = [*icbhi_inp]
test_set_out = np.array(icbhi_out)

# # NOTE: For urban8k
# X = pd.read_csv('/media/SSD/tungtk2/UrbanSound8K/metadata/UrbanSound8K.csv')
# X['file_path'] = X.apply(lambda x: f"/media/SSD/tungtk2/UrbanSound8K/audio/fold{ x['fold'] }/{ x['slice_file_name'] }", axis=1)
# urban8k_test_set_inp, urban8k_test_set_out = load_dataset(X[X['fold'] == 10], 'file_path', '', 'classID')

# test_set_inp = [*urban8k_test_set_inp]
# test_set_out = np.array(urban8k_test_set_out)


test_dataset = MyDataset(test_set_inp, test_set_out)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

pred_array = []
target_array = []
model.eval()
model.encoder.eval()
model.decoder.eval()

print("MODEL PARAMETERS: ")
# for parameter in model.parameters():
#     print(parameter)
num_params = sum(param.numel() for param in model.parameters())
print(num_params)

# NOTE: for evaluating profiling model:
# pretrained_models, _ = checkpoint_utils.load_model_ensemble(['/media/SSD/tungtk2/fairseq/outputs/2022-04-19/21-32-58/checkpoints/checkpoint_best.pt'])
# pretrained_model = pretrained_models[0].cuda()
# pretrained_model.eval()

# def cross_entropy(X,y):
#     """
#     X is the output from fully connected layer (num_examples x num_classes)
#     y is labels (num_examples x 1)
#     	Note that y is not one-hot encoded vector. 
#     	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
#     """
#     m = y.shape[0]
#     # We use multidimensional array indexing to extract 
#     # softmax probability of the correct label for each sample.
#     # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
#     log_likelihood = -np.log(X[range(m),y])
#     loss = np.sum(log_likelihood) / m
#     return loss
# ces = []

for inputs, labels, lengths in dataloader:
    inputs = inputs.to('cuda', dtype=torch.float)
    labels = labels.to('cuda')
    lengths = lengths.unsqueeze(dim=1).to('cuda')

    encoder_out = model.encoder(inputs, lengths)
    encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'], dim=1)
    # outputs = model.decoder(encoder_out['encoder_out'])

    outputs = F.softmax(model.decoder(encoder_out['encoder_out']), dim=1)

    # ce_predict = outputs.detach().cpu().numpy()
    # ce_target = labels.detach().cpu().numpy()
    # ces.append(cross_entropy(ce_predict, ce_target))

    # NOTE: For ICBHI dataset:
    pred_array.append(int(outputs.argmax(dim=1).detach().cpu().numpy().squeeze()))

    # NOTE: for evaluating profiling model:
    # with torch.no_grad():
    #     pretrained_out = pretrained_model(inputs, lengths, features_only=True)
    # pretrained_output = pretrained_out['x'].squeeze()
    # pretrained_output = torch.mean(pretrained_output, dim=0)
    # pretrained_output = pretrained_output.unsqueeze(dim=0)
    # if len(pretrained_output.shape) == 1:
    #     pretrained_output = encoder_out['encoder_out']
    # decoder_input = torch.cat((encoder_out['encoder_out'], pretrained_output), dim=1)
    # outputs = model.decoder(decoder_input)

    # if outputs.detach().cpu().numpy().shape[0] == 1:
    #     pred_array.extend([outputs.detach().cpu().numpy().squeeze()[1]])
    # else:
    #     pred_array.extend(list(outputs.detach().cpu().numpy()[:, 1].squeeze()))

    # #NOTE: For Urban8k
    # # print("PREDICTION ARRAY SHAPE: ", outputs.detach().cpu().numpy().shape)
    # pred_array.append(outputs.detach().cpu().numpy().squeeze())

    target_array.extend(list(labels.detach().cpu().numpy()))


# res['assessment_result'] = pred_array

# res.to_csv('results.csv', index=False)

# print(evaluate(pred_array, target_array))

def get_score(hits, counts):
    se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3])
    sp = hits[0] / counts[0]
    print("SENSE: ", se)
    print("SPEC: ", sp)
    sc = (se+sp) / 2.0
    return sc

class_hits = [0.0, 0.0, 0.0, 0.0] # normal, crackle, wheeze, both
class_counts = [0.0, 0.0, 0.0+1e-7, 0.0+1e-7] # normal, crackle, wheeze, both
for idx in range(len(target_array)):
    class_counts[target_array[idx]] += 1.0
    if pred_array[idx] == target_array[idx]:
        class_hits[target_array[idx]] += 1.0

from sklearn.metrics import confusion_matrix
print(class_counts)
print("Accuracy: ", accuracy_score(target_array, pred_array))
conf_matrix = confusion_matrix(target_array, pred_array)
print(conf_matrix)
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
print("Classwise Scores", conf_matrix.diagonal())
print(get_score(class_hits, class_counts))


# import matplotlib.pyplot as plt
# ces = np.array(ces)
# q25, q75 = np.percentile(ces, [25, 75])
# bin_width = 2 * (q75 - q25) * len(ces) ** (-1/3)
# bins = round((ces.max() - ces.min()) / bin_width)
# print("Freedman–Diaconis number of bins:", bins)
# fig = plt.hist(ces, bins=bins)
# plt.savefig('cross_entropy_loss_train.png')
