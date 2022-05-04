import os
import torch
from torch.utils.data import Dataset, DataLoader
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


def load_dataset(X, file_path, dir_path, label, id, offset=4):
    X[file_path] = X[file_path].apply(lambda x: dir_path + str(x))
    if offset == 0:
        feats = list(X[file_path].apply(lambda x: np.load(x + '_mel_2048_128.npy')).values)
    else:
        feats = list(X[file_path].apply(lambda x: np.load(x[:-offset] + '_mel_2048_128.npy')).values)

    return feats, X[label].values, X[id].values


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    ids, _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][1].shape[0]
    features = torch.zeros((len(data), n_ftrs, max_len))
    labels = torch.tensor(labels)
    # lengths = torch.tensor(lengths)
    lengths = torch.zeros((len(data), max_len))

    for i in range(len(data)):
        j, k = data[i][1].shape[0], data[i][1].shape[1]
        # print(torch.from_numpy(data[i][0]).shape)
        # print(torch.zeros((j, max_len - k)).shape)
        features[i] = torch.cat([torch.from_numpy(data[i][1]), torch.zeros((j, max_len - k))], dim=1)
        lengths[i][:k] = 1

    return ids, features.float(), labels.long(), lengths.long()


class MyDataset(Dataset):
    def __init__(self, ids, data, targets):
        self.ids = ids
        self.data = data
        self.targets = torch.from_numpy(targets)

    def __getitem__(self, index):
        id = self.ids[index]
        x = self.data[index]
        y = self.targets[index]

        return id, x, y, x.shape[1]

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
# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coughvid/df_fold.csv')
# coughvid_test_set_inp, coughvid_test_set_out = load_dataset(X[X['fold'] == 4], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coughvid/public_dataset/', 'label_covid', offset=5)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_train/public_train_metadata_fold.csv')
# aicvvn_test_set_inp, aicvvn_test_set_out = load_dataset(X[X['fold'] == 4], 'uuid', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_train/public_train_audio_files/', 'assessment_result', offset=0)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/df_fold.csv')
# coswara_test_set_inp, coswara_test_set_out = load_dataset(X[X['fold'] == 0], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/Coswara-Data_0511/', 'label_covid')

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_test/public_test_sample_submission.csv')
# res = X.copy()
# aicvvn_test_set_inp, aicvvn_test_set_out = load_dataset(X, 'uuid', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_test/public_test_audio_files/', 'assessment_result', offset=0)

# test_set_inp = [*coughvid_test_set_inp, *aicvvn_test_set_inp, *coswara_test_set_inp]
# test_set_out = np.concatenate((coughvid_test_set_out, aicvvn_test_set_out, coswara_test_set_out))

# # NOTE: PROFILING
# def map_age(age):
#     groups = [[0, 2], [3, 5], [6, 13], [14, 18], [19, 33], [34, 48], [49, 64], [65, 78], [79, 98]]
#     for group in groups:
#         if group[0] <= age <= group[1]:
#             return f'group_{group[0]}_{group[1]}'
#     print(age)
# # X['a'] = X['a'].apply(map_age)

# def create_profile(row):
#     age = row['subject_age']
#     gender = row['subject_gender']
#     phase = 'train' if row['fold'] in [0, 1, 2] else 'valid' if row['fold'] == 3 else 'test'
#     return age + '_' + gender + '_' + phase
# def create_profile_by_id(row):
#     id = row['id']
#     phase = 'train' if row['fold'] in [1, 2, 3] else 'valid' if row['fold'] == 4 else 'test'
#     return id + '_' + phase
# X['profile'] = X.apply(create_profile_by_id, axis=1)
# # X['profile'] = X.apply(lambda row: row['a'] + '_' + row['g'], axis=1)
# # X['profile'] = X.apply(lambda row: row['subject_age'] + '_' + row['subject_gender'], axis=1)

# ids = list(X[X['fold'] == 0]['profile'])
profiles = torch.load('/media/SSD/tungtk2/fairseq/outputs/2022-04-19/21-32-58/checkpoints/profile.pt')

X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/assets/df_min.csv')
aicovidvn_new_min_inp, aicovidvn_new_min_out, aicovidvn_new_min_id = load_dataset(X[X['fold'] == 1], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/', 'label', 'id')

X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/df_min.csv')
sounddr_min_inp, sounddr_min_out, sounddr_min_id = load_dataset(X[X['fold'] == 1], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/', 'label', 'id')

test_set_inp = [*sounddr_min_inp, *aicovidvn_new_min_inp]
test_set_out = np.concatenate((sounddr_min_out, aicovidvn_new_min_out))
test_set_id = np.concatenate((sounddr_min_id, aicovidvn_new_min_id))

# # NOTE: For urban8k
# X = pd.read_csv('/media/SSD/tungtk2/UrbanSound8K/metadata/UrbanSound8K.csv')
# X['file_path'] = X.apply(lambda x: f"/media/SSD/tungtk2/UrbanSound8K/audio/fold{ x['fold'] }/{ x['slice_file_name'] }", axis=1)
# urban8k_test_set_inp, urban8k_test_set_out = load_dataset(X[X['fold'] == 10], 'file_path', '', 'classID')

# test_set_inp = [*urban8k_test_set_inp]
# test_set_out = np.array(urban8k_test_set_out)


test_dataset = MyDataset(test_set_id, test_set_inp, test_set_out)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

pred_array = []
target_array = []
model.eval()
model.encoder.eval()
model.decoder.eval()

for ids, inputs, labels, lengths in dataloader:
    inputs = inputs.to('cuda', dtype=torch.float)
    labels = labels.to('cuda')
    lengths = lengths.unsqueeze(dim=1).to('cuda')

    encoder_out = model.encoder(inputs, lengths)
    encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'], dim=1)

    # NOTE: only works for batch size = 1
    profile = profiles[ids[0]].to('cuda')
    profile = profile.unsqueeze(dim=0)

    decoder_input = torch.cat((encoder_out['encoder_out'], profile), dim=1)

    outputs = model.decoder(decoder_input)

    if outputs.detach().cpu().numpy().shape[0] == 1:
        pred_array.extend([outputs.detach().cpu().numpy().squeeze()[1]])
    else:
        pred_array.extend(list(outputs.detach().cpu().numpy()[:, 1].squeeze()))

    # #NOTE: For Urban8k
    # # print("PREDICTION ARRAY SHAPE: ", outputs.detach().cpu().numpy().shape)
    # pred_array.append(outputs.detach().cpu().numpy().squeeze())

    target_array.extend(list(labels.detach().cpu().numpy()))


# res['assessment_result'] = pred_array

# res.to_csv('results.csv', index=False)

print(evaluate(pred_array, target_array))
