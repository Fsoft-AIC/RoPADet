import os
import torch
from torch.utils.data import Dataset, DataLoader
from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data import MelAudioDataset, AddTargetDataset, Dictionary
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score


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


def load_dataset(X, file_path, dir_path, label):
    X[file_path] = X[file_path].apply(lambda x: dir_path + str(x))
    feats = list(X[file_path].apply(lambda x: np.load(x[:-4] + '_mel.npy')).values)
    # feats = np.stack(feats, axis=0)

    return feats, X[label].values


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
models, _model_args = checkpoint_utils.load_model_ensemble([args.path])
model = models[0].cuda()

print(model)

X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/df_fold.csv')
# X = X[X['fold'] == 4]
inp, out = load_dataset(X, 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/Coswara-Data_0511/', 'label_covid')
test_dataset = MyDataset(inp, out)
dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

pred_array = []
target_array = []
model.eval()
for inputs, labels, lengths in dataloader:
    inputs = inputs.to('cuda', dtype=torch.float)
    labels = labels.to('cuda')
    lengths = lengths.unsqueeze(dim=1).to('cuda')

    encoder_out = model.encoder(inputs, lengths)
    encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'], dim=1)
    # print(encoder_out['encoder_out'])
    outputs = model.decoder(encoder_out['encoder_out'])
    # outputs = model(inputs)
    # _, preds = torch.max(outputs, 1)
    print(outputs.detach().cpu().numpy())
    if outputs.detach().cpu().numpy().shape[0] == 1:
        pred_array.extend([outputs.detach().cpu().numpy().squeeze()[1]])
    else:
        pred_array.extend(list(outputs.detach().cpu().numpy()[:, 1].squeeze()))
    target_array.extend(list(labels.detach().cpu().numpy()))

print(evaluate(pred_array, target_array))
