from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score, accuracy_score, precision_recall_fscore_support, brier_score_loss
import os
import numpy as np
import torch
from torch.utils.data import Dataset

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


def evaluate(ensem_preds, targets, args):
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

    # for th in np.arange(0.0, 0.6, 0.01):
    #     pred = (ensem_preds > th).astype(int)
    #     score = f1_score(targets, pred)
    #     if score > best_score:
    #         best_th = th
    #         best_score = score
    f = open('output.txt', 'a')

    print('DATASET: ', args.input_file, file=f)

    start_id = args.path.find('-')-4
    end_id = args.path.rfind('-')+3
    print(args.path[start_id:end_id], file=f)

    print(f"\nAUC score: {roc_auc_score(targets, ensem_preds):12.4f}", file=f)
    # print(f"Best threshold {best_th:12.4f}")

    # preds = (ensem_preds > best_th).astype(int)
    # print(classification_report(targets, preds, digits=3))

    preds = (np.array(ensem_preds) > 0.5).astype(int)

    cm1 = confusion_matrix(targets, preds)
    print('\nConfusion Matrix : \n', cm1, file=f)
    precision, recall, f1 = compute_metrics(cm1)
    
    print('\n=============', file=f)
    print (f'Precision    : {precision:12.4f}', file=f)

    print(f'Recall : {recall:12.4f}', file=f)

    print(f'F1 Score : {f1:12.4f}', file=f)

    total1=sum(sum(cm1))

    print('\n=============', file=f)
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print (f'Accuracy    : {accuracy1:12.4f}', file=f)
    brier_score = brier_score_loss(targets, preds)
    print(f'Brier Score : {brier_score:12.4f}', file=f)
    print('\n', file=f)


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


def load_fairseq_dataset(dir_path, orig_dir, splits=['train', 'valid', 'test'], profiling=False):
    '''
    dir_path: directory to fairseq meta data files
    orig_dir: original directory that stored the spectrum
    '''
    feats = []
    labels = []
    profile_ids = []
    for split in splits:
        with open(os.path.join(dir_path, f'{split}.tsv')) as f, open(os.path.join(dir_path, f'{split}.label')) as label:
            for line, label in zip(f.read().split('\n')[1:-1], label.read().split('\n')):
                if profiling:
                    file_path, _, _, profile_id = line.split('\t')
                    profile_ids.append(profile_id)
                else:
                    file_path, _, _ = line.split('\t')
                spec = np.load(os.path.join(orig_dir, file_path))
                feats.append(spec)
                labels.append(int(label))

    if profiling:
        return feats, labels, profile_ids
    else:
        return feats, labels


def profile_collate_fn(data):
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


class ProfileDataset(Dataset):
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
