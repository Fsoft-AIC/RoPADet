from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score, accuracy_score, precision_recall_fscore_support, brier_score_loss
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wandb
from tqdm import tqdm


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


def get_run(args):
    api = wandb.Api()

    workspaces = ['wav2vec2_covid', 'wav2vec2_icbhi', 'wav2vec2_covid_profile_self_training', 'wav2vec2_covid_pretrain', 'wav2vec2_icbhi_pretrain']
    for workspace in workspaces:
        if hasattr(args,'run_id') and args.run_id is not None:
            run = api.run(f'snp-robustness/{workspace}/{args.run_id}')
        elif hasattr(args,'run_name') and args.run_name is not None:
            runs = api.runs(f'snp-robustness/{workspace}')
            for curr_run in runs:
                if curr_run.name == args.run_name:
                    run = curr_run
                    break
        else:
            runs = api.runs(f'snp-robustness/{workspace}')
            run = runs[0]

        return run

def update_run(run, k, v):
    if (isinstance(run.summary, wandb.old.summary.Summary) and k not in run.summary):
        run.summary._root_set(run.summary._path, [(k, {})])
    run.summary[k] = v


def icbhi_evaluate(final_predicts, final_targets, args):
    def get_score(hits, counts):
        se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3])
        sp = hits[0] / counts[0]
        print(f"SENSE: {se:.4f}", file=f)
        print(f"SPEC: {sp:.4f}", file=f)
        sc = (se+sp) / 2.0
        return sc

    f = open('output.txt', 'a')

    print('DATASET: ', args.input_file, file=f)

    start_id = args.path.find('-')-4
    end_id = args.path.rfind('-')+3
    print(args.path[start_id:end_id], file=f)

    class_hits = [0.0, 0.0, 0.0, 0.0] # normal, crackle, wheeze, both
    class_counts = [0.0, 0.0, 0.0+1e-7, 0.0+1e-7] # normal, crackle, wheeze, both
    for idx in range(len(final_targets)):
        class_counts[final_targets[idx]] += 1.0
        if final_predicts[idx] == final_targets[idx]:
            class_hits[final_targets[idx]] += 1.0

    print(class_counts)
    print("Accuracy: ", accuracy_score(final_targets, final_predicts), file=f)
    conf_matrix = confusion_matrix(final_targets, final_predicts)
    print(conf_matrix, file=f)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    print(f"Classwise Scores: {conf_matrix.diagonal()}", file=f)
    print(f"ICBHI score: {get_score(class_hits, class_counts):.4f}", file=f)


def ovr_evaluate(final_predicts, final_targets, args):
    f = open('output.txt', 'a')

    print('DATASET: ', args.input_file, file=f)

    start_id = args.path.find('-')-4
    end_id = args.path.rfind('-')+3
    print(args.path[start_id:end_id], file=f)

    auc = roc_auc_score(final_targets, final_predicts, average='micro')
    print(f'\nAUC score: {auc:12.4f}', file=f)


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
    f = open('output.txt', 'a')

    print('DATASET: ', args.input_file, file=f)

    start_id = args.path.find('-')-4
    end_id = args.path.rfind('-')+3
    print(args.path[start_id:end_id], file=f)

    auc = roc_auc_score(targets, ensem_preds)
    print(f"\nAUC score: {auc:12.4f}", file=f)

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
    brier_score = brier_score_loss(targets, ensem_preds)
    print(f'Brier Score : {brier_score:12.4f}', file=f)
    print('\n', file=f)

    run = args.run
    update_run(run, f'test_{args.input_file}', dict())
    run.summary[f'test_{args.input_file}']['auc'] = round(auc, 4)
    run.summary[f'test_{args.input_file}']['precision'] = round(precision, 4)
    run.summary[f'test_{args.input_file}']['recall'] = round(recall, 4)
    run.summary[f'test_{args.input_file}']['f1_score'] = round(f1, 4)
    run.summary[f'test_{args.input_file}']['accuracy'] = round(accuracy1, 4)
    run.summary[f'test_{args.input_file}']['brier_score'] = round(brier_score, 4)
    run.summary.update()


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
            for line, label in tqdm(zip(f.read().split('\n')[1:-1], label.read().split('\n'))):
                if profiling:
                    try:
                        file_path, _, _, profile_id = line.split('\t')
                    except Exception as _:
                        file_path, _, _ = line.split('\t')
                        profile_id = file_path[file_path.rfind('/') + 1: -17]
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

def load_fairseq_dataset_clean(dir_path, orig_dir, splits=['train', 'valid', 'test'], profiling=False):
    '''
    dir_path: directory to fairseq meta data files
    orig_dir: original directory that stored the spectrum
    '''
    cleaning_profiles = ['bad_cough_2021-08-18T08:43:58.113Z', 'good_cough_2021-08-02T02:20:03.507Z', 'good_cough_2021-09-06T12:52:43.247Z', 'good_cough_2021-09-06T05:42:10.955Z', 'good_cough_2021-09-23T09:07:09.841Z', 'good_cough_2021-09-12T02:55:10.157Z', 'good_cough_2021-08-20T16:27:48.296Z', 'isofh_undefined-1637216183.504766', 'isofh_undefined-1642222653.2569923', 'isofh_undefined-1645671757.060289', 'isofh_undefined-1645681104.1764894', 'isofh_undefined-1645758127.0673249', 'isofh_undefined-1647192523.4555895', 'isofh_undefined-1636079123.4377832', 'isofh_undefined-1644327539.345447', 'isofh_undefined-1633924005.3468943', 'isofh_undefined-1636877044.0882564', 'isofh_undefined-1636960952.7196362', 'isofh_undefined-1644507712.6994777', 'isofh_undefined-1644558767.9239337', 'isofh_undefined-1645000102.4192054', 'isofh_undefined-1645152964.4192548', 'isofh_undefined-1634131789.5582144', 'isofh_undefined-1634282003.54855', 'isofh_undefined-1647307289.5514479', 'isofh_undefined-1639401508.6477778', 'isofh_undefined-1634905349.1400065', 'isofh_undefined-1646214697.4001377', 'isofh_undefined-1646272200.5174289', 'isofh_undefined-1635680127.5080094', 'isofh_undefined-1635719184.5851254']
    feats = []
    labels = []
    profile_ids = []
    for split in splits:
        with open(os.path.join(dir_path, f'{split}.tsv')) as f, open(os.path.join(dir_path, f'{split}.label')) as label:
            for line, label in zip(f.read().split('\n')[1:-1], label.read().split('\n')):
                if profiling:
                    file_path, _, _, profile_id = line.split('\t')
                    if profile_id in cleaning_profiles:
                        profile_ids.append(profile_id)
                else:
                    file_path, _, _ = line.split('\t')
                if profile_id in cleaning_profiles:
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
