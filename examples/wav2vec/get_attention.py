import os
import torch
from torch.utils.data import Dataset, DataLoader
from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data import MelAudioDataset, AddTargetDataset, Dictionary
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score, accuracy_score, precision_recall_fscore_support
from collections import defaultdict
from tqdm import tqdm

def load_dataset(X, file_path, dir_path, label, id, offset=4):
    X[file_path] = X[file_path].apply(lambda x: dir_path + str(x))
    if offset == 0:
        feats = list(X[file_path].apply(lambda x: np.load(x + '_mel_2048_128.npy')).values)
    else:
        feats = list(X[file_path].apply(lambda x: np.load(x[:-offset] + '_mel_2048_128.npy')).values)

    return feats, X[label].values, X[id].values


def load_fairseq_dataset(dir_path, orig_dir, splits=['train', 'valid', 'test']):
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
                file_path, _, _, profile_id = line.split('\t')
                spec = np.load(os.path.join(orig_dir, file_path))
                feats.append(spec)
                profile_ids.append(profile_id)
                labels.append(int(label))

    return feats, labels, profile_ids


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
parser = options.get_generation_parser(default_task='stft_audio_pretraining')
args = options.parse_args_and_arch(parser)


# Setup task
task = tasks.setup_task(args)
# Load model
print(f' | loading model from ${args.path}')
models, _model_args = checkpoint_utils.load_model_ensemble([args.path])
# models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/SSD/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4'})
model = models[0].cuda()

print(model)

# NOTE: For COVID-19 dataset
# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coughvid/df_fold.csv')
# coughvid_test_set_inp, coughvid_test_set_out = load_dataset(X[X['fold'] == 3], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coughvid/public_dataset/', 'label_covid', offset=5)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_train/public_train_metadata_fold.csv')
# aicvvn_test_set_inp, aicvvn_test_set_out = load_dataset(X, 'uuid', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv115m_final_public_train/public_train_audio_files/', 'assessment_result', offset=0)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/df_fold.csv')
# coswara_test_set_inp, coswara_test_set_out = load_dataset(X, 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/coswara/Coswara-Data_0511/', 'label_covid')

# X_aicovidvn_new_min = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/assets/df_min.csv')
# aicovidvn_new_min_inp, aicovidvn_new_min_out, aicovidvn_new_min_id = load_dataset(X_aicovidvn_new_min, 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/', 'label', 'id')

# X_sounddr_min = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/df_min.csv')
# sounddr_min_inp, sounddr_min_out, sounddr_min_id = load_dataset(X_sounddr_min, 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/', 'label', 'id')

# icbhi_inp, icbhi_out, icbhi_id = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32_official_unnormalized_unresized_augmented', '/media/SSD/tungtk2/RespireNet', splits=['train', 'valid'])
icbhi_inp, icbhi_out, icbhi_id = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32_official_unnormalized_unresized_augmented', '/media/SSD/tungtk2/RespireNet', splits=['valid'])

# test_set_inp = [*coughvid_test_set_inp, *aicvvn_test_set_inp, *coswara_test_set_inp]
# test_set_out = np.concatenate((coughvid_test_set_out, aicvvn_test_set_out, coswara_test_set_out))

test_set_inp = [*icbhi_inp]
test_set_out = np.array(icbhi_out)
test_set_id = np.array(icbhi_id)

test_dataset = MyDataset(test_set_id, test_set_inp, test_set_out)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

model.eval()
# model.encoder.eval()
# model.decoder.eval()
# model = model.encoder

import matplotlib.pyplot as plt
count = 0
attentions = [0]*4

for ids, inputs, labels, lengths in tqdm(dataloader):
    inputs = inputs.to('cuda', dtype=torch.float)
    labels = labels.to('cuda')
    lengths = lengths.unsqueeze(dim=1).to('cuda')

    # if labels.data!=2:
    #     continue

    with torch.no_grad():
        outs = model(inputs, lengths, features_only=True)

    for layer in range(len(outs['layer_results'])):
        if layer==0:
            print(outs['layer_results'][layer][1].shape)
            attentions[count] = outs['layer_results'][layer][1].detach().cpu().numpy().squeeze()
            print(attentions[count].shape, count)
            lst = []
            for i in range(attentions[count].shape[0]):
                shp = attentions[count][i,:,:].shape[0]
                lst.extend([-0.1*np.ones((int(shp/5), shp)), attentions[count][i,:,:]])
            attentions[count] = np.concatenate(lst, axis=0)
            # attentions[count] = attentions[count].reshape(attentions[count].shape[0]*attentions[count].shape[1], attentions[count].shape[2])
        else:
            temp_attention = outs['layer_results'][layer][1].detach().cpu().numpy().squeeze()
            lst = []
            for i in range(temp_attention.shape[0]):
                lst.extend([-0.1*np.ones((int(temp_attention[i,:,:].shape[0]/5), temp_attention[i,:,:].shape[1])), temp_attention[i,:,:]])
            temp_attention = np.concatenate(lst, axis=0)
            # temp_attention = temp_attention.reshape(temp_attention.shape[0]*temp_attention.shape[1], temp_attention.shape[2])
            attentions[count] = np.concatenate((attentions[count], -0.1*np.ones((attentions[count].shape[0], int(attentions[count].shape[0]/20))), temp_attention), axis=1)
    count += 1
    if count == 4:
        break

for i in range(len(attentions)):
    plt.imsave(f'attention_{i}.png', attentions[i], cmap='viridis')
