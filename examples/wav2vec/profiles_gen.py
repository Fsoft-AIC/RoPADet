import torch
from torch.utils.data import DataLoader
from fairseq import checkpoint_utils, options, tasks
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils import load_fairseq_dataset, ProfileDataset, profile_collate_fn


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

print("MODEL TYPE: ", type(model))

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

# test_set_inp = [*coughvid_test_set_inp, *aicvvn_test_set_inp, *coswara_test_set_inp]
# test_set_out = np.concatenate((coughvid_test_set_out, aicvvn_test_set_out, coswara_test_set_out))

# NOTE: new loading method for COVID-19 dataset

min_inp, min_out, min_id = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/profile_2048_128', '/media/Z/tungtk2/aicv115m_api_template/', ['train', 'valid', 'test'], profiling=True)
# test_set_inp = [*min_inp]
# test_set_out = np.array(min_out)
# test_set_id = np.array(min_id)


maj_inp, maj_out, maj_id = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/majority_small', '/media/Z/tungtk2/aicv115m_api_template/data', ['train', 'valid', 'test'], profiling=True)
test_set_inp = [*maj_inp, *min_inp]
test_set_out = np.concatenate((maj_out, min_out))
test_set_id = np.concatenate((maj_id, min_id))

# NOTE: For ICBHI dataset
# icbhi_inp, icbhi_out, icbhi_id = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32_official_unnormalized_unresized_augmented', '/media/Z/tungtk2/RespireNet', splits=['train', 'valid'], profiling=True)
# test_set_inp = [*icbhi_inp]
# test_set_out = np.array(icbhi_out)
# test_set_id = np.array(icbhi_id)

test_dataset = ProfileDataset(test_set_id, test_set_inp, test_set_out)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=profile_collate_fn)

model.eval()

profiles = defaultdict(list)

for ids, inputs, labels, lengths in tqdm(dataloader):
    inputs = inputs.to('cuda', dtype=torch.float)
    labels = labels.to('cuda')
    lengths = lengths.unsqueeze(dim=1).to('cuda')

    with torch.no_grad():
        outs = model(inputs, lengths, features_only=True)

    outputs = outs['x']
    outputs = outputs.squeeze()
    if len(outputs.shape) != 1:
        outputs = torch.mean(outputs, dim=0)
    # print(outputs)
    # NOTE: only works for batch size = 1
    profiles[ids[0]].append(outputs)

count = 0
for key, val in profiles.items():
    print(key, len(val))
    count += len(val)
    profiles[key] = torch.mean(torch.stack(val), dim=0)
    print(profiles[key].shape)
print(count)

SAVE_PATH = args.path[:args.path.rfind('/')] + '/profile.pt'
print("Saving profiles to: ", SAVE_PATH)

torch.save(profiles, SAVE_PATH)
