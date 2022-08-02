import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data import MelAudioDataset, AddTargetDataset, Dictionary
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
import pandas as pd
import numpy as np
from utils import load_fairseq_dataset, evaluate, ProfileDataset, profile_collate_fn, get_run


# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='audio_finetuning')
parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('-r', '--run_id', type=str, required=False)
parser.add_argument('-rn', '--run_name', type=str, required=False)
parser.add_argument('-p', '--profile_path', type=str, required=True)
args = options.parse_args_and_arch(parser)

args.run = get_run(args)
args.path = f'/media/data/tungtk2/fairseq/outputs/{args.run.name}/checkpoints/checkpoint_best.pt'

# Setup task
task = tasks.setup_task(args)

task.cfg.profiling = True
task.cfg.profiles_path = args.profile_path
# Load model
print(f' | loading model from ${args.path}')
# models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/SSD/tungtk2/fairseq/data/orig_2048_128', 'w2v_path': '/media/SSD/tungtk2/fairseq/outputs/2022-03-07/08-30-20/checkpoints/checkpoint_best.pt'})
models, _model_args = checkpoint_utils.load_model_ensemble([args.path],  arg_overrides={'data': '/media/data/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4', 'w2v_path': '/media/data/tungtk2/fairseq/outputs/2022-04-19/21-32-58/checkpoints/checkpoint_best.pt'}, task=task)
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

'''
# NOTE: For ICBHI dataset
profiles = torch.load('/media/data/tungtk2/fairseq/outputs/2022-06-22/14-26-59/checkpoints/profile.pt')
icbhi_inp, icbhi_out, icbhi_id = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32_official_unnormalized_unresized_augmented', '/media/SSD/tungtk2/RespireNet', splits=['test'], profiling=True)
# profiles = torch.load('/media/SSD/tungtk2/fairseq/outputs/2022-05-20/22-37-59/checkpoints/profile.pt')
# icbhi_inp, icbhi_out, icbhi_id = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32', '/media/SSD/tungtk2/RespireNet', splits=['test'], profiling=True)
test_set_inp = [*icbhi_inp]
test_set_out = np.array(icbhi_out)
test_set_id = np.array(icbhi_id)
# ENDNOTE
'''

# NOTE: For Covid dataset
# profiles = torch.load('/media/data/tungtk2/fairseq/outputs/2022-07-26/22-38-53/checkpoints/profile.pt')
# profiles = torch.load('/media/data/tungtk2/fairseq/outputs/2022-07-25/00-11-55/checkpoints/profile.pt')
profiles = torch.load(args.profile_path)

if args.input_file == 'majority_small':
    maj_inp, maj_out, maj_id = load_fairseq_dataset('/media/data/tungtk2/fairseq/data/majority_small', '/media/data/tungtk2/aicv115m_api_template/data', ['test'], profiling=True)
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
    test_set_id = np.array(maj_id)
elif args.input_file == 'majority':
    maj_inp, maj_out, maj_id = load_fairseq_dataset('/media/data/tungtk2/fairseq/data/majority_clean', '/media/data/tungtk2/aicv115m_api_template/data', ['test'], profiling=True)
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
    test_set_id = np.array(maj_id)
elif args.input_file == 'minority':
    min_inp, min_out, min_id = load_fairseq_dataset('/media/data/tungtk2/fairseq/data/profile_2048_128', '/media/data/tungtk2/aicv115m_api_template/', ['test'], profiling=True)
    test_set_inp = [*min_inp]
    test_set_out = np.array(min_out)
    test_set_id = np.array(min_id)

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/assets/df_min.csv')
# aicovidvn_new_min_inp, aicovidvn_new_min_out, aicovidvn_new_min_id = load_dataset(X[X['fold'] == 1], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/aicv_new/', 'label', 'id')

# X = pd.read_csv('/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/df_min.csv')
# sounddr_min_inp, sounddr_min_out, sounddr_min_id = load_dataset(X[X['fold'] == 1], 'file_path', '/host/ubuntu/tungtk2/aicovid/aicv115m_api_template/data/sounddr/', 'label', 'id')

# test_set_inp = [*sounddr_min_inp, *aicovidvn_new_min_inp]
# test_set_out = np.concatenate((sounddr_min_out, aicovidvn_new_min_out))
# test_set_id = np.concatenate((sounddr_min_id, aicovidvn_new_min_id))

# # NOTE: For urban8k
# X = pd.read_csv('/media/SSD/tungtk2/UrbanSound8K/metadata/UrbanSound8K.csv')
# X['file_path'] = X.apply(lambda x: f"/media/SSD/tungtk2/UrbanSound8K/audio/fold{ x['fold'] }/{ x['slice_file_name'] }", axis=1)
# urban8k_test_set_inp, urban8k_test_set_out = load_dataset(X[X['fold'] == 10], 'file_path', '', 'classID')

# test_set_inp = [*urban8k_test_set_inp]
# test_set_out = np.array(urban8k_test_set_out)


test_dataset = ProfileDataset(test_set_id, test_set_inp, test_set_out)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=profile_collate_fn)

pred_array = []
target_array = []
model.eval()
model.encoder.eval()
model.decoder.eval()

# for ids, inputs, labels, lengths in dataloader:
#     # NOTE: only works for batch size = 1
#     try:
#         profile = profiles[ids[0]].to('cuda')
#     except:
#         print(ids[0], profiles[ids[0]])
#         continue


for ids, inputs, labels, lengths in dataloader:
    inputs = inputs.to('cuda', dtype=torch.float)
    labels = labels.to('cuda')
    lengths = lengths.unsqueeze(dim=1).to('cuda')

    encoder_out = model.encoder(inputs, lengths)
    encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'], dim=1)

    # NOTE: only works for batch size = 1
    try:
        profile = profiles[ids[0]].to('cuda')
    except:
        print(ids[0], profiles[ids[0]])
    profile = profile.unsqueeze(dim=0)

    decoder_input = torch.cat((encoder_out['encoder_out'], profile), dim=1)
    # decoder_input = encoder_out['encoder_out'] + encoder_out['encoder_out'] * profile

    # NOTE: unlike on Majority_small, on Minority, with softmax produce better output
    outputs = F.softmax(model.decoder(decoder_input), dim=1)
    # outputs = model.decoder(decoder_input)

    # # NOTE: For ICBHI dataset:
    # pred_array.append(int(outputs.argmax(dim=1).detach().cpu().numpy().squeeze()))
    if outputs.detach().cpu().numpy().shape[0] == 1:
        pred_array.extend([outputs.detach().cpu().numpy().squeeze()[1]])
    else:
        pred_array.extend(list(outputs.detach().cpu().numpy()[:, 1].squeeze()))

    target_array.extend(list(labels.detach().cpu().numpy()))


print(evaluate(pred_array, target_array, args))

'''
def get_score(hits, counts):
    se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3])
    sp = hits[0] / counts[0]
    print(f"SENSE: {se:.4f}")
    print(f"SPEC: {sp:.4f}")
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
print(confusion_matrix(target_array, pred_array))
print(f"{get_score(class_hits, class_counts):.4f}")
'''
