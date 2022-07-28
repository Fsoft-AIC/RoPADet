import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fairseq import checkpoint_utils, options, tasks
import numpy as np
from utils import load_fairseq_dataset, evaluate, MyDataset, collate_fn, get_run

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='audio_finetuning')
parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('-r', '--run_id', type=str, required=False)
args = options.parse_args_and_arch(parser)

if args.path is None:
    args.run = get_run(args)
    args.path = f'/media/data/tungtk2/fairseq/outputs/{args.run.name}/checkpoints/checkpoint_best.pt'


# Setup task
task = tasks.setup_task(args)
# Load model
print(f' | loading model from ${args.path}')
# models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/SSD/tungtk2/fairseq/data/orig_2048_128', 'w2v_path': '/media/SSD/tungtk2/fairseq/outputs/2022-03-07/08-30-20/checkpoints/checkpoint_best.pt'})
models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/data/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4', 'w2v_path': '/media/data/tungtk2/fairseq/outputs/2022-03-07/08-30-20/checkpoints/checkpoint_best.pt'})
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

# X = pd.read_csv('/media/data/tungtk2/aicv115m_api_template/data/coswara/df_fold.csv')
# coswara_test_set_inp, coswara_test_set_out = load_dataset(X[X['fold'] == 3], 'file_path', '/media/data/tungtk2/aicv115m_api_template/data/coswara/Coswara-Data_0511/', 'label_covid', offset=4)


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
# icbhi_inp, icbhi_out, _ = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32_official_unnormalized_unresized_augmented', '/media/SSD/tungtk2/RespireNet', splits=['test'])
# icbhi_inp, icbhi_out, _ = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32_official_unnormalized_augmented', '/media/SSD/tungtk2/RespireNet', splits=['test'])
# icbhi_inp, icbhi_out, _ = load_fairseq_dataset('/media/SSD/tungtk2/fairseq/data/ICBHI_256_32', '/media/SSD/tungtk2/RespireNet', splits=['valid'])

# test_set_inp = [*coswara_test_set_inp]
# test_set_out = np.array(coswara_test_set_out)

# test_set_inp = [*coughvid_test_set_inp, *aicvvn_test_set_inp, *coswara_test_set_inp, *sounddr_test_set_inp, *aicvvn_new_test_set_inp]
# test_set_out = np.concatenate((coughvid_test_set_out, aicvvn_test_set_out, coswara_test_set_out, sounddr_test_set_out, aicvvn_new_test_set_out))

# test_set_inp = [*sounddr_maj_inp, *aicovidvn_new_maj_inp]
# test_set_out = np.concatenate((sounddr_maj_out, aicovidvn_new_maj_out))

# test_set_inp = [*sounddr_min_inp, *aicovidvn_new_min_inp]
# test_set_out = np.concatenate((sounddr_min_out, aicovidvn_new_min_out))

# test_set_inp = [*icbhi_inp]
# test_set_out = np.array(icbhi_out)

if args.input_file == 'majority_small':
    maj_inp, maj_out = load_fairseq_dataset('/media/data/tungtk2/fairseq/data/AandS_2048_128_fold4', '/media/data/tungtk2/aicv115m_api_template/', ['test'])
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
elif args.input_file == 'majority':
    maj_inp, maj_out = load_fairseq_dataset('/media/data/tungtk2/fairseq/data/large_2048_128_fold4', '/media/data/tungtk2/aicv115m_api_template/', ['test'])
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
elif args.input_file == 'minority':
    min_inp, min_out, _ = load_fairseq_dataset('/media/data/tungtk2/fairseq/data/profile_2048_128', '/media/data/tungtk2/aicv115m_api_template/', ['test'])
    test_set_inp = [*min_inp]
    test_set_out = np.array(min_out)

test_dataset = MyDataset(test_set_inp, test_set_out)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

pred_array = []
target_array = []
model.eval()
model.encoder.eval()
model.decoder.eval()

print("MODEL PARAMETERS: ")
num_params = sum(param.numel() for param in model.parameters())
print(num_params)

pretrained_models, _ = checkpoint_utils.load_model_ensemble(['/media/data/tungtk2/fairseq/outputs/2022-04-19/21-32-58/checkpoints/checkpoint_best.pt'], arg_overrides={'data': '/media/data/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4'})
# pretrained_models, _ = checkpoint_utils.load_model_ensemble(['/media/data/tungtk2/fairseq/outputs/2022-07-25/00-11-55/checkpoints/checkpoint_best.pt'], arg_overrides={'data': '/media/data/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4'})
# pretrained_models, _ = checkpoint_utils.load_model_ensemble(['/media/data/tungtk2/fairseq/outputs/2022-07-26/22-38-53/checkpoints/checkpoint_best.pt'], arg_overrides={'data': '/media/data/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4'})
pretrained_model = pretrained_models[0].cuda()
pretrained_model.eval()

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

    # outputs = F.softmax(model.decoder(encoder_out['encoder_out']), dim=1)

    # ce_predict = outputs.detach().cpu().numpy()
    # ce_target = labels.detach().cpu().numpy()
    # ces.append(cross_entropy(ce_predict, ce_target))

    # NOTE: For ICBHI dataset:
    # pred_array.append(int(outputs.argmax(dim=1).detach().cpu().numpy().squeeze()))

    # NOTE: for evaluating profiling model:
    with torch.no_grad():
        pretrained_out = pretrained_model(inputs, lengths, features_only=True)
    pretrained_output = pretrained_out['x'].squeeze()
    pretrained_output = torch.mean(pretrained_output, dim=0)
    pretrained_output = pretrained_output.unsqueeze(dim=0)
    if len(pretrained_output.shape) == 1:
        pretrained_output = encoder_out['encoder_out']
    decoder_input = torch.cat((encoder_out['encoder_out'], pretrained_output), dim=1)
    outputs = model.decoder(decoder_input)
    outputs = F.softmax(outputs, dim=1)

    if outputs.detach().cpu().numpy().shape[0] == 1:
        pred_array.extend([outputs.detach().cpu().numpy().squeeze()[1]])
    else:
        pred_array.extend(list(outputs.detach().cpu().numpy()[:, 1].squeeze()))

    # #NOTE: For Urban8k
    # # print("PREDICTION ARRAY SHAPE: ", outputs.detach().cpu().numpy().shape)
    # pred_array.append(outputs.detach().cpu().numpy().squeeze())

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
print("Accuracy: ", accuracy_score(target_array, pred_array))
conf_matrix = confusion_matrix(target_array, pred_array)
print(conf_matrix)
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
print("Classwise Scores", conf_matrix.diagonal())
print(f"{get_score(class_hits, class_counts):.4f}")
'''
