import torch
from torch.utils.data import DataLoader
from fairseq import checkpoint_utils, options, tasks
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils import load_fairseq_dataset, ProfileDataset, profile_collate_fn, get_run


# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='stft_audio_pretraining')
parser.add_argument('-r', '--run_id', type=str, required=False)
parser.add_argument('-o', '--orig_profile', type=str, required=True)
parser.add_argument('-w', '--workspace', type=str, required=True)
parser.add_argument('-u', '--user_name', type=str, required=True)
args = options.parse_args_and_arch(parser)

if args.path is None:
    args.run = get_run(args)
    args.path = f'outputs/{args.run.name}/checkpoints/checkpoint_best.pt'


# Setup task
task = tasks.setup_task(args)
# Load model
print(f' | loading model from ${args.path}')
models, _model_args = checkpoint_utils.load_model_ensemble([args.path])
model = models[0].cuda()

print(model)

print("MODEL TYPE: ", type(model))


min_inp, min_out, min_id = load_fairseq_dataset('profile_2048_128', 'spectrum_', ['train', 'valid', 'test'], profiling=True)
# test_set_inp = [*min_inp]
# test_set_out = np.array(min_out)
# test_set_id = np.array(min_id)


maj_inp, maj_out, maj_id = load_fairseq_dataset('majority_clean', 'spectrum_', ['train', 'valid', 'test'], profiling=True)
test_set_inp = [*maj_inp, *min_inp]
test_set_out = np.concatenate((maj_out, min_out))
test_set_id = np.concatenate((maj_id, min_id))

# NOTE: For ICBHI dataset
# icbhi_inp, icbhi_out, icbhi_id = load_fairseq_dataset('ICBHI_256_32_official_unnormalized_unresized_augmented', 'spectrum_', splits=['train', 'valid'], profiling=True)
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

# Replace profile in labeled set (minority) with profile of pre-trained model, as the true label
previous_profile = torch.load(args.orig_profile)

count = 0
for key, val in profiles.items():
    print(key, len(val))
    count += len(val)
    if len(val) > 1:
        profiles[key] = previous_profile[key]
    else:
        profiles[key] = torch.mean(torch.stack(val), dim=0)
    if profiles[key].shape != torch.Size([256]):
        print(profiles[key].shape)
print(count)



SAVE_PATH = f'outputs/profile_all_{args.run.name}.pt'
print("Saving profiles to: ", SAVE_PATH)

torch.save(profiles, SAVE_PATH)
