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
parser.add_argument('-w', '--workspace', type=str, required=True)
parser.add_argument('-u', '--user_name', type=str, required=True)
args = options.parse_args_and_arch(parser)

args.run = get_run(args)
args.path = f'outputs/{args.run.name}/checkpoints/checkpoint_best.pt'

# Setup task
task = tasks.setup_task(args)

task.cfg.profiling = True
task.cfg.profiles_path = args.profile_path
# Load model
print(f' | loading model from ${args.path}')
models, _model_args = checkpoint_utils.load_model_ensemble([args.path],  arg_overrides={'data': 'orig_2048_128_aicovidvn_fold4', 'w2v_path': 'outputs/2022-04-19/21-32-58/checkpoints/checkpoint_best.pt'}, task=task)
model = models[0].cuda()

print(model)

profiles = torch.load(args.profile_path)

if args.input_file == 'majority_small':
    maj_inp, maj_out, maj_id = load_fairseq_dataset('majority_small', 'spectrum_', ['test'], profiling=True)
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
    test_set_id = np.array(maj_id)
elif args.input_file == 'majority':
    maj_inp, maj_out, maj_id = load_fairseq_dataset('majority_clean', 'spectrum_', ['test'], profiling=True)
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
    test_set_id = np.array(maj_id)
elif args.input_file == 'minority':
    min_inp, min_out, min_id = load_fairseq_dataset('profile_2048_128', 'spectrum_', ['test'], profiling=True)
    test_set_inp = [*min_inp]
    test_set_out = np.array(min_out)
    test_set_id = np.array(min_id)


test_dataset = ProfileDataset(test_set_id, test_set_inp, test_set_out)
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=profile_collate_fn)

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
    try:
        profile = profiles[ids[0]].to('cuda')
    except:
        print(ids[0], profiles[ids[0]])
    profile = profile.unsqueeze(dim=0)

    decoder_input = torch.cat((encoder_out['encoder_out'], profile), dim=1)
    # decoder_input = encoder_out['encoder_out'] + encoder_out['encoder_out'] * profile

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
