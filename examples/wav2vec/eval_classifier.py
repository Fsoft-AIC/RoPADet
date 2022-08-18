import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fairseq import checkpoint_utils, options, tasks
import numpy as np
from utils import load_fairseq_dataset, evaluate, MyDataset, collate_fn, get_run, ovr_evaluate, icbhi_evaluate

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='audio_finetuning')
parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('-r', '--run_id', type=str, required=False)
parser.add_argument('-rn', '--run_name', type=str, required=False)
parser.add_argument('-w', '--workspace', type=str, required=True)
parser.add_argument('-u', '--user_name', type=str, required=True)
args = options.parse_args_and_arch(parser)

args.run = get_run(args)
args.path = f'outputs/{args.run.name}/checkpoints/checkpoint_best.pt'


# Setup task
task = tasks.setup_task(args)
# Load model
print(f' | loading model from ${args.path}')
models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': 'orig_2048_128_coswara_fold4', 'w2v_path': 'outputs/2022-03-07/08-30-20/checkpoints/checkpoint_best.pt'})
model = models[0].cuda()

print(model)

if args.input_file == 'majority_small':
    maj_inp, maj_out, _ = load_fairseq_dataset('majority_small', 'spectrum_', ['test'], profiling=True)
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
elif args.input_file == 'majority':
    maj_inp, maj_out, _ = load_fairseq_dataset('majority_clean', 'spectrum_', ['test'], profiling=True)
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
elif args.input_file == 'minority':
    min_inp, min_out, _ = load_fairseq_dataset('profile_2048_128', 'spectrum_', ['test'], profiling=True)
    test_set_inp = [*min_inp]
    test_set_out = np.array(min_out)
elif args.input_file.startswith('ICBHI'):
    icbhi_inp, icbhi_out, _ = load_fairseq_dataset(f'{args.input_file}', 'spectrum_', splits=['valid'], profiling=True)
    test_set_inp = [*icbhi_inp]
    test_set_out = np.array(icbhi_out)
else:
    inp, out = load_fairseq_dataset(f'{args.input_file}', 'spectrum_', splits=['test'])
    test_set_inp = [*inp]
    test_set_out = np.array(out)

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


for inputs, labels, lengths in dataloader:
    inputs = inputs.to('cuda', dtype=torch.float)
    labels = labels.to('cuda')
    lengths = lengths.unsqueeze(dim=1).to('cuda')

    encoder_out = model.encoder(inputs, lengths)
    encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'], dim=1)
    outputs = model.decoder(encoder_out['encoder_out'])
    outputs = F.softmax(outputs, dim=1)

    if args.input_file.startswith("3class"):
        predicts = outputs.detach().cpu().numpy().ravel()
        raw_targets = labels.detach().cpu().numpy()
        targets =  np.zeros((raw_targets.size, 3))
        targets[np.arange(raw_targets.size),raw_targets] = 1
        targets = targets.ravel()
        target_array.extend(targets)
        pred_array.extend(predicts)
    elif args.input_file.startswith('ICBHI'):
        pred_array.append(int(outputs.argmax(dim=1).detach().cpu().numpy().squeeze()))
        target_array.extend(list(labels.detach().cpu().numpy()))
    else:
        if outputs.detach().cpu().numpy().shape[0] == 1:
            pred_array.extend([outputs.detach().cpu().numpy().squeeze()[1]])
        else:
            pred_array.extend(list(outputs.detach().cpu().numpy()[:, 1].squeeze()))
        target_array.extend(list(labels.detach().cpu().numpy()))


if args.input_file.startswith("3class"):
    ovr_evaluate(pred_array, target_array, args)
elif args.input_file.startswith('ICBHI'):
    icbhi_evaluate(pred_array, target_array, args)
else:
    evaluate(pred_array, target_array, args)
