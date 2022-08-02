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
args = options.parse_args_and_arch(parser)

if args.path is None:
    args.run = get_run(args)
    args.path = f'/media/data/tungtk2/fairseq/outputs/{args.run.name}/checkpoints/checkpoint_best.pt'


# Setup task
task = tasks.setup_task(args)
# Load model
print(f' | loading model from ${args.path}')
# models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/SSD/tungtk2/fairseq/data/orig_2048_128', 'w2v_path': '/media/SSD/tungtk2/fairseq/outputs/2022-03-07/08-30-20/checkpoints/checkpoint_best.pt'})
# models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': '/media/SSD/tungtk2/fairseq/data/orig_2048_128_aicovidvn_fold4', 'w2v_path': '/media/Z/tungtk2/fairseq/outputs/2022-03-07/08-30-20/checkpoints/checkpoint_best.pt'})
models, _model_args = checkpoint_utils.load_model_ensemble([args.path], arg_overrides={'data': 'data/ICBHI_256_32_respirenetsplit_unnormalized_unresized_augmented_fold4', 'w2v_path': '/media/Z/tungtk2/fairseq/outputs/2022-06-22/14-26-59/checkpoints/checkpoint_best.pt'})
model = models[0].cuda()

print(model)

if args.input_file == 'majority':
    maj_inp, maj_out = load_fairseq_dataset('data/AandS_2048_128_fold4', '/media/Z/tungtk2/aicv115m_api_template/', ['test'])
    test_set_inp = [*maj_inp]
    test_set_out = np.array(maj_out)
elif args.input_file == 'minority':
    min_inp, min_out, _ = load_fairseq_dataset('data/profile_2048_128', '/media/Z/tungtk2/aicv115m_api_template/', ['test'], profiling=True)
    test_set_inp = [*min_inp]
    test_set_out = np.array(min_out)
elif args.input_file.startswith('w2g_a2'):
    w2g_a2_inp, w2g_a2_out = load_fairseq_dataset(f'data/{args.input_file}', '/media/Z/tungtk2/aicv115m_api_template/', ['test'])
    test_set_inp = [*w2g_a2_inp]
    test_set_out = np.array(w2g_a2_out)
elif args.input_file.startswith('w2g_a4'):
    w2g_a4_inp, w2g_a4_out = load_fairseq_dataset(f'data/{args.input_file}', '/media/Z/tungtk2/aicv115m_api_template/', ['test'])
    test_set_inp = [*w2g_a4_inp]
    test_set_out = np.array(w2g_a4_out)
elif args.input_file.startswith('ICBHI'):
    icbhi_inp, icbhi_out, _ = load_fairseq_dataset(f'data/{args.input_file}', '/media/Z/tungtk2/RespireNet', splits=['valid'], profiling=True)
    test_set_inp = [*icbhi_inp]
    test_set_out = np.array(icbhi_out)

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

    if args.input_file.startswith("w2g_a4"):
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


if args.input_file.startswith("w2g_a4"):
    ovr_evaluate(pred_array, target_array, args)
elif args.input_file.startswith('ICBHI'):
    icbhi_evaluate(pred_array, target_array, args)
else:
    evaluate(pred_array, target_array, args)
