import os

# os.system('CUDA_VISIBLE_DEVICES="1" fairseq-hydra-train task.data=/media/data/tungtk2/fairseq/data/profile_2048_128 model.w2v_path=/media/data/tungtk2/fairseq/outputs/2022-07-25/00-11-55/checkpoints/checkpoint_best.pt task.profiling=True task.profiles_path=/media/data/tungtk2/fairseq/outputs/2022-07-26/22-38-53/checkpoints/profile.pt common.seed=1 optimization.lr=[5e-5] --config-dir examples/wav2vec/config/finetuning --config-name base_1h')

for run_id in ['1593mesz', 'db0h03qi', 'aemmwjga', '1epbs5jn', '3m7wawp8']:
    os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier_profile.py /media/data/tungtk2/fairseq/data --labels label --input_file minority --run_id {run_id}')
    os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_profile_without_info.py /media/data/tungtk2/fairseq/data --labels label --input_file majority --run_id {run_id}')

# for run_id in ['2qxwbbip', '2344midj', '2uez65y6', '1ko971hf']:
#     os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier.py /media/data/tungtk2/fairseq/data --labels label --input_file minority --run_id {run_id}')
#     os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier.py /media/data/tungtk2/fairseq/data --labels label --input_file majority --run_id {run_id}')
