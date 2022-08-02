import os

# os.system('CUDA_VISIBLE_DEVICES="1" fairseq-hydra-train task.data=/media/data/tungtk2/fairseq/data/profile_2048_128 model.w2v_path=/media/data/tungtk2/fairseq/outputs/2022-07-25/00-11-55/checkpoints/checkpoint_best.pt task.profiling=True task.profiles_path=/media/data/tungtk2/fairseq/outputs/2022-07-26/22-38-53/checkpoints/profile.pt common.seed=1 optimization.lr=[5e-5] --config-dir examples/wav2vec/config/finetuning --config-name base_1h')

# for run_id in ['1593mesz', 'db0h03qi', 'aemmwjga', '1epbs5jn', '3m7wawp8']:
#     os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier_profile.py /media/data/tungtk2/fairseq/data --labels label --input_file minority --run_id {run_id}')
#     os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_profile_without_info.py /media/data/tungtk2/fairseq/data --labels label --input_file majority --run_id {run_id}')

# for run_id in ['2qxwbbip', '2344midj', '2uez65y6', '1ko971hf']:
#     os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier.py /media/data/tungtk2/fairseq/data --labels label --input_file minority --run_id {run_id}')
#     os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier.py /media/data/tungtk2/fairseq/data --labels label --input_file majority --run_id {run_id}')


# for i in range(5):
#     os.system(f'fairseq-hydra-train task.data=/media/data/tungtk2/fairseq/data/w2g_a4_fold{i} model.w2v_path=/media/data/tungtk2/fairseq/outputs/2022-04-19/21-32-58/checkpoints/checkpoint_best.pt criterion.class_weights=[6.6,0.6,1.5] model.clf_output_dim=3 --config-dir examples/wav2vec/config/finetuning --config-name base_1h')
# for i in range(5):
#     os.system(f'CUDA_VISIBLE_DEVICES="1" fairseq-hydra-train task.data=/media/data/tungtk2/fairseq/data/w2g_a2_fold{i} model.w2v_path=/media/data/tungtk2/fairseq/outputs/2022-04-19/21-32-58/checkpoints/checkpoint_best.pt --config-dir examples/wav2vec/config/finetuning --config-name base_1h')

#'2022-07-30/17-32-47'
# for profile in ['2022-07-29-16-37-33', '2022-07-30/21-03-49', '2022-07-31/00-06-54', '2022-07-31/03-06-52', '2022-07-31/14-57-05', '2022-07-31/17-50-10', '2022-07-31/20-49-17', '2022-07-31/23-56-02', '2022-08-01/03-03-25', '2022-08-01/06-01-07', '2022-08-01/08-50-36', '2022-08-01/11-45-29']:
#     profile = profile.replace('/', '-')
#     # for seed in range(2,6):
#     seed = 1
#     os.system(f'CUDA_VISIBLE_DEVICES="1" fairseq-hydra-train task.data=../../../data/profile_2048_128 model.w2v_path=../../../outputs/2022-07-25/00-11-55/checkpoints/checkpoint_best.pt task.profiling=True task.profiles_path=../../../outputs/profile_all_{profile}.pt common.seed={seed} optimization.lr=[5e-5] --config-dir examples/wav2vec/config/finetuning --config-name base_1h')
#     os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier_profile.py /media/data/tungtk2/fairseq/data --labels label --input_file minority --profile_path outputs/profile_all_{profile}.pt')
#     # os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier_profile.py /media/data/tungtk2/fairseq/data --labels label --input_file majority_small --profile_path outputs/profile_all_{profile}.pt')


run_names = []
run_names.append(['2022-08-01/18-49-15', '2022-08-01/18-40-44', '2022-08-01/18-31-51', '2022-08-01/18-22-49', '2022-08-01/17-24-17'])
run_names.append(['2022-08-01/19-30-04', '2022-08-01/19-20-51', '2022-08-01/19-11-45', '2022-08-01/19-02-30', '2022-08-02/03-57-56'])
run_names.append(['2022-08-01/20-42-46', '2022-08-01/20-33-02', '2022-08-01/20-24-22', '2022-08-01/20-15-31', '2022-08-01/19-43-32'])
run_names.append(['2022-08-01/21-23-40', '2022-08-01/21-12-56', '2022-08-01/21-02-18', '2022-08-01/20-52-42', '2022-08-02/04-04-36'])
run_names.append(['2022-08-01/22-06-03', '2022-08-01/21-54-40', '2022-08-01/21-44-35', '2022-08-01/21-34-36', '2022-08-02/04-12-23'])
run_names.append(['2022-08-01/22-44-49', '2022-08-01/22-36-26', '2022-08-01/22-28-15', '2022-08-01/22-17-03', '2022-08-02/04-20-12'])
run_names.append(['2022-08-01/23-17-10', '2022-08-01/23-08-51', '2022-08-01/23-00-46', '2022-08-01/22-52-32', '2022-08-02/04-27-14'])
run_names.append(['2022-08-01/23-52-40', '2022-08-01/23-43-23', '2022-08-01/23-34-30', '2022-08-01/23-25-40', '2022-08-02/04-34-29'])
run_names.append(['2022-08-02/00-27-09', '2022-08-02/00-18-46', '2022-08-02/00-09-56', '2022-08-02/00-01-36', '2022-08-02/04-42-14'])
run_names.append(['2022-08-02/01-02-32', '2022-08-02/00-53-10', '2022-08-02/00-44-19', '2022-08-02/00-35-35', '2022-08-02/04-50-13'])
run_names.append(['2022-08-02/01-37-56', '2022-08-02/01-29-07', '2022-08-02/01-20-20', '2022-08-02/01-11-32', '2022-08-02/04-57-48'])
run_names.append(['2022-08-02/02-22-52', '2022-08-02/02-14-10', '2022-08-02/02-02-56', '2022-08-02/01-49-15', '2022-08-02/05-04-56'])
run_names.append(['2022-08-02/02-56-47', '2022-08-02/02-47-55', '2022-08-02/02-39-36', '2022-08-02/02-31-23', '2022-08-02/05-12-07'])
profiles = ['2022-07-27/18-57-07', '2022-07-29/16-37-33', '2022-07-30/17-32-47', '2022-07-30/21-03-49', '2022-07-31/00-06-54', '2022-07-31/03-06-52', '2022-07-31/14-57-05', '2022-07-31/17-50-10', '2022-07-31/20-49-17', '2022-07-31/23-56-02', '2022-08-01/03-03-25', '2022-08-01/06-01-07', '2022-08-01/08-50-36']
for curr_run_names, profile in zip(run_names, profiles):
    profile = profile.replace('/', '-')
    for run_name in curr_run_names:
        print('RUN NAME: ', run_name)
        # os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier_profile.py /media/data/tungtk2/fairseq/data --labels label --input_file majority_small --profile_path outputs/profile_all_{profile}.pt --run_name {run_name}')
        # os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier_profile.py /media/data/tungtk2/fairseq/data --labels label --input_file minority --profile_path outputs/profile_all_{profile}.pt --run_name {run_name}')
        os.system(f'CUDA_VISIBLE_DEVICES="1" python examples/wav2vec/eval_classifier_profile.py /media/data/tungtk2/fairseq/data --labels label --input_file majority --profile_path outputs/profile_all_{profile}.pt --run_name {run_name}')
