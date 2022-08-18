# Personalization for Robust Voice Pathology Detection Systems in Sound Waves

>Artificial intelligence (AI) based voice pathology detection is promising for non-invasive screening and early intervention in healthcare. Nevertheless, AI systems are susceptible covariate shifts in deployment as sound waves have diverse profiles in terms of background noises and human artifacts; moreover, biases in the data and methodological choices can strongly affect the performance in the real-world scenario. Therefore, there is an emerging need for a robust and reliable detection system overcoming these challenges for efficient, cost-effective, and worldwide health monitoring solutions. In this study, we present a novel end-to-end neural model based on Transformers and 1-D temporal convolution neural networks (CNN). The proposed model operates on a fine-grained scale of spectrum representation of sound to effectively extract features from sound data by utilizing the potent capabilities of neural architectures.
>
>Furthermore, we propose a novel personalization strategy to build a profile for each patient or user that first learns user-specific and task-agnostic features, which may pose be harmful as noises for performing the final task, through a contrastive pre-training stage, and then later combine with a feature extractor to perform the downstream tasks. The profiling method does not only improve the overall classification performance but also shines in the setting of covariate shifts that happen between different source and target datasets. Our experimental results on multiple respiratory illnesses classification tasks with real-world datasets show a large margin compared to previous state-of-the-art methods, with up to 2.5\% improvement in AUC score in the scenario of covariate shift with personalization.

## About this implementation

This implementation is based on the [fairseq toolkit](https://github.com/facebookresearch/fairseq).
Our main source code was written in the following directory:
+ [Profile Generation and Model Evaluation](examples/wav2vec)
+ [Spectrum Data Loading](fairseq/data)
+ [Model Personalization and Implementation](fairseq/models/wav2vec)

## Requirements and Installation
Please follow the instructions to [install the framework](https://github.com/facebookresearch/fairseq#getting-started).

Additionally, install [librosa]():
```
pip install librosa soundfile
```

## Training

### Preparing data

Download and place data in *raw_data* directory, and create the corresponding metadatas in *data* directory.
Run:
```
python gen_spectrum.py --metadata_path=$meta_dir_path --profile=$generate_user_id_or_not --num_fft=$window_length --hop_length=$hop_length
```
### Pre-training

Downstream task pre-training:
```
fairseq-hydra-train task.data=$pre-train_data --config-dir examples/wav2vec/config/pretraining --config-name general_pretrain
```

Profile encoder pre-training:
```
fairseq-hydra-train task.data=$pre-train_data --config-dir examples/wav2vec/config/pretraining --config-name discriminative
```

### Profile encoding:

```
python examples/wav2vec/profiles_gen.py data/ --path $model_path
```

### Self-training:

In each iteration, run:
```
python examples/wav2vec/profiles_gen.py data/ --path $teacher_path; CUDA_VISIBLE_DEVICES="3" fairseq-hydra-train task.data=$self-train_data task.profiles_path=$teacher_profile checkpoint.finetune_from_model=$pre-trained_model --config-dir examples/wav2vec/config/finetuning --config-name profile_self_training
```

### Fine-tuning

Without personalization:
```
fairseq-hydra-train task.data=$meta_dir_path model.w2v_path=pretrained_model --config-dir examples/wav2vec/config/finetuning --config-name without_profile
```

With personalization:
```
fairseq-hydra-train task.data=$meta_dir_path model.w2v_path=pretrained_model task.profiling=True task.profiles_path=$profile_path --config-dir examples/wav2vec/config/finetuning --config-name with_profile
```

## Evaluation

Model without personalization:
```
python examples/wav2vec/eval_classifier.py data --labels label --input_file $meta_dir_path
```

Model with personalization:
```
python examples/wav2vec/eval_classifier_profile.py data --labels label --input_file $meta_dir_path --profile_path $profile_path
```
