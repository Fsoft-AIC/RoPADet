# Personalization for Robust Voice Pathology Detection Systems in Sound Waves

>Automatic voice pathology detection is promising for non-invasive screening and early intervention using sound signals. Nevertheless, existing methods are susceptible to covariate shifts due to background noises, human voice variations, and data selection biases leading to severe performance degradation in real-world scenarios. Hence, we propose a non-invasive framework that contrastively learns personalization from sound waves as a pre-train and predicts  latent-spaced profile features through semi-supervised learning. It allows all subjects from various distributions (e.g., regionality, gender, age) to benefit from personalized predictions for robust voice pathology in a privacy-fulfilled manner. We extensively evaluate the framework on four real-world respiratory illnesses datasets, including Coswara, COUGHVID, ICBHI, and our private dataset - ASound, under multiple covariate shift settings (i.e., cross-dataset), improving up to 4.12% in overall performance.

## About this implementation

This implementation is based on the [fairseq toolkit](https://github.com/facebookresearch/fairseq).
Our main source code was written in the following directory:
+ [Profile Generation and Model Evaluation](examples/RoPADet)
+ [Spectrum Data Loading](fairseq/data)
+ [Model Personalization and Implementation](fairseq/models/RoPADet)

## Requirements and Installation
Please follow the instructions to [install the framework](https://github.com/facebookresearch/fairseq#getting-started).

Additionally, install [librosa]():
```
pip install librosa soundfile
```

## Training

### Preparing data

Create corresponding metadata (including: sample spectrum file path, number of frequency bands, number of time steps, label, profile id (optional)) for your dataset.

Place the meta information inside *data* directory.
<!-- 
Download and place data in *raw_data* directory, and create the corresponding metadatas in *data* directory.
Run:
```
python gen_spectrum.py --metadata_path=$meta_dir_path --profile=$generate_user_id_or_not --num_fft=$window_length --hop_length=$hop_length
``` -->
### Pre-training

Downstream task pre-training:
```
fairseq-hydra-train task.data=$pre-train_data --config-dir examples/RoPADet/config/pretraining --config-name general_pretrain
```

Profile encoder pre-training:
```
fairseq-hydra-train task.data=$pre-train_data --config-dir examples/RoPADet/config/pretraining --config-name discriminative
```

### Profile encoding:

```
python examples/RoPADet/profiles_gen.py data/ --path $model_path
```

### Self-training:

In each iteration, run:
```
python examples/RoPADet/profiles_gen.py data/ --path $teacher_path; CUDA_VISIBLE_DEVICES="3" fairseq-hydra-train task.data=$self-train_data task.profiles_path=$teacher_profile checkpoint.finetune_from_model=$pre-trained_model --config-dir examples/RoPADet/config/finetuning --config-name profile_self_training
```

### Fine-tuning

Without personalization:
```
fairseq-hydra-train task.data=$meta_dir_path model.w2v_path=pretrained_model --config-dir examples/RoPADet/config/finetuning --config-name without_profile
```

With personalization:
```
fairseq-hydra-train task.data=$meta_dir_path model.w2v_path=pretrained_model task.profiling=True task.profiles_path=$profile_path --config-dir examples/RoPADet/config/finetuning --config-name with_profile
```

## Evaluation

Model without personalization:
```
python examples/RoPADet/eval_classifier.py data --labels label --input_file $meta_dir_path
```

Model with personalization:
```
python examples/RoPADet/eval_classifier_profile.py data --labels label --input_file $meta_dir_path --profile_path $profile_path
```
