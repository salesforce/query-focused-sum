# Segment Encoder

NOTE: Run all of the following steps from `<project_dir>/multiencoder`.

## Table of contents
- [Installation](#installation)
- [Reproducing QMSum experiments](#reproducing-qmsum-experiments)
  * [1. Preprocess data](#1-preprocess-data)
  * [2. Train models](#2-train-models)
  * [3. Choose checkpoint for each run](#3-choose-checkpoint-for-each-run)
  * [4. Generate predictions from selected checkpoints](#4-generate-predictions-from-selected-checkpoints)
  * [5. Report rouge scores of all checkpoints](#5-report-rouge-scores-of-all-checkpoints)
- [Running on your own datasets](#running-on-your-own-datasets)
  * [1. Prepare data in appropriate format](#1-prepare-data-in-appropriate-format)
  * [2. Train your model](#2-train-your-model)
  * [3. Evaluate your model](#3-evaluate-your-model)
    + [HuggingFace rouge metric (simpler)](#huggingface-rouge-metric--simpler-)
    + [SummEval rouge metric](#summeval-rouge-metric)

## Installation
```
pip install -r requirements.txt
``` 

## Reproducing QMSum experiments

### 1. Preprocess data
To convert files to a format that can be used by the Segment Encoder, run the following:
```
python preprocess.py
```

The output files will be in `data/qmsum/preprocessed`.

### 2. Train models

See `scripts/train_qmsum_*.sh`

### 3. Choose checkpoint for each run

`bash scripts/select_checkpoints.sh`. 

Copies best checkpoint for each run (based on mean validation rouge) to `selected_checkpoint` directory.

### 4. Generate predictions from selected checkpoints

`bash scripts/predict_val.sh`
 
Writes out val predictions for all selected checkpoints to `selected_checkpoint/predictions.val`.

`bash scripts/predict_test.sh`
 
Writes out test predictions for all selected checkpoints to `selected_checkpoint/predictions.test`.

### 5. Report rouge scores of all checkpoints

`bash scripts/report_rouge_val.sh`

Reports mean rouge scores on validation set.

`bash scripts/report_rouge_test.sh`

Reports mean rouge scores on test set.

Note that these last scripts may prompt you with a small number of additional install steps.

## Running on your own datasets

### 1. Prepare data in appropriate format

The Segment Encoder data loaders expect a `.jsonl` file, with each line in the following format:

```
{"source": <full source document>, "query": <optional query>, "target": <summary>}
```

### 2. Train your model

You will need to execute [train.py](train.py) with the appropriate command-line arguments. Below is a template
for executing train.py based on the hyperparameters for the best-performing model ([scripts/train_qmsum_16_512_strided.sh](scripts/train_qmsum_16_512_strided.sh)). 
You will need to set `train_file` and `validation_file` to point to `.jsonl` files in the format described in Step 1, and `output_dir`
to point to the directory where the model checkpoints will be saved.
 
```bash
python train.py \
  --do_train \
  --train_file PATH_TO_TRAIN_FILE \
  --do_eval \
  --validation_file PATH_TO_VALIDATION_FILE \
  --model_name_or_path facebook/bart-large \
  --multiencoder_type bart \
  --multiencoder_max_num_chunks 32 \
  --multiencoder_stride \
  --max_source_len 512 \
  --learning_rate 0.000005 \
  --save_strategy epoch \
  --num_train_epochs 10 \
  --gradient_checkpointing \
  --output_dir PATH_TO_SAVE_MODEL \
  --per_device_train_batch_size 1 \
  --generation_max_len 256 \
  --val_max_target_length 256 \
  --evaluation_strategy epoch \
  --per_device_eval_batch_size 1 \
  --metric_for_best_model eval_mean_rouge \
  --compute_rouge_for_train \
  --predict_with_generate \
  --logging_strategy epoch \
  --load_best_model_at_end \
  --seed 1
```
 
Argument descriptions:
* `do_train`: Required boolean flag
* `train_file`: Path to your training file (in `.jsonl` format described above).
* `do_eval`: Boolean flag to evaluate model on validation set during training
* `validation_file`: Path to your optional validation file (in `.jsonl` format described above)
* `model_name_or_path`: Name of or path to Huggingface model (recommend `facebook/bart-large`). Currently only supports BART checkpoints.
* `multiencoder_type`: Set to `bart`
* `multiencoder_max_num_chunks`: Number of segments
* `multiencoder_stride`: Boolean flag to use 50%-overlap strides in segmentation. If not set, segments will be disjoint, which may degrade model performance.
* `max_source_len`: Segment length
* `learning_rate`: Learning rate (recommend 0.000005 if replicating paper experiments)
* `save_strategy`: Set to `epoch` to save checkpoint at end of each epoch
* `num_train_epochs`: Number of epochs
* `gradient_checkpointing` (recommended for larger models): Boolean flag to turn on gradient checkpointing, which reduces memory footprint and increases compute.
This may be necessary for some models depending on number of segments, size of segments, and GPU memory available.
* `output_dir`: Output directory for saved model checkpoints and logs
* `per_device_train_batch_size`: Batch size, typically 1 for larger models
* `generation_max_len` and `val_max_target_length`: Set to the maximum target length
* `evaluation_strategy`:  Set to `epoch` if you wish to evaluate at the end of each epoch
* `per_device_eval_batch_size`: Evaluation batch size, typically 1 for larger models
* `metric_for_best_model` (see also `compute_rouge_for_train` and `predict_with_generate` below): Set to `eval_mean_rouge` (recommended) if you wish use mean rouge as criterion for selecting checkpoint. Leave off to use cross entropy.
* `compute_rouge_for_train`: Include if you wish compute rouge as part of the eval in training (necessary if `metric_for_best_model` = `eval_mean_rouge` )
* `predict_with_generate`: Required boolean flag if `compute_rouge_for_train` set to True
* `logging_strategy`: Set to `epoch` to log results at end of each epoch
* `overwrite_output_dir`: Boolean flag to overwrite output directory with multiple runs
* `load_best_model_at_end`: Boolean flag to load the best checkpoint at the end
* `seed`: Optional random seed
* Optionally, other arguments for the Huggingface Seq2SeqTrainer specified in
 [Seq2SeqTrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments)

See [train.py](train.py) for
documentation on other arguments. Note that [train.py](train.py) is based on the standard HuggingFace [training script for summarization](https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py),
and uses many of the same command-line arguments.

### 3. Evaluate your model

There are two main options for evaluation, described below.

#### HuggingFace rouge metric (simpler)
This relies on [`datasets.load_metric()`](https://huggingface.co/docs/datasets/loading_metrics.html).

Run [train.py](train.py) with appropriate arguments for testing. Example template consistent with training template from Step 2:

```bash
python train.py \
  --do_predict \
  --test_file PATH_TO_TEST_FILE \
  --model_name_or_path PATH_TO_SAVE_MODEL \
  --multiencoder_type bart \
  --multiencoder_max_num_chunks 32 \
  --multiencoder_stride \
  --max_source_len 512 \
  --output_dir PATH_TO_TEST_OUTPUT \
  --generation_max_len 256 \
  --val_max_target_length 256 \
  --per_device_eval_batch_size 1 \
  --predict_with_generate \
  --prediction_path PATH_TO_PREDICTION_OUTPUT
```

You will need to set `test_file` to a test file in the `.jsonl` format described in Step 1. Set `model_name_or_path` to the
top-level `PATH_TO_SAVE_MODEL` specified in the training script; this top-level directory has the best-performing checkpoint 
according to the `metric_for_best_model` argument to the training script. Set `output_dir`
to the directory where testing outputs will go and `prediction_path` to the file where generated predictions will go. 
If you change any model parameters in the training
script be sure to update corresponding arguments in the test script (e.g. number of segments, segment length).

#### SummEval rouge metric


The [SummEval](https://github.com/Yale-LILY/SummEval) implementation uses the original PERL script for computing rouge.

To run this, you will need to first run the test script above, and then additionally run
 [`report_rouge.py`](../rouge/report_rouge.py) based on the generated predictions from the test script. You
can see examples of this in steps 4-5 in the [Reproducing Experiments section](#reproducing-experiments).

