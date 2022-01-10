# Segment Encoder

## Installation
```
pip install -r requirements.txt
``` 

## Reproducing experiments

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

You will need to execute [train.py](train.py) with the appropriate command-line arguments.
 See [scripts/train_qmsum_16_512_strided.sh](scripts/train_qmsum_16_512_strided.sh) for an example.
Some arguments that you may need to modify (not comprehensive):
* `train_file`: Path to your training file (in `.jsonl` format described (above).
* `validation_file`: Path to your evaluation file (optional)
* `do_eval`: Whether to perform evaluation during training
* `learning_rate`: Learning rate
* `gradient_checkpointing`: Whether to use gradient checkpointing, which reduces memory footprint and increases compute.
This may be necessary for some models depending on number of segments, size of segments, and GPU memory available.
* `metric_for_best_model`: Only include if you wish to use a different metric than the training objective (cross-entropy)
* `max_source_len`: Segment length
* `generation_max_len` and `val_max_target_length`: Set these to the maximum target length
* `multiencoder_max_num_chunks`: Number of segments
* `multiencoder_stride`: Whether to use 50%-overlap strides in segmentation. If not set, segments will be disjoint.
* `compute_rouge_for_train`: Whether to compute rouge as part of the eval in training

See [train.py](train.py) for
documentation on other arguments. Note that [train.py](train.py) is based on the standard HuggingFace [training script for summarization](https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py),
and uses many of the same command-line arguments.

### 3. Evaluate your model

You have 3 options here from simplest to easiest:
1. Perform evaluation during training (for validation file) in previous step. This uses the rouge score implementation from [Huggingface Datasets](https://huggingface.co/docs/datasets/loading_metrics.html). 
2. Run [train.py](train.py) in test mode (described below). This uses the rouge score implementation from [Huggingface Datasets](https://huggingface.co/docs/datasets/loading_metrics.html). 
3. Replicate steps 2-5 from the [previous section](#reproducing-experiments), which uses the [SummEval](https://github.com/Yale-LILY/SummEval) implementation of rouge.

#### Using Huggingface rouge implementation

You can simply run [train.py](train.py), with the same arguments as the training script, but with the following changes:
* `do_train` and `do_eval`: Remove these
* `do_predict`: Include this to run in predict/test mode
* `train_file` and `validation_file`: Remove these
* `test_file`: Set this to your test file path
* `evaluation_strategy`: Remove
* `load_best_model_at_end`: Remove
* `save_strategy`: Remove
* `logging_strategy`: Remove

Keep the rest of the arguments, though you may optionally remove any that are training-specific (e.g. learning_rate)






