# Two-Step Models

NOTE: Run all of the following steps from `<project_dir>/extractors`.

## Table of contents
- [Installation](#installation)
- [Extractor Component](#extractor-component)
  * [1. Preprocess QMSum](#1-preprocess-qmsum)
  * [2. Download RelReg training code](#2-download-relreg-training-code)
  * [3. Run RelReg pipeline](#3-run-relreg-pipeline)
  * [4. Run RelRegTT pipeline](#4-run-relregtt-pipeline)
- [Abstractor Component](#abstractor-component)
  * [1. Train models](#1-train-models)
  * [2. Choose Checkpoint](#2-choose-checkpoint)
  * [3. Generate Predictions](#3-generate-predictions)
  * [4. Report rouge scores](#4-report-rouge-scores)
  * [5. Pretrained Models](#5-pretrained-models)


## Installation
```
pip install -r ../requirements.txt
``` 

## Extractor Component

### 1. Preprocess QMSum

To perform the preprocessing of QMSum necessary to reproduce the experiments, follow the instructions in the
[preprocessing](../preprocessing/README.md) directory.

### 2. Download RelReg training code

```
git clone https://github.com/huggingface/transformers.git
cd transformers 
git checkout 65659a29cf5a079842e61a63d57fa24474288998
cd ..
```

### 3. Run RelReg pipeline

```
# Data prep, training, inference, postprocessing on utterance-level input
# switch to 1 for segment-level; outputs files for seq2seq training to output-relreg-utt
bash run_relreg.sh 0 output-relreg-utt
```

### 4. Run RelRegTT pipeline

```
# switch to 1 for segment-level; outputs files to output-relregTT-utt
bash run_relreg_tt.sh 0 output-relregTT-utt
```

## Abstractor Component

### 1. Train models

`bash train_qmsum_bart.sh`

### 2. Choose Checkpoint

Select best checkpoints from runs in the previous step, where NAME is taken from the `train_qmsum_bart.sh` script: </br>

`python ../multiencoder/select_checkpoints.py NAME`

### 3. Generate Predictions

To generate predictions on the validation set: 

`bash predict_qmsum_bart.sh`

### 4. Report rouge scores

`python ../rouge/report_rouge.py --ref-path PATH_TO_REFERENCES  --pred-paths PATH_TO_PREDICTIONS`

### 5. Pretrained Models

We have included checkpoints for all 5 training runs of the RelReg-W model used in the final evaluation, along with their performance on the **validation** set:

| Run       | ROUGE-1 | ROUGE-2 | ROUGE-L | Checkpoint |                                                                                                  
|-----------|---------|----| ---  |-------------------------------------------------------------------------------------------------------------------|
| 1 | 37.03 | 12.47   |   32.47   | [download](https://storage.googleapis.com/sfr-query-focused-sum-research/relreg-qmsum-256-wikisum-1.tar.gz) |
| 2 | 36.44 | 12.27   |   32.18   | [download](https://storage.googleapis.com/sfr-query-focused-sum-research/relreg-qmsum-256-wikisum-2.tar.gz) |
| 3 | 37.10 | 12.47   |   32.61   | [download](https://storage.googleapis.com/sfr-query-focused-sum-research/relreg-qmsum-256-wikisum-3.tar.gz) |
| 4 | 36.45 | 12.11   |   32.30   | [download](https://storage.googleapis.com/sfr-query-focused-sum-research/relreg-qmsum-256-wikisum-4.tar.gz) |
| 5 | 36.82 | 11.91   |   32.43   | [download](https://storage.googleapis.com/sfr-query-focused-sum-research/relreg-qmsum-256-wikisum-5.tar.gz) |

To generate predictions using these models, please download the above checkpoints and replace the `--model_name_or_path` line in `predict_qmsum_bart.sh` accordingly.