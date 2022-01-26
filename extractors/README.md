## Extractor code
Run all of the following steps from `<project_dir>/extractors`

### Installation
```
pip install -r ../requirements.txt
``` 

### Preprocessing data
To perform the preprocessing of QMSum necessary to reproduce the experiments, follow the instructions in the
[preprocessing](../preprocessing/README.md) directory.

### Download RelReg training code
```
git clone https://github.com/huggingface/transformers.git
cd transformers 
git checkout 65659a29cf5a079842e61a63d57fa24474288998
cd ..
```

### Run RelReg pipeline (data prep, training, inference, postprocessing) on utterance-level input

```
# switch to 1 for segment-level; outputs files for seq2seq training to output-relreg-utt
bash run_relreg.sh 0 output-relreg-utt
```

### Run RelRegTT pipeline

```
# switch to 1 for segment-level; outputs files to output-relregTT-utt
bash run_relreg_tt.sh 0 output-relregTT-utt
```
