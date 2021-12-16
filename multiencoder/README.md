## Multi-encoder processing
Run all of the following steps from `<project_dir>/multiencoder`

### Installation
```
pip install -r requirements.txt
``` 

### Preprocess Data
To convert files to a format that can be used by the multiencoder, run the following:
```
python preprocess.py
```

The output files will be in `data/qmsum/preprocessed`.

### Train models

See `scripts/train_qmsum_*.sh`

### Choose checkpoint for each run

`bash scripts/select_checkpoints.sh`. 

Copies best checkpoint for each run (based on mean validation rouge) to `selected_checkpoint` directory.

### Generate predictions from selected checkpoints

`bash scripts/predict_val.sh`
 
Writes out val predictions for all selected checkpoints to `selected_checkpoint/predictions.val`.

`bash scripts/predict_test.sh`
 
Writes out test predictions for all selected checkpoints to `selected_checkpoint/predictions.test`.


### Report rouge scores of all checkpoints

TODO: Link to other README for installing the rouge perl script dependencies.

`bash scripts/report_rouge_val.sh`

Reports mean rouge scores on validation set.

`bash scripts/report_rouge_test.sh`

Reports mean rouge scores on test set.

