# Preprocessing code

Run all of the following steps from `<project_dir>/preprocessing`

## Download and format QMSum data for two-stage and Segment Encoder models
```
git clone https://github.com/Yale-LILY/QMSum.git
mv QMSum ../
python prep_qmsum.py ../QMSum
mv ../QMSum/data/ALL/jsonl/final ../data
``` 
