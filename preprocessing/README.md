# Preprocessing code

Run all of the following steps from `<project_dir>/preprocessing`

## Download and format QMSum data for two-stage and Segment Encoder models
```
git clone https://github.com/Yale-LILY/QMSum.git
mv QMSum ../
python prep_qmsum.py ../QMSum
mv ../QMSum/data/ALL/jsonl/final ../data
``` 

This will produce three files for each of the train, val, and test splits in the `../data` folder:


`<split>.jsonl` 

contains one data point per line. Each data point consists of an individual query, query and meeting ids, a reference summary, and the general/specific query type label.

`<split>-meetings.jsonl` 

contains the meeting transcripts along with the associated `meeting_id` used to join the data with `<split>.jsonl`.

`<split>.target` 

contains one data point per line in the format used for ROUGE evaluation. The order of the data points aligns with `<split>.jsonl`.

