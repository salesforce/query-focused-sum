"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import csv
import json

from tqdm import tqdm
from datasets import load_metric

sys.setrecursionlimit(10000)
metric = load_metric("rouge")

if __name__ == "__main__":
    do_chunks = int(sys.argv[1])

    for split in ["train", "val", "test"]:
        id2meetingsrc = {}
        if do_chunks:
            fname = os.path.join(os.path.dirname( __file__ ), \
                 "data", f'{split}.rouge.256.jsonl')
        else:
            fname = os.path.join(os.path.dirname( __file__ ), "..", \
                "data", f'{split}.rouge.jsonl')
        with open(fname) as f:
            for line in f:
                meeting_data = json.loads(line)
                if do_chunks:
                    id2meetingsrc[meeting_data['meeting_id']] = meeting_data['chunks']
                else:
                    id2meetingsrc[meeting_data['meeting_id']] = meeting_data['meeting_transcripts']

        if do_chunks:
            fname = os.path.join(os.path.dirname( __file__ ), "data", f'{split}.rouge.256.jsonl')
            fname_out_csv = os.path.join(os.path.dirname( __file__ ), \
                "..", "data", f"{split}.relreg.256.csv")
        else:
            fname = os.path.join(os.path.dirname( __file__ ), "..", "data", f"{split}.jsonl")
            fname_out_csv = os.path.join(os.path.dirname( __file__ ), \
                "..", "data", f"{split}.relreg.csv")

        totals = {"train": 1257, "val": 272, "test": 281}
        with open(fname) as f, open(fname_out_csv, "w") as outr:
            writer = csv.DictWriter(outr, fieldnames=["sentence1", "sentence2", "label"])
            writer.writeheader()
            for line in tqdm(f, total=totals[split]):
                data = json.loads(line)

                meeting_utterances = id2meetingsrc[data['meeting_id']]
                target = data['answer']
                query = data['query']
                scores = data["utt_rouge_f1"]

                for score, utt in zip(scores, meeting_utterances):
                    sent1 = query
                    sent2 = utt
                    label = score
                    cur_dict = {"sentence1": sent1, "sentence2": sent2, "label": label}
                    writer.writerow(cur_dict)
                