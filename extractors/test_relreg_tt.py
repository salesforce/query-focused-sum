
"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import json
import csv
import torch
from sentence_transformers import SentenceTransformer, util
from train_relreg_tt import get_examples

sys.setrecursionlimit(10000)


def write_to_file(examples, meetings_dict, output_dir, split):
    id2meetingsrc_embed = {}
    for curid, utts in meetings_dict.items():
        utts = [f"[DOC] {utt}" for utt in utts]
        meeting_embeddings = model.encode(utts, convert_to_tensor=True)
        id2meetingsrc_embed[curid] = meeting_embeddings

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(f"{output_dir}/{split}.csv", "w") as out, \
        open(f"{output_dir}/{split}.source", "w") as outs, \
        open(f"{output_dir}/{split}.target", "w") as outt, \
            open(f"{output_dir}/{split}.locator.jsonl", "w") as outl:
        writer = csv.DictWriter(out, fieldnames=["text", "summary"])
        writer.writeheader()
        for example in examples:
            query = example["query"]
            query_embedding = model.encode(f"[QRY] {query}", convert_to_tensor=True)

            cur_meeting_embeddings = id2meetingsrc_embed[example['meeting_id']]
            cur_meeting_utterances = meetings_dict[example['meeting_id']]
            cur_meeting_utterances = [x.replace("[DOC] ", "") for x in cur_meeting_utterances]

            cos_scores = util.pytorch_cos_sim(query_embedding, cur_meeting_embeddings)[0]
            top_results = torch.topk(cos_scores, k=len(cur_meeting_embeddings))

            scores = top_results[0].cpu().tolist()
            indices = top_results[1].cpu().tolist()

            assert len(cur_meeting_utterances) == len(indices)

            example["indices"] = indices
            example["scores"] = scores
            json.dump(example, outl)
            outl.write("\n")

            utts_ordered = [cur_meeting_utterances[x] for x in indices]
            meeting_source = " ".join(utts_ordered)

            target = example["answer"]
            source = f"{query}</s> {meeting_source}"
            cur_data = {"text": source, "summary": target}
            writer.writerow(cur_data)

            outs.write(source + "\n")
            outt.write(target + "\n")


if __name__ == "__main__":
    model_name = sys.argv[1]
    do_chunks = int(sys.argv[2])
    output_dir = sys.argv[3]
    max_seq_length = int(sys.argv[4])

    model = SentenceTransformer(model_name, device='cuda')
    model.max_seq_length = max_seq_length
    model.eval()

    _, train_meetings_dict, train_examples = get_examples("train", do_chunks)
    _, val_meetings_dict, val_examples = get_examples("val", do_chunks)
    _, test_meetings_dict, test_examples = get_examples("test", do_chunks)

    write_to_file(test_examples, test_meetings_dict, output_dir, "test")
    write_to_file(train_examples, train_meetings_dict, output_dir, "train")
    write_to_file(val_examples, val_meetings_dict, output_dir, "val")
