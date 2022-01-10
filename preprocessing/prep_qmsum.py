"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import os
import re
import json
from pathlib import Path

def format_utterance(utterance):
    content = re.sub(r'\{[^\}]*\}', '', \
            utterance['content']).strip()  # Remove all content of form "{...}"
    content = content.replace('A_M_I_', 'AMI')
    content = content.replace('L_C_D_', 'LCD')
    content = content.replace('P_M_S', 'PMS')
    content = content.replace('T_V_', 'TV')
    return f"{utterance['speaker']}: {content}" if content else None

def fix_spans(prev_relevant_text_spans, none_count_arr):
    relevant_text_spans = []
    for relevant_text_span in prev_relevant_text_spans:
        start_, end_ = int(relevant_text_span[0]), int(relevant_text_span[1])
        # move the span index to the left by the number
        # of blank utterances to the left of that index
        start = max(start_ - none_count_arr[start_], 0)
        end = max(end_ - none_count_arr[end_], 0)
        relevant_text_spans.append([str(start), str(end)])
        # The start/end utterances should be the same, except in the case
        #where the start/end was a blank utterance (which does occur in the annotations)
        assert meeting_transcripts_w_none[end_] == meeting_transcripts[end] \
                or meeting_transcripts_w_none[end_] is None
        assert meeting_transcripts_w_none[start_] == meeting_transcripts[start] \
                or meeting_transcripts_w_none[start_] is None
    return relevant_text_spans


if __name__ == "__main__":
    # path to qmsum github folder
    qmsum_path_str = sys.argv[1]
    qmsum_path = Path(qmsum_path_str)

    academic = qmsum_path / "data/Academic/jsonl"
    academic_topics = set()
    for split in ["train", "val", "test"]:
        with open(os.path.join(academic, f"{split}.jsonl")) as f:
            for line in f:
                data = json.loads(line)
                academic_topics.add(str(data['topic_list']))

    committee = qmsum_path / "data/Committee/jsonl"
    committee_topics = set()
    for split in ["train", "val", "test"]:
        with open(os.path.join(committee, f"{split}.jsonl")) as f:
            for line in f:
                data = json.loads(line)
                committee_topics.add(str(data['topic_list']))

    product = qmsum_path / "data/Product/jsonl"
    product_topics = set()
    for split in ["train", "val", "test"]:
        with open(os.path.join(product, f"{split}.jsonl")) as f:
            for line in f:
                data = json.loads(line)
                product_topics.add(str(data['topic_list']))


    for split in ["train", "val", "test"]:
        query_count = 0
        if not os.path.exists(f"{str(qmsum_path)}/data/ALL/jsonl/final"):
            os.mkdir(f"{str(qmsum_path)}/data/ALL/jsonl/final")

        with open(f"{qmsum_path_str}/data/ALL/jsonl/{split}.jsonl") as f, \
                open(f"{qmsum_path_str}/data/ALL/jsonl/final/{split}.jsonl", "w") as outqf, \
                open(f"{qmsum_path_str}/data/ALL/jsonl/final/{split}-meetings.jsonl", "w") as outmf, \
                open(f"{qmsum_path_str}/data/ALL/jsonl/final/{split}.target", "w") as outt:
            for line_count, line in enumerate(f):
                data = json.loads(line)
                topic_str = str(data['topic_list'])
                if topic_str in product_topics:
                    domain = "product"
                elif topic_str in committee_topics:
                    domain = "committee"
                else:
                    domain = "academic"

                # Write meeting to output file
                meeting_id = f"m_{split}_{line_count}"
                meeting_data = {}
                meeting_data["meeting_id"] = meeting_id
                meeting_data["domain"] = domain

                meeting_transcripts_w_none = [format_utterance(utt) for utt in data["meeting_transcripts"]]
                none_count_arr = [0] * len(meeting_transcripts_w_none)
                none_count = 0
                for i in range(len(meeting_transcripts_w_none)):
                    if meeting_transcripts_w_none[i] is None:
                        none_count += 1
                    none_count_arr[i] = none_count
                meeting_transcripts = [x for x in meeting_transcripts_w_none if x is not None]
                meeting_data["meeting_transcripts"] = meeting_transcripts

                topic_list = []
                for topic in data["topic_list"]:
                    relevant_text_spans = fix_spans(topic['relevant_text_span'], none_count_arr)
                    topic_list.append({"topic": topic["topic"], "relevant_text_span": relevant_text_spans})
                meeting_data["topic_list"] = topic_list

                json.dump(meeting_data, outmf)
                outmf.write("\n")


                for gen_query in data['general_query_list']:
                    query_data = {}
                    query_id = f"q_{split}_{query_count}"
                    query_data["query_id"] = query_id
                    query_data["query"] = gen_query["query"]
                    query_data["answer"] = gen_query["answer"]
                    query_data["query_type"] = "general"
                    query_data["meeting_id"] = meeting_id
                    json.dump(query_data, outqf)
                    outqf.write("\n")
                    outt.write(gen_query["answer"] + "\n")
                    query_count += 1

                for spec_query in data['specific_query_list']:
                    query_data = {}
                    query_id = f"q_{split}_{query_count}"
                    query_data["query_id"] = query_id
                    query_data["query"] = spec_query["query"]
                    query_data["answer"] = spec_query["answer"]
                    query_data["query_type"] = "specific"
                    query_data["meeting_id"] = meeting_id

                    relevant_text_spans = fix_spans(spec_query["relevant_text_span"], none_count_arr)
                    query_data["relevant_text_span"] = relevant_text_spans

                    json.dump(query_data, outqf)
                    outqf.write("\n")
                    outt.write(spec_query["answer"] + "\n")
                    query_count += 1
