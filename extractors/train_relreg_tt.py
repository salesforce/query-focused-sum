"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
import json
import math

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

sys.setrecursionlimit(10000)


def get_examples(split, do_chunks):
    id2meetingsrc = {}
    if do_chunks:
        fname = os.path.join(os.path.dirname( __file__ ), "data", f"{split}.rouge.256.jsonl")
    else:
        fname = os.path.join(os.path.dirname( __file__ ), "..", "data", f'{split}.rouge.jsonl')
    with open(fname) as f:
        for line in f:
            meeting_data = json.loads(line)
            if do_chunks:
                id2meetingsrc[meeting_data['meeting_id']] = meeting_data['chunks']
            else:
                id2meetingsrc[meeting_data['meeting_id']] = meeting_data['meeting_transcripts']

    if do_chunks:
        fname = os.path.join(os.path.dirname( __file__ ), "data", f"{split}.rouge.256.jsonl")
    else:
        fname = os.path.join(os.path.dirname( __file__ ), "..", "data", f"{split}.rouge.jsonl")

    totals = {"train": 1257, "val": 272, "test": 281}
    cur_examples = []
    cur_examples_meta = []
    with open(fname) as f:
        for line in tqdm(f, total=totals[split]):
            data = json.loads(line)
            cur_examples_meta.append(data)

            meeting_utterances = id2meetingsrc[data['meeting_id']]
            scores = data['utt_rouge_f1']
            query = data['query']

            for utt, utt_score in zip(meeting_utterances, scores):
                if len(utt.strip()) == 0:
                    continue
                label = float(utt_score)
                cur_example = InputExample(texts=[f"[QRY] {query}", f"[DOC] {utt}"], label=label)
                cur_examples.append(cur_example)
    return cur_examples, id2meetingsrc, cur_examples_meta


if __name__ == "__main__":
    model_name = sys.argv[1] # 'nli-distilroberta-base-v2'
    do_chunks = int(sys.argv[2])
    output_dir = sys.argv[3]
    max_seq_length = int(sys.argv[4])

    model = SentenceTransformer(model_name, device='cuda')
    model.max_seq_length = max_seq_length

    word_embedding_model = model._first_module()

    tokens = ["[DOC]", "[QRY]"]
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    train_batch_size = 16
    num_epochs = 4

    train_examples, _, _ = get_examples("train", do_chunks)
    dev_examples, _, _ = get_examples("val", do_chunks)


    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='qmsum-val')
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1,
            warmup_steps=warmup_steps,
            output_path=output_dir)
            # 1000