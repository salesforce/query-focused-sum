"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Report mean rouge scores given a file of reference summaries and one or more files of predicted summaries

import argparse
from collections import defaultdict
from statistics import mean

import stanza

import rouge_metric

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')

def preprocess(text):
    doc = nlp(text)
    return '\n'.join(
        ' '.join(token.text for token in sentence.tokens)
            for sentence in doc.sentences
    )


def report_mean_rouge(ref_path, pred_paths):
    metric = rouge_metric.RougeMetric()

    with open(ref_path) as f:
        refs = [preprocess(line) for line in f]
        print('First ref')
        print(refs[0])

    all_scores = defaultdict(list)
    for i, pred_path in enumerate(pred_paths):
        with open(pred_path) as f:
            preds = [preprocess(line) for line in f]
            if i == 0:
                print('First pred')
                print(preds[0])
            results = metric.evaluate_batch(preds, refs, aggregate=True)
            # print(results)
            all_scores['rouge1'].append(results['rouge']['rouge_1_f_score'] * 100)
            all_scores['rouge2'].append(results['rouge']['rouge_2_f_score'] * 100)
            all_scores['rougeL'].append(results['rouge']['rouge_l_f_score'] * 100)
    for metric_name, scores in sorted(all_scores.items()):
        print()
        print('*' * 10)
        print(metric_name)
        print('Individual scores:', ', '.join(f'{score:.2f}' for score in scores))
        print(f'Mean: {mean(scores):.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-path', help='path to file with reference summaries')
    parser.add_argument('--pred-paths', nargs='+', help='paths to prediction files')
    args = parser.parse_args()
    report_mean_rouge(args.ref_path, args.pred_paths)
