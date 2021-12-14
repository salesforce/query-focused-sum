"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Logs results from multiple training runs

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from operator import itemgetter
from statistics import mean
import argparse
import glob
import os
import re
import json
from collections import defaultdict
from statistics import mean, stdev
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train_dir_prefix',
        help='prefix of output directories for training runs being reported on'
    )
    parser.add_argument(
        '--sort_metric',
        type=str,
        default='mean',
    )
    parser.add_argument(
        '--reverse_sort',
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    metrics = ['eval_rouge1', 'eval_rouge2', 'eval_rougeLsum', 'eval_loss', 'eval_gen_len']

    scores = defaultdict(list)
    n_runs = 0
    results = []
    print(f"****** {args.train_dir_prefix} ******")
    for filepath in sorted(glob.glob(f"{args.train_dir_prefix}*")):
        m = re.match(rf'{re.escape(args.train_dir_prefix)}_?(\d+)$', filepath)
        if m:
            run_index = int(m.group(1))
            results.append((run_index, filepath))
    if args.sort_metric == 'mean':
        sort_func = lambda x: mean([x['eval_rouge1'], x['eval_rouge2'], x['eval_rougeLsum']])
    else:
        sort_func = itemgetter(f'eval_{args.sort_metric}')
    for run_index, filepath in sorted(results):
        try:
            with open(os.path.join(filepath, "trainer_state.json")) as f:
                data = json.load(f)
                epoch_logs = [log for log in data['log_history'] if 'eval_loss' in log]
                sorted_epochs = sorted(
                    epoch_logs,
                    key=sort_func,
                    reverse=args.reverse_sort)
                best_epoch = sorted_epochs[-1]
                best_checkpoint = f'{filepath}/checkpoint-{best_epoch["step"]}'
                print(best_checkpoint)
                for metric in metrics:
                    scores[metric].append(best_epoch[metric])
                n_runs += 1
        except FileNotFoundError:
            pass

    print(f"Num runs: {n_runs}")
    for metric in metrics:
        print(metric)
        print(f"\tMean: {mean(scores[metric]):.2f}")
        if n_runs > 1:
            print(f"\tStd Dev: {stdev(scores[metric]):.2f}")
        print(f"\tRange: {min(scores[metric]):.2f}-{max(scores[metric]):.2f}")
        print("\tScores: " + ", ".join(f"{score:.2f}" for score in scores[metric]))

