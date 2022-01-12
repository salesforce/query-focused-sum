# Query-Focused Summarization

This is the official code repository for [Exploring Neural Models for Query-Focused Summarization](https://arxiv.org/abs/2112.07637)
by Jesse Vig<sup>\*</sup>, Alexander R. Fabbri<sup>\*</sup>, Wojciech Kryściński<sup>\*</sup>, Chien-Sheng Wu, and Wenhao Liu
(*equal contribution). We present code and instructions for reproducing the paper experiments or running the models against your own datasets.

## Preprocessing data

See [preprocessing](preprocessing/README.md) directory for instructions and code to perform the QMSum preprocessing 
 necessary to replicate the experiments.

## Running two-stage models

See [extractors](extractors/README.md) directory for instructions and code for training and evaluating two-stage models.

## Running Segment Encoder model

See [multiencoder](multiencoder/README.md) directory for instructions and code for training and evaluating the Segment
Encoder model, including instructions on how to run against your own dataset.

## Citation

When referencing this repository, please cite [this paper](https://arxiv.org/abs/2112.07637):

```
@misc{vig-etal-2021-exploring,
      title={Exploring Neural Models for Query-Focused Summarization}, 
      author={Jesse Vig and Alexander R. Fabbri and Wojciech Kry{\'s}ci{\'n}ski and Chien-Sheng Wu and Wenhao Liu},
      year={2021},
      eprint={2112.07637},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2112.07637}
}
```





