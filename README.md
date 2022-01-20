# Exploring Neural Models for Query-Focused Summarization

This is the official code repository for [Exploring Neural Models for Query-Focused Summarization](https://arxiv.org/abs/2112.07637)
by [Jesse Vig](https://twitter.com/jesse_vig)<sup>\*</sup>, [Alexander R. Fabbri](https://twitter.com/alexfabbri4)<sup>\*</sup>,
[Wojciech Kryściński](https://twitter.com/iam_wkr)<sup>\*</sup>, [Chien-Sheng Wu](https://twitter.com/jasonwu0731), and
[Wenhao Liu](https://twitter.com/owenhaoliu) (*equal contribution). 

We present code and instructions for reproducing the paper experiments and running the models against your own datasets.

## Table of contents
- [Introduction](#introduction)
- [Preprocessing data](#preprocessing-data)
- [Two-stage models](#two-stage-models)
- [Segment Encoder](#segment-encoder)
- [Citation](#citation)
- [License](#license)

## Introduction
Query-focused summarization (QFS) aims to produce summaries that answer particular questions of interest, enabling greater user control and personalization.
In [our paper](https://arxiv.org/abs/2112.07637) we conduct a systematic exploration of neural approaches to QFS, considering two general classes of methods: two-stage extractive-abstractive solutions and end-to-end models.
Within those categories, we investigate existing methods and present two model extensions that achieve state-of-the-art performance on the QMSum dataset  by a margin of up to 3.38 ROUGE-1, 3.72 ROUGE-2, and 3.28 ROUGE-L.

## Preprocessing data
To perform the preprocessing of QMSum necessary to reproduce the experiments, see the 
[preprocessing](preprocessing/README.md) directory.

## Two-stage models

Two-step approaches consist of an *extractor* model, which extracts parts of the source document relevant to the input query, and an *abstractor* model,
which synthesizes the extracted segments into a final summary.

See [extractors](extractors/README.md) directory for instructions and code for training and evaluating two-stage models.

## Segment Encoder

The Segment Encoder is an end-to-end model that uses sparse local attention to achieve SOTA ROUGE scores on the QMSum dataset.

To [replicate](multiencoder/README.md#reproducing-qmsum-experiments) the QMSum experiments, or train and evaluate Segment Encoder
[on your own dataset](multiencoder/README.md#running-on-your-own-datasets), see the 
 [multiencoder](multiencoder/README.md) directory.

## Citation

When referencing this repository, please cite [this paper](https://arxiv.org/abs/2112.07637):

```bibtex
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

## License

This repository is released under the [BSD-3 License](LICENSE.txt).





