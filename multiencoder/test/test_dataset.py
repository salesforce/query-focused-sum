import json
import unittest

import torch
from transformers import AutoTokenizer

from dataset import ChunkTokenizer, MultiEncoderDataset


class TestDataset(unittest.TestCase):

    def test_chunker(self):
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

        chunk_size = 10
        max_num_chunks = 2
        pad = False
        chunk_tokenizer = ChunkTokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_num_chunks=max_num_chunks,
            pad=pad
        )
        output = chunk_tokenizer(
            source="a b c d e",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b c d e</s>']
        )

        output = chunk_tokenizer(
            source="a b c d e f g",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b c d e f',
             '<s>y z</s> g</s><pad><pad><pad><pad>']
        )

        output = chunk_tokenizer(
            source="a b",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b</s><pad><pad><pad>', ]
        )

        pad = True
        chunk_tokenizer = ChunkTokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_num_chunks=max_num_chunks,
            pad=pad
        )
        output = chunk_tokenizer(
            source="a b c d e",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b c d e</s>',
             '<s>y z</s><pad><pad><pad><pad><pad><pad>']
        )

        # Test with stride
        chunk_size = 10
        max_num_chunks = 2
        pad = False
        stride = True
        chunk_tokenizer = ChunkTokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_num_chunks=max_num_chunks,
            pad=pad,
            stride=stride
        )
        output = chunk_tokenizer(
            source="a b c d e",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b c d e</s>',
             '<s>y z</s> d e</s><pad><pad><pad>']
        )

        max_num_chunks = 1
        chunk_tokenizer = ChunkTokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_num_chunks=max_num_chunks,
            pad=pad,
            stride=stride
        )
        output = chunk_tokenizer(
            source="a b c d e",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b c d e</s>']
        )

        max_num_chunks = 2
        pad = True
        chunk_tokenizer = ChunkTokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_num_chunks=max_num_chunks,
            pad=pad,
            stride=stride
        )
        output = chunk_tokenizer(
            source="a b c d e",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b c d e</s>',
             '<s>y z</s><pad><pad><pad><pad><pad><pad>',
             '<s>y z</s> d e</s><pad><pad><pad>'
            ]

        )

        output = chunk_tokenizer(
            source="a b",
            query="y z"
        )
        input_ids = output['input_ids']
        tokens = tokenizer.batch_decode(input_ids)
        self.assertEqual(
            tokens,
            ['<s>y z</s>a b</s><pad><pad><pad>',
             '<s>y z</s><pad><pad><pad><pad><pad><pad>',
             '<s>y z</s><pad><pad><pad><pad><pad><pad>']
        )

    def test_multiencoder_dataset(self):
        data_path = "data/dataset.jsonl"
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        chunk_size = 10
        max_target_length = 14
        max_num_chunks = 2
        stride = False
        d = MultiEncoderDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_target_length=max_target_length,
            max_num_chunks=max_num_chunks,
            stride=stride,
        )

        # Basic test
        self.assertEqual(len(d), 2)

        # Test chunk_tokenizer init
        self.assertEqual(d.chunk_tokenizer.tokenizer, tokenizer)
        self.assertEqual(d.chunk_tokenizer.chunk_size, chunk_size)
        self.assertEqual(d.chunk_tokenizer.max_num_chunks, max_num_chunks)
        self.assertEqual(d.chunk_tokenizer.stride, stride)

        # Test source
        with open(data_path) as f:
            row = next(f)
            data = json.loads(row)
        actual_input_ids = d.chunk_tokenizer(data['source'], data['query'])['input_ids']
        item = d[0]
        self.assertTrue(torch.equal(item['input_ids'], actual_input_ids))

        # Test labels
        item = d[0]
        self.assertEqual(
            item['labels'].tolist(),
            tokenizer(
                "<s>1 2 3 4 5 6 7 8 9 10 11 12</s>",
                add_special_tokens=False
            )['input_ids']
        )
        item = d[1]
        self.assertEqual(
            item['labels'].tolist(),
            tokenizer(
                "<s>1 2 3 4 5 6 7 8 9 10</s>",
                add_special_tokens=False
            )['input_ids'] + [-100, -100]
        )


if __name__ == '__main__':
    unittest.main()
