import unittest

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import MultiEncoderDataset
from models import BartForMultiConditionalGeneration


class TestModel(unittest.TestCase):

    def test_bart(self):
        data_path = "data/dataset.jsonl"
        model_name = 'facebook/bart-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chunk_size = 10
        max_num_chunks = 4
        max_target_length = 20
        pad = True

        d = MultiEncoderDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_num_chunks=max_num_chunks,
            max_target_length=max_target_length,
            pad=pad
        )

        batch_size = 2
        model = BartForMultiConditionalGeneration.from_pretrained(model_name)
        dataloader = DataLoader(d, batch_size=batch_size)
        batch = next(iter(dataloader))
        self.assertTrue(batch['input_ids'].shape == (batch_size, max_num_chunks, chunk_size))
        model.eval()
        output = model(**batch)
        self.assertTrue(output.encoder_last_hidden_state.shape == (batch_size, chunk_size * max_num_chunks, 768))
        batch.pop('labels')
        output = model.generate(**batch, return_dict=True)
        self.assertTrue(output.shape[0] == batch_size)
        # for tokens in output:
        #     print(tokenizer.decode(tokens))


if __name__ == '__main__':
    unittest.main()
