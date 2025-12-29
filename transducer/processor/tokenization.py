import os
from typing import List, Union

import sentencepiece as spm

from transducer.dataset.config import TokenizerConfig


class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.spm = None

        if self.config.spe_tokenizer_path is not None:
            self.load()
        else:
            assert self.config.train_new_tokenizer is True, (
                "Pretrained tokenizer path not given; set 'train_new_tokenizer' to train a new one."
            )
            assert os.path.exists(self.config.tokenizer_dataset_path)
            self.train()

    def train(self):
        tokenizer_dataset_path = self.config.tokenizer_dataset_path
        spm.SentencePieceTrainer.train(
            input=tokenizer_dataset_path,
            model_prefix=self.config.spe_model_prefix,
            vocab_size=self.config.vocab_size,
            unk_id=self.config.unk_id,
            model_type=self.config.spe_model_type,
            character_coverage=1.0,
            normalization_rule_name="nmt_nfkc_cf",
        )
        tokenizer_path = f"{self.config.spe_model_prefix}.model"
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(tokenizer_path)
        print(f"Trained tokenizer saved at {tokenizer_path}")
        print("blank token ", self.spm.id_to_piece(0))

        text = "Transducers for automatic speech recognition is superior"
        print(text, self.spm.encode(text, out_type=int))

    def encode(self, texts: Union[str, List[str]]):
        return self.spm.encode(texts, out_type=int)

    def decode(self, texts: Union[List[int], List[List[int]]]) -> str:
        return self.spm.decode(texts)

    @property
    def pad_id(self):
        return self.spm.unk_id()

    @property
    def unk_id(self):
        return self.spm.unk_id()

    def load(self):
        tokenizer_path = self.config.spe_tokenizer_path
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError("Tokenizer path given in config does not exist.")
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(tokenizer_path)


if __name__ == "__main__":
    config = TokenizerConfig()
    tokenizer = Tokenizer(config)

    text = "hello world"
    output = tokenizer.encode(text)
    print(tokenizer.pad_id)
    print(tokenizer.unk_id)
    print(output)
