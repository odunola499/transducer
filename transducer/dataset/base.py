import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union
from torchcodec.decoders import AudioDecoder
from typing import Optional
from torchaudio.functional import resample
from transducer.dataset.config import DatasetConfig


class BaseDataset(Dataset):
    def __init__(self, config:DatasetConfig) -> None:
        super().__init__()
        self.config = config

        self.sample_rate = config.sample_rate
        self.resamplers = {}

    def load_audio(self, audio:Union[np.ndarray, torch.Tensor, bytes, str], orig_sr:Optional[int] = None) -> torch.FloatTensor:

        if isinstance(audio, np.ndarray):
            assert orig_sr is not None, "orig_sr must be provided for type numpy array"
            tensor = torch.from_numpy(audio)
        elif isinstance(audio, torch.Tensor):
            assert orig_sr is not None, "orig_sr must be provided for type torch tensor"
            tensor = audio
        elif isinstance(audio, str) or isinstance(audio, bytes):
            decoder = AudioDecoder(audio, sample_rate=self.sample_rate, num_channels=1)
            tensor = decoder.get_all_samples().T
            orig_sr = self.sample_rate

        else:
            raise ValueError("Unknown audio type")

        if tensor.ndim > 1:
            if tensor.shape[0] >2:
                tensor = tensor.T
            tensor = tensor.mean(dim = 0)

        if orig_sr != self.sample_rate:
                tensor = resample(tensor, orig_freq=orig_sr, new_freq=self.sample_rate, )

        return tensor

    def train_new_tokenizer(self, texts):
        pass

    def validate_pretrained_tokenizer(self):
        """
        Check if
        1. Pretrained tokenizer exists
        2. Pretrained tokenizer vocab size is valid with vocab_size in config
        3.
        """