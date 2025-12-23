from datasets import load_dataset, Audio
from transducer.dataset.base import BaseDataset
from transducer.dataset.config import DatasetConfig, HFDatasetStruct
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class HFDataset(BaseDataset):
    def __init__(self, dataset:HFDatasetStruct, config:DatasetConfig):
        super().__init__(config)
        self.hf_dataset = load_dataset(
            dataset.hf_dataset_name,
            dataset.hf_dataset_suffix,
            split=dataset.hf_dataset_split,
            cache_dir = dataset.hf_cache_dir,
        )
        assert dataset.audio_column_name in self.hf_dataset.column_names
        assert dataset.text_column_name in self.hf_dataset.column_names

        self.dataset = self.dataset.cast_column(self.config.audio_column_name, Audio(decode = False))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        audio = row[self.config.audio_column_name]['bytes']
        text = row[self.config.text_column_name]
        audio = self.load_audio(audio)
        text = self.processor.tokenize(text)
        return {
            'audio':audio,
            'text':text,
        }

    def __collate_fn(self, batch):
        audios = [i['audio'] for i in batch]
        texts = [i['text'] for i in batch]

        features = self.processor.extract_features(audios, sampling_rate=self.sample_rate, return_tensors='pt')
        # Lazy, for features just take all frames as important when computing loss, might help with silence actually
        labels = pad_sequence(texts, batch_first = True, padding_value=self.processor.pad_id)
        label_lens = (labels!=self.processor.pad_id).sum(dim = -1)
        return {
            'audio_features':features,
            'labels':labels,
            'label_lens':label_lens,
            'audio_lens':None
        }

    def get_loader(self, batch_size):
        num_workers = self.config.num_workers
        pin_memory = self.config.pin_memory

        loader = DataLoader(
            self,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = pin_memory,
            collate_fn = self.__collate_fn,
        )
        return loader

class StreamingHFDataset(BaseDataset):
    def __init__(self, dataset:HFDatasetStruct, config:DatasetConfig):
        super().__init__(config)
        self.hf_dataset = load_dataset(
            dataset.hf_dataset_name,
            dataset.hf_dataset_suffix,
            split=dataset.hf_dataset_split,
            cache_dir=dataset.hf_cache_dir, streaming = True
        )
        assert dataset.audio_column_name in self.hf_dataset.column_names
        assert dataset.text_column_name in self.hf_dataset.column_names

        self.dataset = self.dataset.cast_column(self.config.audio_column_name, Audio(decode=False))

    def __iter__(self):
        for row in self.dataset:
            audio = row[self.config.audio_column_name]['bytes']
            text = row[self.config.text_column_name]
            audio = self.load_audio(audio)
            text = self.processor.tokenize(text)
            yield {
                'audio': audio,
                'text': text,
            }
