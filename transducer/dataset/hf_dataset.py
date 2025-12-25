from datasets import load_dataset, Audio
from transducer.dataset.base import BaseDataset, StreamingBaseDataset
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

        self.hf_dataset = self.hf_dataset.cast_column(dataset.audio_column_name, Audio(decode=False))
        self.dataset_config = dataset
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        audio = row[self.dataset_config.audio_column_name]['bytes']
        raw_text = row[self.dataset_config.text_column_name]
        audio = self.load_audio(audio)
        text = self.processor.tokenize(raw_text)
        return {
            'audio': audio,
            'text': text,
            'raw_text': raw_text,
            'index': idx,
        }

    def _collate_fn(self, batch):
        audios = [i['audio'] for i in batch]
        texts = [i['text'] for i in batch]
        indices = [i['index'] for i in batch]
        raw_texts = [i.get("raw_text") for i in batch]

        features = self.processor.extract_features(
            audios, sampling_rate=self.sample_rate, return_tensors='pt'
        )
        if isinstance(features, dict):
            features = features["input_values"]
        else:
            features = features.input_values
        # Lazy, for features just take all frames as important when computing loss, might help with silence actually
        labels = pad_sequence(texts, batch_first = True, padding_value=self.processor.pad_id)
        label_lens = (labels!=self.processor.pad_id).sum(dim = -1)
        return {
            'audio_features': features,
            'labels': labels,
            'label_lens': label_lens,
            'audio_lens': None,
            'indices': indices,
            'texts': raw_texts,
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
            collate_fn = self._collate_fn,
        )
        return loader

class StreamingHFDataset(StreamingBaseDataset):
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

        self.hf_dataset = self.hf_dataset.cast_column(dataset.audio_column_name, Audio(decode=False))
        self.dataset_config = dataset

    def __iter__(self):
        for index, row in enumerate(self.hf_dataset):
            if not self.should_yield(index):
                continue
            audio = row[self.dataset_config.audio_column_name]['bytes']
            raw_text = row[self.dataset_config.text_column_name]
            audio = self.load_audio(audio)
            text = self.processor.tokenize(raw_text)
            yield {
                'audio': audio,
                'text': text,
                'raw_text': raw_text,
                'index': index,
            }

    def _collate_fn(self, batch):
        audios = [i['audio'] for i in batch]
        texts = [i['text'] for i in batch]
        indices = [i['index'] for i in batch]
        raw_texts = [i.get("raw_text") for i in batch]

        features = self.processor.extract_features(
            audios, sampling_rate=self.sample_rate, return_tensors='pt'
        )
        if isinstance(features, dict):
            features = features["input_values"]
        else:
            features = features.input_values
        # Lazy, for features just take all frames as important when computing loss, might help with silence actually
        labels = pad_sequence(texts, batch_first = True, padding_value=self.processor.pad_id)
        label_lens = (labels!=self.processor.pad_id).sum(dim = -1)
        return {
            'audio_features': features,
            'labels': labels,
            'label_lens': label_lens,
            'audio_lens': None,
            'indices': indices,
            'texts': raw_texts,
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
            collate_fn = self._collate_fn,
        )
        return loader
