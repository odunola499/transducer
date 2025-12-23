
from typing import List, Literal, Optional, Union
from transducer.commons import Args
from dataclasses import dataclass


class TokenizerConfig(Args):
    vocab_size:int = 1024
    spe_tokenizer_path: Optional[str] = '../processor/lowercase_tokenizer.model'
    spe_model_prefix: str = 'tokenizer'
    train_new_tokenizer: bool = False
    tokenizer_dataset_path: Optional[str] = None
    unk_id: int = 0
    spe_model_type: Literal['bpe', 'unigram', 'char', 'word'] = 'bpe'

@dataclass
class DatasetStruct:
    audio_column_name: str = 'audio'
    text_column_name: str = 'text'

class HFDatasetStruct(DatasetStruct):
    hf_dataset_name: Optional[str] = None
    hf_dataset_suffix: Optional[str] = None
    hf_dataset_split: str = 'train'
    hf_cache_dir: str = 'data'

class JsonlDatasetStruct(DatasetStruct):
    jsonl_filepath: Optional[str] = None


class DatasetConfig(Args):
    dataset_type: Literal['hf', 'jsonl']
    train_data: Union[DatasetStruct, JsonlDatasetStruct]
    val_data: Union[DatasetStruct, JsonlDatasetStruct]
    tokenizer_config: TokenizerConfig = TokenizerConfig()
    feature_extractor_type:Literal['wav2vec2', 'wav2vecbert'] = 'wav2vec'
    sample_rate:int = 16000

    min_audio_length_ms:Optional[int] = 100
    max_audio_length_ms:Optional[int] = 30000
    min_text_length:Optional[int] = 100
    max_text_length:Optional[int] = 3000

    shuffle:bool = True
    num_workers:int = 8
    pin_memory:bool = True






