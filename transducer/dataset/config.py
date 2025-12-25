
from typing import Literal, Optional, Union
from pydantic import StrictBool, StrictInt, StrictStr
from transducer.config import Args


class TokenizerConfig(Args):
    vocab_size:StrictInt = 1024
    spe_tokenizer_path: Optional[StrictStr] = '../processor/lowercase_tokenizer.model'
    spe_model_prefix: StrictStr = 'tokenizer'
    train_new_tokenizer: StrictBool = False
    tokenizer_dataset_path: Optional[StrictStr] = None
    unk_id: StrictInt = 0
    spe_model_type: Literal['bpe', 'unigram', 'char', 'word'] = 'bpe'

class DatasetStruct(Args):
    audio_column_name: StrictStr = 'audio'
    text_column_name: StrictStr = 'text'

class HFDatasetStruct(DatasetStruct):
    hf_dataset_name: Optional[StrictStr] = None
    hf_dataset_suffix: Optional[StrictStr] = None
    hf_dataset_split: StrictStr = 'train'
    hf_cache_dir: StrictStr = 'data'

class JsonlDatasetStruct(DatasetStruct):
    jsonl_filepath: Optional[StrictStr] = None


class DatasetConfig(Args):
    dataset_type: Literal['hf', 'jsonl']
    train_data: Union[HFDatasetStruct, JsonlDatasetStruct]
    val_data: Union[HFDatasetStruct, JsonlDatasetStruct]
    tokenizer_config: TokenizerConfig = TokenizerConfig()
    feature_extractor_type:Literal['wav2vec2', 'wav2vec-bert'] = 'wav2vec2'
    sample_rate:StrictInt = 16000

    min_audio_length_ms:Optional[StrictInt] = 100
    max_audio_length_ms:Optional[StrictInt] = 30000
    min_text_length:Optional[StrictInt] = 100
    max_text_length:Optional[StrictInt] = 3000

    shuffle:StrictBool = True
    num_workers:StrictInt = 8
    pin_memory:StrictBool = True
