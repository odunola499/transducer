
from typing import List, Literal, Optional, Union
from transducer.commons import Args

class DatasetConfig(Args):
    dataset_type: Literal['hf', 'jsonl']
    sample_rate:int = 16000

    # HF dataset
    dataset_name: Optional[str] = None
    dataset_suffix: Optional[str] = None

    # jsonl dataset
    jsonl_filepath:Optional[str] = None

    train_split: Optional[float] = None
    val_split: Optional[float] = None
    test_split: Optional[float] = None

    min_audio_length_ms:Optional[int] = 100
    max_audio_length_ms:Optional[int] = 30000
    min_text_length:Optional[int] = 100
    max_text_length:Optional[int] = 3000

    shuffle:bool = True
    num_workers:int = 8
    pin_memory:bool = True

    feature_extractor_type:Literal['wav2vec', 'wav2vecbert'] = "wav2vec"

    spe_tokenizer_path:Optional[str] = None
    train_new_tokenizer:bool = False
    vocab_size:int = 64

