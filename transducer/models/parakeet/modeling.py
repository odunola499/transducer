from typing import Optional

import torch
from torch import nn, Tensor

from safetensors.torch import safe_open
from huggingface_hub import hf_hub_download

from transducer.models.modules import GenerationMixin
from transducer.models.parakeet.config import (
    FastConformerConfig,
    ParakeetDecoderConfig,
    ParakeetModelConfig,
)
from transducer.models.parakeet.encoder import FastConformerEncoder
from transducer.models.parakeet.feature_extractor import NemoFeatureExtractor
from transducer.models.parakeet.joiner import NemoJoiner
from transducer.models.parakeet.predictor import NemoPredictor
from transducer.models.parakeet.tokenizer import ParakeetTokenizer


class Parakeet(nn.Module, GenerationMixin):
    def __init__(self, config: ParakeetModelConfig):
        super().__init__()
        self.config = config
        decoder_config = config.decoder_config
        vocab_size = decoder_config.vocab_size

        encoder_config = config.encoder_config
        self.encoder = FastConformerEncoder(encoder_config)
        self.decoder = NemoPredictor(
            config=decoder_config,
            vocab_size=vocab_size,
        )

        self.joint = NemoJoiner(
            encoder_dim=encoder_config.hidden_size,
            pred_dim=decoder_config.pred_dim,
            joint_dim=decoder_config.joint_dim,
            num_classes=vocab_size,
        )
        self.blank_id = (
            config.blank_id if config.blank_id is not None else decoder_config.vocab_size
        )
        self.sampler_type = config.sampler_type
        self._feature_extractor = None
        self._tokenizer = None

    def forward(
        self,
        audio_features: Tensor,
        labels: Tensor,
        audio_lens: Tensor,
        label_lens: Tensor | None = None,
    ):
        encoder_outputs = self.encoder(audio_features, audio_lens)
        predictor_outputs, label_lens, _ = self.decoder(labels, label_lens)
        logits = self.joint(encoder_outputs, predictor_outputs)
        return {"loss": None, "logits": logits}

    def get_feature_extractor(self):
        if self._feature_extractor is not None:
            return self._feature_extractor
        return NemoFeatureExtractor()

    def get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        tokenizer_path = getattr(self.config, "tokenizer_path", None)
        if tokenizer_path:
            return ParakeetTokenizer(tokenizer_path)
        raise NotImplementedError("tokenizer is not set on the model.")

    def get_encoder(self):
        return self.encoder

    def get_joiner(self):
        return self.joint

    def get_predictor(self):
        return self.decoder

    @classmethod
    def from_pretrained(
        cls,
        attn_impl: str = "sdpa",
    ):
        encoder_config = FastConformerConfig()
        encoder_config.attn_impl = attn_impl
        decoder_config = ParakeetDecoderConfig()

        repo_id = 'odunola/parakeet-EOU'
        model_path = hf_hub_download(
            repo_id = repo_id,
            filename = 'model.safetensors'
        )
        tokenizer_path = hf_hub_download(
            repo_id = repo_id,
            filename = 'tokenizer.model'
        )

        config = ParakeetModelConfig(
            model_name='parakeet',
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            blank_id=decoder_config.vocab_size,
            tokenizer_path=tokenizer_path,
        )

        state_dict = dict()
        with safe_open(model_path, framework = 'pt', device = 'cpu') as fp:
            for key in fp.keys():
                state_dict[key] = fp.get_tensor(key)

        model = cls(config)
        model.load_state_dict(state_dict)
        model._feature_extractor = NemoFeatureExtractor()
        model._tokenizer = ParakeetTokenizer(tokenizer_path)
        
        return model
