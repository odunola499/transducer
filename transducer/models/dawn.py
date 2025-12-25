import torch
from torch import Tensor
from typing import Optional

from transducer.models.base import BaseModel
from transducer.models.config import ModelConfig
from transducer.losses import LOSSES
from transducer.models.decoder.rnnt import SimpleJoiner, RNNPredictor
from transducer.models.encoder.wav2vec2bert import Wav2Vec2BertModel
from transducer.samplers import GenerationMixin

class DawnModel(BaseModel, GenerationMixin):
    def __init__(self, vocab_size:int, config: ModelConfig):
        super().__init__(config)
        self.model_name = config.model_name
        self.blank_id = config.blank_id

        LOSS = LOSSES[config.loss_type]
        if config.loss_type == 'tdt':
            assert config.loss_duration is not None, "Duration is required for tdt loss"
        self.loss_func = LOSS(
            blank_id = self.blank_id,
            reduction=config.loss_reduction,
            fastemit_lambda=config.fastemit_lambda,
        )

        encoder_config = config.encoder_config
        decoder_config = config.decoder_config

        self.encoder = Wav2Vec2BertModel(encoder_config)
        self.predictor = RNNPredictor(decoder_config)
        self.joiner = SimpleJoiner(
            vocab_size,
            encoder_dim=encoder_config.hidden_size,
            config=decoder_config
        )

    def get_feature_extractor(self):
        if hasattr(self, "feature_extractor"):
            return self.feature_extractor
        raise NotImplementedError("feature_extractor is not set on the model.")

    def get_tokenizer(self):
        if hasattr(self, "tokenizer"):
            return self.tokenizer
        raise NotImplementedError("tokenizer is not set on the model.")

    def forward(
            self,
            audio_features:Tensor,
            labels:Tensor,
            label_lens:Tensor,
            audio_lens:Optional[Tensor] = None,
    ):
        encoder_outputs = self.encoder(audio_features)
        predictor_outputs, _ = self.predictor(labels)
        logits = self.joiner(encoder_outputs, predictor_outputs)
        loss = self.compute_loss(
            logits,
            labels,
            act_lens=audio_lens,
            label_lens=label_lens,
        )
        return {
            "loss": loss,
            "logits": logits,
        }

    def compute_loss(self, lattice:Tensor, labels:Tensor, act_lens:Tensor, label_lens:Tensor):
        return self.loss_func(
            acts = lattice,
            labels = labels,
            label_lens = label_lens,
            act_lens = act_lens,
        )

