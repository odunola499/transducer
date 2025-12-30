from torch import nn, Tensor

from transducer.models.config import (
    ModelConfig,
)
from transducer.models.decoder import NemoJoiner, NemoPredictor
from transducer.models.encoder import FastConformerEncoder

from transducer.samplers import GenerationMixin
from transducer.processor import NemoFeatureExtractor, TokenizerConfig, Tokenizer


class Parakeet(nn.Module, GenerationMixin):
    def __init__(self, config: ModelConfig):
        super().__init__()
        vocab_size = config.vocab_size

        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
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

    def forward(
        self,
        audio_features: Tensor,
        labels: Tensor,
        audio_lens: Tensor,
        label_lens: Tensor | None = None,
    ):
        encoder_outputs = self.encoder(audio_features, audio_lens)
        predictor_outputs, _ = self.decoder(
            labels,
        )
        logits = self.joint(encoder_outputs, predictor_outputs)
        return {"loss": None, "logits": logits}

    def get_feature_extractor(self):
        return NemoFeatureExtractor()

    def get_tokenizer(self):
        tokenizer_config = TokenizerConfig()
        return Tokenizer(tokenizer_config)

    def get_encoder(self):
        return self.encoder

    def get_joiner(self):
        return self.joint

    def get_predictor(self):
        return self.decoder
