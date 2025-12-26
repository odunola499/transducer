import torch
from torch import Tensor
from typing import Optional

from transducer.models.base import BaseModel
from transducer.models.config import ModelConfig, Wav2Vec2BertConfig, Wav2VecConfig
from transducer.losses import LOSSES
from transducer.models.decoder.rnnt import SimpleJoiner, RNNPredictor
from transducer.models.encoder.wav2vec import Wav2VecModel
from transducer.models.encoder.wav2vec2bert import Wav2Vec2BertModel
from transducer.samplers import GenerationMixin

class DawnModel(BaseModel, GenerationMixin):
    def __init__(self, vocab_size:int, config: ModelConfig):
        super().__init__(config)
        self.model_name = config.model_name
        # Place blank token at the end of the vocabulary space
        base_vocab_size = config.decoder_config.vocab_size
        self.blank_id = base_vocab_size if config.blank_id is None else config.blank_id
        if self.blank_id != base_vocab_size:
            raise ValueError(
                f"blank_id ({self.blank_id}) must equal decoder vocab_size ({base_vocab_size})"
            )
        self.vocab_size_with_blank = base_vocab_size + 1

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

        # Select encoder based on provided config class
        if isinstance(encoder_config, Wav2Vec2BertConfig):
            self.encoder = Wav2Vec2BertModel(encoder_config)
        elif isinstance(encoder_config, Wav2VecConfig):
            self.encoder = Wav2VecModel(encoder_config)
        else:
            raise ValueError(f"Unsupported encoder_config type: {type(encoder_config)}")
        self.predictor = RNNPredictor(decoder_config, vocab_size=self.vocab_size_with_blank)
        self.joiner = SimpleJoiner(
            self.vocab_size_with_blank,
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
        lattice = lattice.float()
        if self.config.loss_type in {"rnnt", "rnnt_triton"}:
            self._assert_rnnt_inputs(lattice, labels, act_lens, label_lens)
        with torch.autocast(device_type=lattice.device.type, enabled=False):
            return self.loss_func(
                acts=lattice,
                labels=labels,
                label_lens=label_lens,
                act_lens=act_lens,
            )

    def _assert_rnnt_inputs(
        self,
        lattice: Tensor,
        labels: Tensor,
        act_lens: Optional[Tensor],
        label_lens: Tensor,
    ) -> None:
        if lattice.dim() != 4:
            raise RuntimeError(f"RNNT logits must be 4D, got shape {tuple(lattice.shape)}")
        if not torch.isfinite(lattice).all().item():
            raise RuntimeError("RNNT logits contain NaN/Inf values.")

        batch, max_t, max_u, vocab = lattice.shape
        if vocab != self.vocab_size_with_blank:
            raise RuntimeError(
                "RNNT logits last dim must match vocab_size_with_blank "
                f"({vocab} != {self.vocab_size_with_blank})."
            )
        if labels.dim() != 2 or labels.shape[0] != batch:
            raise RuntimeError(
                f"RNNT labels must be [B, U], got shape {tuple(labels.shape)}"
            )
        if labels.device != lattice.device:
            raise RuntimeError("RNNT labels must be on the same device as logits.")
        if label_lens is None:
            raise RuntimeError("label_lens is required for RNNT loss.")
        if label_lens.dim() != 1 or label_lens.shape[0] != batch:
            raise RuntimeError(
                f"label_lens must be shape [B], got shape {tuple(label_lens.shape)}"
            )
        if label_lens.device != lattice.device:
            raise RuntimeError("label_lens must be on the same device as logits.")

        def _assert_no(condition: Tensor, message: str) -> None:
            if condition.any().item():
                raise RuntimeError(message)

        _assert_no(labels < 0, "RNNT labels must be >= 0.")
        _assert_no(
            labels >= self.blank_id,
            f"RNNT labels must be < blank_id ({self.blank_id}).",
        )
        _assert_no(label_lens < 0, "label_lens must be >= 0.")
        _assert_no(label_lens >= max_u, "label_lens must be <= U-1.")
        _assert_no(
            label_lens > labels.shape[1],
            "label_lens exceeds labels sequence length.",
        )

        if act_lens is None:
            act_lens = torch.full(
                (batch,), max_t, device=lattice.device, dtype=torch.long
            )
        elif act_lens.device != lattice.device:
            raise RuntimeError("act_lens must be on the same device as logits.")
        elif act_lens.dim() != 1 or act_lens.shape[0] != batch:
            raise RuntimeError(
                f"act_lens must be shape [B], got shape {tuple(act_lens.shape)}"
            )

        _assert_no(act_lens <= 0, "act_lens must be >= 1.")
        _assert_no(act_lens > max_t, "act_lens must be <= T.")
