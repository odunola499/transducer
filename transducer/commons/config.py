from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr


class Args(BaseModel):
    extra_kwargs: Dict[str, Any] = {}

    def to_dict(self):
        return self.model_dump()

    def to_json(self):
        return self.model_dump_json()


class EncoderConfig(Args):
    attn_impl: Literal["flash_attn", "sdpa", "math"] = "sdpa"

    @property
    def embed_dim(self):
        return getattr(self, "hidden_size", None)


class DecoderConfig(Args):
    rnn_type: Literal["gru", "lstm"] = "gru"
    embed_dim: StrictInt
    hidden_dim: StrictInt = 1024
    pred_dim: StrictInt = 640
    joint_dim: StrictInt = 640
    num_layers: StrictInt
    dropout: StrictFloat
    vocab_size: StrictInt = 1024


class ModelConfig(Args):
    model_name: StrictStr
    loss_type: Literal["tdt", "tdt_triton", "rnnt", "rnnt_triton"] = "rnnt"
    loss_duration: Optional[list[StrictInt]] = None
    fastemit_lambda: StrictFloat = 0.0
    blank_id: Optional[StrictInt] = None
    loss_reduction: Literal["sum", "mean"] = "sum"
    sampler_type: Literal["greedy_search", "beam_search"] = "greedy_search"
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
