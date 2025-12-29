import nemo.collections.asr as nemo_asr
import torch

from transducer.models.encoder.fast_conformer import (
    ConformerEncoder,
    FastConformerConfig,
)

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="nvidia/parakeet_realtime_eou_120m-v1"
)

asr_feature_extractor = asr_model.preprocessor.featurizer
encoder = asr_model.encoder.eval()

audio = torch.randn(2, 64000)
lengths = torch.tensor([64000, 64000]).to(torch.long)

features, lengths = asr_feature_extractor(audio, lengths)
encoder_outputs, _ = encoder(audio_signal=features, length=lengths)

config = FastConformerConfig()
conf_encoder = ConformerEncoder(config=config).eval()
conf_encoder.load_state_dict(encoder.state_dict())
conf_encoder_outputs = conf_encoder(features)

print(torch.allclose(encoder_outputs, conf_encoder_outputs))
