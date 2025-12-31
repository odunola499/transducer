import torch
from torch import nn
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
model_name="nvidia/parakeet_realtime_eou_120m-v1"
).to(device).eval()

encoder = asr_model.encoder
B, T = 4, 300

audio_signal, audio_lens = encoder.input_example(max_batch = B, max_dim=T)
audio_signal = audio_signal.to(device)
audio_lens = audio_lens.to(device)

print('init input',audio_signal.shape)
print('init input',audio_lens)

output_audio_signal, output_audio_lens = encoder.forward_internal(
audio_signal = audio_signal,
length = audio_lens
)
print('final_output',output_audio_signal.shape)
print('final_output_lens',output_audio_lens)


curr_att_context_size = encoder.att_context_size
att_context_style = encoder.att_context_style

print('curr att context length',curr_att_context_size)
print('attn context style',att_context_style)

pre_encode = encoder.pre_encode
pre_encode_output, pre_encode_lens = pre_encode(x = audio_signal.transpose(1,2), lengths = audio_lens)
print(pre_encode_output.shape)
print(pre_encode_lens)

max_audio_length = pre_encode_output.shape[1]

print(encoder.streaming_cfg)
print(encoder.export_cache_support)
print(encoder.is_access_enabled(getattr(encoder, 'model_guid', None)))

cache_len = encoder.streaming_cfg.last_channel_cache_size
cache_keep_size = max_audio_length - encoder.streaming_cfg.cache_drop_size
max_audio_length = max_audio_length + cache_len
padding_length = pre_encode_lens + cache_len