import numpy as np
import torch
from torch import Tensor
from torchaudio.models.wav2vec2.components import FeatureExtractor

from transducer.commons import GenerationOutput
from transducer.samplers.base import BaseSampler
from typing import Union

class GreedySearchSampler(BaseSampler):
    def __init__(self, max_symbols_per_timestep = 5, blank_id = 0):
        super().__init__(
            max_symbols_per_timestep = max_symbols_per_timestep,
            blank_id=blank_id
        )

    def offline_decode(self, audio:Tensor):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        encoder = self.get_encoder()
        predictor = self.get_predictor()

        features = feature_extractor(audio)
        encoder_output = encoder(features)

        batch_size, frame_size = encoder_output.shape[:-1]
        pred_state = predictor.init_state(batch_size)
        start_ids = torch.full((batch_size,), self.blank_id, dtype=torch.long, device=encoder_output.device)
        pred_out, pred_state = predictor.step(start_ids, state=pred_state)

        t = 0
        results = []
        while t < frame_size:
            frame = encoder_output[:, t:t + 1, :]
            hyp, pred_out, pred_state = self.decode_frame(frame, pred_out, pred_state)
            t += 1
            results.append(hyp)
        results = torch.concat(results, dim = -1)
        texts = tokenizer.decode(results)

        return GenerationOutput(
            ids = results,
            labels=texts
        )


    def decode_frame(self, frame:Tensor, pred_out:Tensor, pred_state:Tensor):
        predictor = self.get_predictor()
        joiner = self.get_joiner()

        if frame.dim() == 2:
            frame = frame.unsqueeze(1)

        symb = self.max_symbols_per_timestep
        hyp = []
        while symb > 0:
            symb -= 1
            logits = joiner(frame, pred_out)
            ids = logits.argmax(-1)
            ids = ids.squeeze(-1).squeeze(-1)
            if (ids == self.blank_id).all():
                break

            hyp.append(ids.unsqueeze(-1))
            pred_out, pred_state = predictor.step(ids, state = pred_state)

        if hyp:
            hyp = torch.concat(hyp, dim = -1)
        else:
            hyp = torch.empty((frame.shape[0], 0), dtype=torch.long, device=frame.device)
        return hyp, pred_out, pred_state