import torch
from torch import Tensor
from typing import List


from transducer.samplers.base import BaseSampler
from transducer.commons import Hypothesis

class BeamSearchSampler(BaseSampler):
    def __init__(self, max_symbols_per_timestep = 5, blank_id = 0, beam_size = 4):
        super().__init__(
            max_symbols_per_timestep = max_symbols_per_timestep,
            blank_id=blank_id
        )
        self.beam_size = beam_size

    def offline_decode(self, audio:Tensor):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        encoder = self.get_encoder()
        predictor = self.get_predictor()

        features = feature_extractor(audio)
        encoder_output = encoder(features)

        batch_size, frame_size = encoder_output.shape[:-1]
        start_ids = torch.full((batch_size, 1), fill_value=self.blank_id, dtype = torch.long, device = encoder_output.device)

        pred_state = predictor.init_state(batch_size)
        pred_out, pred_state = predictor.step(start_ids, state = pred_state)

        hyps = [
            Hypothesis(
                tokens = torch.empty((batch_size, 0), dtype = torch.long, device = encoder_output.device),
                pred_out = pred_out,
                pred_state = pred_state,
                score = torch.zeros(batch_size, 1, device = encoder_output.device),
            )
        ]

        for t in range(frame_size):
            frame = encoder_output[:, t:t+1, :]
            hyps = self.decode_frame(frame, hyps)


    def decode_frame(self, frame, hyps:List[Hypothesis]):
        predictor = self.get_predictor()
        joiner = self.get_joiner()

        if frame.ndim == 2:
            frame = frame.unsqueeze(1)

        new_hyps = []
        for hyp in hyps:
            logits = joiner(frame, hyp.pred_out)
            logp = torch.log_softmax(logits, dim = -1)

            blank_score = logp[..., self.blank_id].squeeze(-1).squeeze(-1)
            hyp.score = hyp.score + blank_score
            new_hyps.append(
                hyp
            )

            topk_scores, topk_ids = logp.squeeze(1).squeeze(1).topk(
                self.beam_size + 1, dim = -1
            )

            for i in range(topk_ids.size(-1)):
                token = topk_ids[:, i]
                if (token == self.blank_id).all():
                    continue

                pred_out, pred_state = predictor.step(token, state=hyp.pred_state)

                new_hyps.append(
                    Hypothesis(
                        tokens=torch.cat([hyp.tokens, token.unsqueeze(-1)], dim=-1),
                        pred_out=pred_out,
                        pred_state=pred_state,
                        score=hyp.score + topk_scores[:, i],
                    )
                )
        new_hyps = sorted(
            new_hyps,
            key = lambda hyp: hyp.score.mean().item(),
            reverse = True
        )[:self.beam_size]
        return new_hyps





