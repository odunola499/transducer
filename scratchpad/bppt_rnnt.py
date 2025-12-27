import torch
import torch.nn as nn
import torch.nn.functional as F

LOG0 = -1e30  # practical -inf for log-space

def logsumexp2(a, b):
    # stable logsumexp for two terms
    m = torch.maximum(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))

class RNNTScanLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, enc, pred, labels, enc_lens, label_lens, blank_id: int, joiner: nn.Module):
        """
        enc: [B, T, Denc] float, requires_grad=True
        pred: [B, U+1, Dpred] float, requires_grad=True
        labels: [B, U] long
        enc_lens: [B] long
        label_lens: [B] long
        joiner: module mapping (enc_t:[B,Denc], pred:[B,U+1,Dpred]) -> logits [B,U+1,V]
        """

        device = enc.device
        B, T, _ = enc.shape
        _, Up1, _ = pred.shape
        U = Up1 - 1

        # Save tensors for backward (we will recompute logits in backward)
        ctx.save_for_backward(enc, pred, labels, enc_lens, label_lens)
        ctx.blank_id = blank_id
        ctx.joiner = joiner

        # We will compute a scalar loss per sample, then average.
        losses = torch.zeros(B, device=device, dtype=enc.dtype)

        # For simplicity: do per-sample loops over batch lengths.
        # (Real kernels fuse/parallelize this.)
        for b in range(B):
            Tb = int(enc_lens[b].item())
            Ub = int(label_lens[b].item())

            # alpha for current time row: size Ub+1 (we ignore padding tail)
            alpha = torch.full((Ub + 1,), LOG0, device=device, dtype=enc.dtype)
            alpha[0] = 0.0

            # Scan over time
            for t in range(Tb):
                # Compute logits for this timestep, only for needed u positions [0..Ub]
                # pred[b, :Ub+1] shape [Ub+1, Dpred]
                logits = joiner(enc[b, t], pred[b, :Ub+1])  # [Ub+1, V]
                logp = F.log_softmax(logits, dim=-1)        # [Ub+1, V]

                blank_lp = logp[:, blank_id]               # [Ub+1]

                # label logp for u in 0..Ub-1 (transition from u -> u+1)
                # for u = Ub, there's no next label, treat as -inf.
                if Ub > 0:
                    next_labels = labels[b, :Ub]           # [Ub]
                    label_lp = logp[:Ub, :].gather(1, next_labels[:, None]).squeeze(1)  # [Ub]
                else:
                    label_lp = torch.empty((0,), device=device, dtype=enc.dtype)

                # DP to compute alpha_next:
                # alpha_next[u] = logsumexp( alpha_next[u-1] + label_lp[u-1], alpha[u] + blank_lp[u] )
                alpha_next = torch.full((Ub + 1,), LOG0, device=device, dtype=enc.dtype)

                # u=0 only reachable via blanks from (t,0)
                alpha_next[0] = alpha[0] + blank_lp[0]

                for u in range(1, Ub + 1):
                    from_blank = alpha[u] + blank_lp[u]
                    from_label = alpha_next[u - 1] + label_lp[u - 1]  # u-1 must exist
                    alpha_next[u] = logsumexp2(from_blank, from_label)

                alpha = alpha_next

            # termination: after Tb time steps, must emit remaining blanks to finish at (Tb, Ub)
            # In standard RNNT, the final probability includes a blank at (Tb, Ub) in some conventions.
            # Here, we assume the last step already accounted transitions through time;
            # common convention adds blank at terminal (Tb, Ub-?) - implementations vary.
            # We'll use a typical "end with blank after last frame" by applying joiner at last frame:
            # (This is a detail you must align with your reference implementation!)
            if Tb > 0:
                logits_T = joiner(enc[b, Tb - 1], pred[b, :Ub+1])
                logp_T = F.log_softmax(logits_T, dim=-1)
                terminal_blank = logp_T[Ub, blank_id]
            else:
                terminal_blank = torch.tensor(0.0, device=device, dtype=enc.dtype)

            loglike = alpha[Ub] + terminal_blank
            losses[b] = -loglike

        return losses.mean()

    @staticmethod
    def backward(ctx, grad_out):
        """
        Compute grads w.r.t enc and pred by:
        - recomputing logits per t
        - recomputing local alpha/beta (or equivalently compute posteriors)
        - forming grad_logits
        - using autograd on joiner/log_softmax to get grad enc_t and grad pred_u
        """
        enc, pred, labels, enc_lens, label_lens = ctx.saved_tensors
        blank_id = ctx.blank_id
        joiner = ctx.joiner

        device = enc.device
        B, T, _ = enc.shape
        _, Up1, _ = pred.shape

        grad_enc = torch.zeros_like(enc)
        grad_pred = torch.zeros_like(pred)

        # scale from outer reduction (mean)
        scale = grad_out / B

        # We do per-sample backward.
        for b in range(B):
            Tb = int(enc_lens[b].item())
            Ub = int(label_lens[b].item())

            # --- Recompute forward alphas for all t as checkpoints (time-only) ---
            # To keep code readable, store alpha rows for this sample.
            # In the "paper-style" memory save, you'd checkpoint only every K and recompute within blocks.
            alphas = [None] * (Tb + 1)
            alpha = torch.full((Ub + 1,), LOG0, device=device, dtype=enc.dtype)
            alpha[0] = 0.0
            alphas[0] = alpha

            for t in range(Tb):
                logits = joiner(enc[b, t], pred[b, :Ub+1])  # [Ub+1, V]
                logp = F.log_softmax(logits, dim=-1)

                blank_lp = logp[:, blank_id]
                if Ub > 0:
                    next_labels = labels[b, :Ub]
                    label_lp = logp[:Ub, :].gather(1, next_labels[:, None]).squeeze(1)
                else:
                    label_lp = torch.empty((0,), device=device, dtype=enc.dtype)

                alpha_next = torch.full((Ub + 1,), LOG0, device=device, dtype=enc.dtype)
                alpha_next[0] = alpha[0] + blank_lp[0]
                for u in range(1, Ub + 1):
                    alpha_next[u] = logsumexp2(alpha[u] + blank_lp[u], alpha_next[u-1] + label_lp[u-1])
                alpha = alpha_next
                alphas[t + 1] = alpha

            # terminal blank convention (must match forward)
            if Tb > 0:
                logits_T = joiner(enc[b, Tb - 1], pred[b, :Ub+1])
                logp_T = F.log_softmax(logits_T, dim=-1)
                terminal_blank = logp_T[Ub, blank_id]
            else:
                terminal_blank = torch.tensor(0.0, device=device, dtype=enc.dtype)

            loglike = alphas[Tb][Ub] + terminal_blank  # scalar

            # --- Backward: compute betas sequentially over time (explicit DP) ---
            # beta[t,u] = log prob of completing from (t,u) to end
            # We'll compute beta rows backward over time.
            beta_next = torch.full((Ub + 1,), LOG0, device=device, dtype=enc.dtype)
            beta_next[Ub] = terminal_blank  # at t=Tb, u=Ub (terminal)
            # other beta_next[u<Ub] should be -inf because can't emit labels after time ends (in this simple convention)

            for t in reversed(range(Tb)):
                # recompute logp at this t
                # IMPORTANT: make enc_t and pred slice require grad for local autograd
                enc_t = enc[b, t].detach().requires_grad_(True)            # [Denc]
                pred_u = pred[b, :Ub+1].detach().requires_grad_(True)      # [Ub+1, Dpred]

                logits = joiner(enc_t, pred_u)                              # [Ub+1, V]
                logp = F.log_softmax(logits, dim=-1)

                blank_lp = logp[:, blank_id]                                # [Ub+1]
                if Ub > 0:
                    next_labels = labels[b, :Ub]
                    label_lp = logp[:Ub, :].gather(1, next_labels[:, None]).squeeze(1)  # [Ub]
                else:
                    label_lp = torch.empty((0,), device=device, dtype=enc.dtype)

                # compute beta row at time t
                beta = torch.full((Ub + 1,), LOG0, device=device, dtype=enc.dtype)

                # At (t,u), you can:
                # - emit blank to (t+1,u): contributes blank_lp[u] + beta_next[u]
                # - emit label to (t,u+1): contributes label_lp[u] + beta[u+1]
                beta[Ub] = blank_lp[Ub] + beta_next[Ub]  # only blank possible at u=Ub
                for u in reversed(range(Ub)):
                    from_blank = blank_lp[u] + beta_next[u]
                    from_label = label_lp[u] + beta[u + 1]
                    beta[u] = logsumexp2(from_blank, from_label)

                # Now compute posteriors for transitions originating at time t:
                # gamma_blank(t,u) ∝ exp(alpha[t,u] + blank_lp[u] + beta_next[u] - loglike)
                # gamma_label(t,u) ∝ exp(alpha[t,u] + label_lp[u] + beta[u+1] - loglike)
                alpha_t = alphas[t]  # [Ub+1]

                # For stability, work in log then exp.
                log_gamma_blank = alpha_t + blank_lp + beta_next - loglike
                gamma_blank = torch.exp(log_gamma_blank)                   # [Ub+1]

                if Ub > 0:
                    log_gamma_label = alpha_t[:Ub] + label_lp + beta[1:] - loglike
                    gamma_label = torch.exp(log_gamma_label)               # [Ub]
                else:
                    gamma_label = torch.empty((0,), device=device, dtype=enc.dtype)

                # Gradient w.r.t logp entries is -posterior (since loss = -loglike)
                # Then propagate through log_softmax -> logits via autograd.
                grad_logp = torch.zeros_like(logp)                         # [Ub+1, V]
                grad_logp[:, blank_id] -= gamma_blank
                if Ub > 0:
                    grad_logp[:Ub, :].scatter_add_(1, next_labels[:, None], -gamma_label[:, None])

                # scale and backprop local graph to enc_t and pred_u
                (g_enc_t, g_pred_u) = torch.autograd.grad(
                    outputs=logp,
                    inputs=(enc_t, pred_u),
                    grad_outputs=grad_logp * scale,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                grad_enc[b, t] += g_enc_t
                grad_pred[b, :Ub+1] += g_pred_u

                beta_next = beta

        # No grads for labels/lengths/joiner arg (joiner params are NOT handled here)
        # If you want joiner parameter grads too, you must include joiner parameters as inputs
        # or compute joiner outside the custom Function and only custom-backward the DP part.
        return grad_enc, grad_pred, None, None, None, None, None


class RNNTScanLoss(nn.Module):
    def __init__(self, joiner: nn.Module, blank_id: int = 0):
        super().__init__()
        self.joiner = joiner
        self.blank_id = blank_id

    def forward(self, enc, pred, labels, enc_lens, label_lens):
        return RNNTScanLossFn.apply(enc, pred, labels, enc_lens, label_lens, self.blank_id, self.joiner)
