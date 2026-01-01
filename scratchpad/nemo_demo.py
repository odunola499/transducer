import nemo.collections.asr as nemo_asr
import torch

from transducer.models.parakeet.config import FastConformerConfig
from transducer.models.parakeet.encoder import ConformerEncoder


def print_params(module: torch.nn.Module):
    return sum([p.numel() for p in module.parameters()])


def compare_tensors(
    task: str,
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    label_a: str = "a",
    label_b: str = "b",
):
    same_shape = out_a.shape == out_b.shape
    max_diff = (out_a - out_b).abs().max().item() if same_shape else float("nan")
    close = same_shape and torch.allclose(out_a, out_b)
    print(
        f"{task}: allclose={close}, "
        f"{label_a}_shape={tuple(out_a.shape)}, {label_b}_shape={tuple(out_b.shape)}, "
        f"max_abs_diff={max_diff}"
    )
    return close


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name="nvidia/parakeet_realtime_eou_120m-v1"
    ).to(device)

    asr_feature_extractor = asr_model.preprocessor.featurizer
    encoder = asr_model.encoder.eval()
    print("asr_encoder params", print_params(encoder))

    audio = torch.randn(4, 64000, device=device)
    lengths = torch.tensor([64000, 54000, 6400, 64000], device=device).to(torch.long)

    features, feature_lengths = asr_feature_extractor(audio, lengths.clone())

    config = FastConformerConfig()
    config.att_context_size = encoder.att_context_size
    config.att_context_style = encoder.att_context_style
    config.attn_impl = "math"
    conf_encoder = ConformerEncoder(config=config).eval().to(device)
    conf_encoder.load_state_dict(encoder.state_dict(), strict=True)
    print("conf_encoder params", print_params(conf_encoder))

    with torch.no_grad():
        encoder_outputs, nemo_forward_lengths = encoder(
            audio_signal=features, length=feature_lengths.clone()
        )
        conf_encoder_outputs, conf_forward_lengths = conf_encoder(
            features, length=feature_lengths.clone(), return_lengths=True
        )

    print(f"encoder_output {encoder_outputs.shape}")
    print(f"conf_encoder_output {conf_encoder_outputs.shape}")
    compare_tensors(
        "forward_output", encoder_outputs, conf_encoder_outputs, "nemo", "conf"
    )
    compare_tensors(
        "forward_lengths", nemo_forward_lengths, conf_forward_lengths, "nemo", "conf"
    )

    nemo_pre_encode = encoder.pre_encode
    conf_pre_encode = conf_encoder.pre_encode

    nemo_lengths = feature_lengths.clone()
    conf_lengths = feature_lengths.clone()

    nemo_sub_out, nemo_lengths = nemo_pre_encode(
        x=features.transpose(1, 2), lengths=nemo_lengths
    )
    conf_sub_out, conf_lengths = conf_pre_encode(features.transpose(1, 2), conf_lengths)
    compare_tensors("pre_encode_out", nemo_sub_out, conf_sub_out, "nemo", "conf")
    compare_tensors("pre_encode_len", nemo_lengths, conf_lengths, "nemo", "conf")

    nemo_pad_mask, nemo_att_mask = encoder._create_masks(
        att_context_size=encoder.att_context_size,
        padding_length=nemo_lengths,
        max_audio_length=nemo_sub_out.size(1),
        offset=None,
        device=device,
    )
    conf_pad_mask, conf_att_mask = conf_encoder._create_masks(
        padding_length=conf_lengths,
        max_audio_length=conf_sub_out.size(1),
        device=device,
    )

    nemo_pos = encoder.pos_enc
    conf_pos = conf_encoder.pos_enc

    nemo_pos_out, nemo_pos_emb = nemo_pos(nemo_sub_out)
    conf_pos_out, conf_pos_emb = conf_pos(conf_sub_out)
    compare_tensors("pos_enc_out", nemo_pos_out, conf_pos_out, "nemo", "conf")
    compare_tensors("pos_emb", nemo_pos_emb, conf_pos_emb, "nemo", "conf")

    nemo_layers = getattr(encoder, "layers", None)

    nemo_x = nemo_pos_out
    conf_x = conf_pos_out
    for idx, (nemo_layer, conf_layer) in enumerate(
        zip(nemo_layers, conf_encoder.layers)
    ):
        nemo_x = nemo_layer(
            x=nemo_x, pos_emb=nemo_pos_emb, pad_mask=nemo_pad_mask, att_mask=nemo_att_mask
        )
        conf_x = conf_layer(
            x=conf_x, pos_emb=conf_pos_emb, pad_mask=conf_pad_mask, att_mask=conf_att_mask
        )
        compare_tensors(f"layer_{idx}_output", nemo_x, conf_x, "nemo", "conf")

    compare_tensors(
        "final_encoder_outputs", encoder_outputs, conf_encoder_outputs, "nemo", "conf"
    )

    with torch.no_grad():
        nemo_out, nemo_len = encoder(
            audio_signal=features, length=feature_lengths.clone()
        )
        conf_out, conf_len = conf_encoder(
            features, length=feature_lengths.clone(), return_lengths=True
        )

    diff = (nemo_out - conf_out).abs()
    max_total = diff.max()
    mask_valid = (
        torch.arange(conf_out.size(-1), device=conf_out.device)
        .unsqueeze(0)
        .expand(conf_len.size(0), -1)
    ) < conf_len.unsqueeze(1)  # True where frames are valid
    max_valid = diff.permute(0, 2, 1)[
        mask_valid
    ].max()  # permute to (B,T,C) to mask over T
    print(max_total, max_valid)


if __name__ == "__main__":
    main()
