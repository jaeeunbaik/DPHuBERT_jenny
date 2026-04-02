"""Convert fairseq's HuBERT to our format."""

import torch
import fairseq
from torchaudio.models.wav2vec2.utils import import_fairseq_model

from wav2vec2.model import wav2vec2_model, vhubert_large, hubert_large


if __name__ == "__main__":
    out_name = "pretrained/vhubert-large.pth"

    # fairseq_ckpt = "/bada/yh/jenny/DPHuBERT/zoo/avhubert/vhubert_large_edited.pt"
    fairseq_ckpt = "/home/hdd2/jenny/AVKD/pretrained/large_vox_iter5.pth"
    # fairseq_ckpt = "/bada/yh/jenny/DPHuBERT/zoo/avhubert/self_large_vox_433h_vsr.pt"
    # ensemble, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_ckpt])
    cfg = torch.load(fairseq_ckpt, map_location="cpu")['cfg']
    # original = ensemble[0]
    # imported = import_fairseq_model(original)
    # print(imported)

    # default config of hubert base
    hubert_large_config = dict(
        # extractor_mode="group_norm",    # hubert base only uses a group norm at the first conv layer
        # extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        # extractor_conv_bias=False,
        # encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        # encoder_pos_conv_kernel=128,
        # encoder_pos_conv_groups=16,
        # encoder_num_layers=12,
        # encoder_use_attention=[True] * 12,
        # encoder_use_feed_forward=[True] * 12,
        # encoder_num_heads=[12] * 12,
        # encoder_head_dim=64,
        encoder_attention_dropout=0.1,
        # encoder_ff_interm_features=[3072] * 12,
        # encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        # encoder_layer_norm_first=False,     # hubert base uses post norm
        encoder_layer_drop=0.05,
        aux_num_out=None,
        # normalize_waveform=False,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

    torch.save(
        {
            'state_dict': torch.load(fairseq_ckpt, map_location="cpu")['model'],
            'config': hubert_large_config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    # model = mrhubert_base(**ckpt['config'])
    model = vhubert_large(**ckpt['config'])
    # model = hubert_large(**ckpt['config'])
    res = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Missing: {res.missing_keys}\nUnexpected: {res.unexpected_keys}")
    txt_path = "result.txt"
    with open(txt_path, "w") as f:
        f.write(f"Missing: {res.missing_keys}\nUnexpected: {res.unexpected_keys}")
    f.close()