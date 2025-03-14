import os
import time
import torch
from wav2vec2.model import (
    hubert_large,
    conformer_base,
    vhubert_large,
)
import torchsummary
import numpy as np
import random

device = torch.device("cuda")

num_layers = 3
conformer_config = dict(
    extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
    # extractor_conv_layer_config=[(256, 10, 5)] + [(256, 3, 2)] * 6  + [(256, 2, 2)] * 3,
    encoder_embed_dim=768,
    encoder_projection_dropout=0.1,
    encoder_pos_conv_kernel=128,
    encoder_pos_conv_groups=12,
    encoder_num_layers=num_layers,
    encoder_use_attention=[True] * num_layers,
    encoder_use_feed_forward=[True] * num_layers,
    encoder_num_heads=[12] * num_layers,
    encoder_head_dim=64,
    encoder_attention_dropout=0.1,
    encoder_ff_interm_features=[3072] * num_layers,
    encoder_ff_interm_dropout=0.0,
    encoder_dropout=0.1,
    encoder_layer_drop=0.1,
    aux_num_out=None,
    normalize_waveform=False,
    extractor_prune_conv_channels=False,
    encoder_prune_attention_heads=False,
    encoder_prune_attention_layer=False,
    encoder_prune_feed_forward_intermediate=False,
    encoder_prune_feed_forward_layer=False,
    modality="audio",
)
conformer_a = conformer_base(**conformer_config).to(device)

# Dummy Input
dummy_audio = torch.randn(1, 80000)
dummy_video = torch.randn(1, 125, 1, 88, 88) # (B, T, C, H, W), grayscasle ver.

# Inference time
start_time = time.time()
for _ in range(100):
    # hubert_large(dummy_audio.to(device))
    # vhubert_large(dummy_video.to(device))
    conformer_a(dummy_audio.to(device))
    # conformer_v(dummy_video.to(device))
end_time = time.time()
print(f"inf time : {end_time - start_time}s")
original_num_params = sum(p.numel() for p in conformer_a.parameters())

txt_path = "result.txt"
with open(txt_path, 'w') as f:
    for name,param in conformer_a.named_parameters():
        f.write(f'({name}, {param.shape})')

print(f"original_num_params: {original_num_params}")