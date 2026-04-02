import torch

from wav2vec2.model import conformer_base

pretrained_ckpt = torch.load("/home/hdd2/jenny/AVKD/pretrained/hubert-large-ll60k.hf.pth")['state_dict']
num_layers = 3
# result_txt = "result.txt"
# with open(result_txt, 'w') as f:
#     f.write(f'{pretrained_ckpt.keys()}')
# f.close()
new_dict = {}
for k, v in pretrained_ckpt.items():
    if k.startswith('feature_extractor'):
        k = k.replace('feature_extractor.', '')
        new_dict[k] = v
pretrained_ckpt['state_dict'] = new_dict
torch.save(pretrained_ckpt, "/home/hdd2/jenny/AVKD/pretrained/hubert-large-frontend.pth")

student_config = dict(
    extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
    # extractor_conv_layer_config=[(256, 10, 5)] + [(256, 3, 2)] * 6  + [(256, 2, 2)] * 3,
    encoder_embed_dim=512,
    encoder_projection_dropout=0.1,
    encoder_pos_conv_kernel=128,
    encoder_pos_conv_groups=16,
    encoder_num_layers=num_layers,
    encoder_use_attention=[True] * num_layers,
    encoder_use_feed_forward=[True] * num_layers,
    encoder_num_heads=[16] * num_layers,
    encoder_head_dim=64,
    encoder_attention_dropout=0.1,
    encoder_ff_interm_features=[4096] * num_layers,
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
student_model = conformer_base(**student_config)
# for name, param in student_model.named_parameters():
#     print(name)
student_result = student_model.load_state_dict(pretrained_ckpt, strict=False)
res = student_model.feature_extractor.load_state_dict(pretrained_ckpt, strict=False)
print(res)
