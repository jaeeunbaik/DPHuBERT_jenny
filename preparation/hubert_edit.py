import torch

from wav2vec2.model import wav2vec2_model, mrhubert_large

mrhubert_ckpt = torch.load("./zoo/avhubert/self_large_vox_433h_vsr.pt", map_location="cpu")
output_ckpt = "./zoo/avhubert/vhubert_large_edited.pt"
state_dict = mrhubert_ckpt['model']

new_state_dict = {}

for k, v in state_dict.items():
    if k.startswith('encoder.w2v_model.'):
	    k = k.replace('encoder.w2v_model.', '')
    k = k.replace('feature_extractor_video.resnet.', 'feature_extractor.')
    k = k.replace('post_extract_proj', 'encoder.feature_projection.projection')
    if k.startswith('encoder.pos_conv.'):
        k = k.replace('encoder.pos_conv.0.', 'encoder.transformer.pos_conv_embed.conv.')
    if k.startswith('encoder.layers.'):
        k = k.replace('encoder.layers.', 'encoder.transformer.layers.')
    k = k.replace('self_attn', 'attention')
    k = k.replace('attention_layer_norm', 'layer_norm')
    k = k.replace('fc1', 'feed_forward.intermediate_dense')
    k = k.replace('fc2', 'feed_forward.output_dense')
    if k.startswith('post_extract_proj.'):
        k = k.replace('post_extract_proj.', 'encoder.feature_projection.projection.')
    if k.startswith('layer_norm.'):
        k = k.replace('layer_norm.', 'encoder.feature_projection.layer_norm.')
    if k.startswith('encoder.layer_norm.'):
        k = k.replace('encoder.layer_norm.', 'encoder.transformer.layer_norm.')
    if k.startswith('feature_extractor_video.proj'):
        k = k.replace('feature_extractor_video.proj', 'feature_extractor.proj')
    new_state_dict[k] = v

mrhubert_ckpt['model'] = new_state_dict
torch.save(mrhubert_ckpt, output_ckpt)