import torch
import torch.nn as nn

"""
FVTA Pytorch Model
Default parameters:
hidden_size 50 --image_feat_dim 2537 \
--use_image_trans --image_trans_dim 100 --use_char --char_emb_size 100 --char_out_size 100 --add_tanh \
--simiMatrix 2 --use_question_att --use_3d --use_time_warp --warp_type 5 --is_test --load_best \
--batch_size 1
"""
class FVTA(nn.Model):
    def __init__(self, hidden_size=50, image_feat_dim=2537, use_image_trans=True, images_trans_dim=100,
                use_char=True, char_emb_size=100, char_out_size=100, add_tanh=True, simiMatrix=2, use_question_att=True, use_3d=True, use_time_warp=True, warp_type=5):
        super().__init__()
