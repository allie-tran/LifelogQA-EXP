import sys
from config import *
import torch
import torch.nn as nn
from utils import LSTM, Linear, populate_tensors
import torch.nn.functional as F

sys.path.append('../')


def get_char(char, batch):
    d = batch.shared['char2idx']
    if char in d:
        return d[char]
    return 1


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()


class BiDAF(nn.Module):
    def __init__(self):
        super(BiDAF, self).__init__()

        self.N = params['batch_size']
        self.VW = params['word_vocab_size']
        self.VC = params['char_vocab_size']
        self.W = params['max_word_size']

        # embedding dim
        self.cd = params['char_emb_size']
        self.wd = params['word_emb_size']
        self.cwd = params['char_out_size']

        self.idim = params['image_feat_dim']
        self.num_choice = 4

        # album title
        # [N,M,JXA]
        self.at = torch.zeros((self.N, params['max_num_albums'], params['max_sent_album_title_size']),
                              dtype=torch.int32)
        self.at_c = torch.zeros((self.N, params['max_num_albums'], params['max_sent_album_title_size'], self.W),
                                dtype=torch.int32)
        self.at = torch.zeros((self.N, params['max_num_albums'], params['max_sent_album_title_size']), dtype=torch.bool)

        # album description
        # [N,M,JD]
        self.ad = torch.zeros((self.N, params['max_num_albums'], params['max_sent_des_size']), dtype=torch.int32)
        self.ad_c = torch.zeros((self.N, params['max_num_albums'], params['max_sent_des_size'], self.W),
                                dtype=torch.int32)
        self.ad_mask = torch.zeros((self.N, params['max_num_albums'], params['max_sent_des_size']), dtype=torch.bool)

        # album when, where
        # [N,M,JT/JG]
        self.when = torch.zeros((self.N, params['max_num_albums'], params['max_when_size']), dtype=torch.int32)
        self.when_c = torch.zeros((self.N, params['max_num_albums'], params['max_when_size'], self.W),
                                  dtype=torch.int32)
        self.when_mask = torch.zeros((self.N, params['max_num_albums'], params['max_when_size']), dtype=torch.bool)

        self.where = torch.zeros((self.N, params['max_num_albums'], params['max_where_size']), dtype=torch.int32)
        self.where_c = torch.zeros((self.N, params['max_num_albums'], params['max_where_size'], self.W),
                                   dtype=torch.int32)
        self.where_mask = torch.zeros((self.N, params['max_num_albums'], params['max_where_size']), dtype=torch.bool)

        # photo titles
        # [N,M,JI,JXP]
        self.pts = torch.zeros((self.N, params['max_num_albums'], params['max_num_photos'],
                                params['max_sent_photo_title_size']), dtype=torch.int32)
        self.pts_c = torch.zeros((self.N, params['max_num_albums'], params['max_num_photos'],
                                  params['max_sent_photo_title_size'], self.W), dtype=torch.int32)
        self.pts_mask = torch.zeros((self.N, params['max_num_albums'], params['max_num_photos'],
                                     params['max_sent_photo_title_size']), dtype=torch.bool)

        # photo
        # [N,M,JI] # each is a photo index
        self.pis = torch.zeros((self.N, params['max_num_albums'], params['max_num_photos']), dtype=torch.int32)
        self.pis_mask = torch.zeros((self.N, params['max_num_albums'], params['max_num_photos']), dtype=torch.bool)

        # question
        # [N,JQ]
        self.q = torch.zeros((self.N, params['max_question_size']), dtype=torch.int32)
        self.q_c = torch.zeros((self.N, params['max_question_size'], self.W), dtype=torch.int32)
        self.q_mask = torch.zeros((self.N, params['max_question_size']), dtype=torch.bool)

        # answer + choice words
        # [N,4,JA]
        self.choices = torch.zeros((self.N, self.num_choice, params['max_answer_size']), dtype=torch.int32)
        self.choices_c = torch.zeros((self.N, self.num_choice, params['max_answer_size'], self.W), dtype=torch.int32)
        self.choices_mask = torch.zeros((self.N, self.num_choice, params['max_answer_size']), dtype=torch.bool)

        # 4 choice classification
        self.y = torch.zeros((self.N, self.num_choice), dtype=torch.bool)

        # feed in the pretrain word vectors for all batch
        self.existing_emb_mat = torch.zeros((self.VW, params['word_emb_size']), dtype=torch.float)

        # feed in the image feature for this batch
        # [photoNumForThisBatch,image_dim]
        # self.image_emb_mat = tf.placeholder("float", [None, config.image_feat_dim], name="image_emb_mat")

        self.is_train = params['is_train']

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(self.VC, self.cd, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Sequential(
            nn.Conv2d(1, params['char_channel_size'], (self.cd, params['char_channel_width'])),
            nn.ReLU()
        )

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        # self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        # assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)

        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(params['hidden_size'] * 2, params['hidden_size'] * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(params['hidden_size'] * 2, params['hidden_size'] * 2),
                                  nn.Sigmoid()))
        self.dropout = nn.Dropout(p=params['dropout'])

    def forward(self, batch):
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.contiguous().view(-1, self.cd, x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, params['char_channel_size'])
            return x

        populate_tensors(self, batch)
        xat = char_emb_layer(self.at_c)
        xad = char_emb_layer(self.ad_c)
        xwhen = char_emb_layer(self.when_c)
        xwhere = char_emb_layer(self.where_c)
        xpts = char_emb_layer(self.pts_c)
        qq = char_emb_layer(self.q_c)
        qchoices = char_emb_layer(self.choices_c)

        #word embedding


        return xat

