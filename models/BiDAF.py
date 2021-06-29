import sys
from config import *
import torch
import torch.nn as nn
from utils import LSTM, Linear, populate_tensors
import torch.nn.functional as F
import torchtext

sys.path.append('../')

glove = torchtext.vocab.GloVe(name="6B", dim=100, max_vectors=50000)


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
        self.at_mask = torch.zeros((self.N, params['max_num_albums'], params['max_sent_album_title_size']),
                                   dtype=torch.bool)

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
        # feed in the pretrain word vectors for all batch
        # self.existing_emb_mat = torch.zeros((self.VW, params['word_emb_size']), dtype=torch.float64)
        self.glove_emb = nn.Embedding.from_pretrained(glove.vectors, freeze=True)

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

        # BiDirectional LSTM
        self.context_LSTM = nn.LSTM(input_size=params['hidden_size'] * 2,
                                    hidden_size=params['hidden_size'],
                                    num_layers=2,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=params['dropout'])

        self.image_encoder = nn.LSTM(input_size=params['image_feat_dim'],
                                     hidden_size=params['hidden_size'],
                                     num_layers=2,
                                     bidirectional=True,
                                     batch_first=True,
                                     dropout=params['dropout'])

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

        def highway_network(x1, x2):
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
                return x

        populate_tensors(self, batch)

        # Char Embeddings
        xat = char_emb_layer(self.at_c.reshape(self.N, -1, params['max_word_size']))
        xad = char_emb_layer(self.ad_c.reshape(self.N, -1, params['max_word_size']))
        xwhen = char_emb_layer(self.when_c.reshape(self.N, -1, params['max_word_size']))
        xwhere = char_emb_layer(self.where_c.reshape(self.N, -1, params['max_word_size']))
        xpts = char_emb_layer(self.pts_c.reshape(self.N, -1, params['max_word_size']))
        qq = char_emb_layer(self.q_c.reshape(self.N, -1, params['max_word_size']))
        qchoices = char_emb_layer(self.choices_c.reshape(self.N, -1, params['max_word_size']))

        # Word Embeddings
        w_at = self.glove_emb(self.at)
        w_ad = self.glove_emb(self.ad)
        w_when = self.glove_emb(self.when)
        w_where = self.glove_emb(self.where)
        w_pts = self.glove_emb(self.pts)
        w_q = self.glove_emb(self.q)
        w_choices = self.glove_emb(self.choices)

        # Reshape word embeddings
        w_at = w_at.reshape(self.N, -1, self.wd)
        w_ad = w_ad.reshape(self.N, -1, self.wd)
        w_when = w_when.reshape(self.N, -1, self.wd)
        w_where = w_where.reshape(self.N, -1, self.wd)
        w_pts = w_pts.reshape(self.N, -1, self.wd)
        w_q = w_q.reshape(self.N, -1, self.wd)
        w_choices = w_choices.reshape(self.N, -1, self.wd)

        # Highway Network
        h_at = highway_network(xat, w_at)  # Shape (Batch Size, 32, 200) (char_dim + word_dim = 200)
        h_ad = highway_network(xad, w_ad)  # Shape (Batch Size, 40, 200)
        h_when = highway_network(xwhen, w_when)  # Shape (Batch Size, 24, 200)
        h_where = highway_network(xwhere, w_where)  # Shape (Batch Size, 24, 200)
        h_pts = highway_network(xpts, w_pts)  # Shape (Batch Size, 256, 200)
        h_qq = highway_network(qq, w_q)  # Shape (Batch Size, 23, 200)
        h_choices = highway_network(qchoices, w_choices)  # Shape (Batch Size, 24, 200)

        # Image Features
        image_feat = nn.Embedding(self.image_emb_mat.shape[0], self.image_emb_mat.shape[1])
        image_feat.weight.data.copy_(torch.from_numpy(self.image_emb_mat))
        image2feat = image_feat(self.pis)

        # Bidirectional LSTMs
        qq_out, qq_out_states = self.context_LSTM(h_qq)  # Questions
        at_out, at_out_states = self.context_LSTM(h_at)  # Album Title
        ad_out, ad_out_states = self.context_LSTM(h_ad)  # Album Description
        when_out, when_out_states = self.context_LSTM(h_when)  # When
        where_out, where_out_states = self.context_LSTM(h_where)  # Where
        pts_out, pts_out_states = self.context_LSTM(h_pts)  # Photo Titles
        choice_out, choices_out_states = self.context_LSTM(h_choices)  # Choices

        images_out, images_out_states = self.image_encoder(image2feat.reshape(params['batch_size'], -1,
                                                                              params['image_feat_dim']))

        return images_out, qq_out_states[1]
