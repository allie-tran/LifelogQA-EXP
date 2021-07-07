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
                                    num_layers=1,
                                    bidirectional=True,
                                    batch_first=True)

        self.image_encoder = nn.LSTM(input_size=params['image_feat_dim'],
                                     hidden_size=params['hidden_size'],
                                     bidirectional=True,
                                     batch_first=True)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(params['hidden_size'] * 2, 1)
        self.att_weight_q = Linear(params['hidden_size'] * 2, 1)
        self.att_weight_cq = Linear(params['hidden_size'] * 2, 1)

        # 5. Modeling Layer
        self.out1 = Linear(params['hidden_size'] * 2, params['hidden_size'] * 2 * 7)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(params['hidden_size'] * 70, 1),
            torch.nn.LeakyReLU(0.1)
        )

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

        def attention_flow_layer(c, q):
            N = c.size(0)
            w = c.size(-1)
            c = c.reshape(N, -1, w)

            c_len = c.size(1)
            q_len = q.size(1)
            cq = []
            for i in range(q_len):
                qi = q.select(1, i).unsqueeze(1)  # (batch, 1, hidden_size * 2)
                ci = self.att_weight_cq(c * qi).squeeze()  # (batch, c_len, 1)
                cq.append(ci)
            cq = torch.stack(cq, dim=-1)  # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len,
                                                                                                          -1) + cq
            a = F.softmax(s, dim=2)
            c2q_att = torch.bmm(a, q)
            # print('c2q', c2q_att.size())
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # print('q2c1', q2c_att.size())
            # (batch, c_len, hidden_size * 2) (tiled)
            # q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # print('q2c2', q2c_att.size())
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)
            # (batch, c_len, hidden_size * 8)
            # x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return q2c_att, s

        def attention(hinfo, hq, hinfo_mask=None, hq_mask=None, wd=None):
            print('hinfo size', hinfo.size())
            print('hq size', hq.size())
            N = hinfo.size()[0]
            w = hinfo.size()[-1]

            # M = hinfo.size()[1]
            JQ = hq.size()[1]

            hinfo = hinfo.reshape(N, -1, w)
            if hinfo_mask is not None:
                hinfo_mask = hinfo_mask.reshape(N, -1)

            V = hinfo.size()[1]

            h_aug = torch.tile(torch.unsqueeze(hinfo, 2), [1, 1, JQ, 1])
            q_aug = torch.tile(torch.unsqueeze(hq, 1), [1, V, 1, 1])


            h_q_cat = torch.cat([(h_aug - q_aug) * (h_aug - q_aug), h_aug * q_aug], 3)
            print('h_q_cat', h_q_cat.size())

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

        # Reshape to original size
        h_at = h_at.reshape(self.N * params['max_num_albums'], params['max_sent_album_title_size'], -1)
        h_ad = h_ad.reshape(self.N * params['max_num_albums'], params['max_sent_des_size'], -1)
        h_when = h_when.reshape(self.N * params['max_num_albums'], params['max_when_size'], -1)
        h_where = h_where.reshape(self.N * params['max_num_albums'], params['max_where_size'], -1)
        h_pts = h_pts.reshape(self.N * params['max_num_albums'] * params['max_num_photos'],
                              params['max_sent_photo_title_size'], -1)
        h_qq = h_qq.reshape(self.N, params['max_question_size'], -1)
        h_choices = h_choices.reshape(self.N * 4, params['max_answer_size'], -1)

        # Image Features
        image_feat = nn.Embedding(self.image_emb_mat.shape[0], self.image_emb_mat.shape[1], device="cuda")
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

        # Resize LSTMs outputs and final hidden states

        at_out = at_out.reshape(self.N, params['max_num_albums'], -1, 200)
        at_forward_h = at_out_states[0][0, :, :].reshape(self.N, params['max_num_albums'], -1)
        at_backward_h = at_out_states[0][1, :, :].reshape(self.N, params['max_num_albums'], -1)
        at_final_h = torch.cat([at_forward_h, at_backward_h], dim=2)

        ad_out = ad_out.reshape(self.N, params['max_num_albums'], -1, 200)
        ad_forward_h = ad_out_states[0][0, :, :].reshape(self.N, params['max_num_albums'], -1)
        ad_backward_h = ad_out_states[0][1, :, :].reshape(self.N, params['max_num_albums'], -1)
        ad_final_h = torch.cat([ad_forward_h, ad_backward_h], dim=2)

        when_out = when_out.reshape(self.N, params['max_num_albums'], -1, 200)
        when_forward_h = when_out_states[0][0, :, :].reshape(self.N, params['max_num_albums'], -1)
        when_backward_h = when_out_states[0][1, :, :].reshape(self.N, params['max_num_albums'], -1)
        when_final_h = torch.cat([when_forward_h, when_backward_h], dim=2)

        where_out = where_out.reshape(self.N, params['max_num_albums'], -1, 200)
        where_forward_h = where_out_states[0][0, :, :].reshape(self.N, params['max_num_albums'], -1)
        where_backward_h = where_out_states[0][1, :, :].reshape(self.N, params['max_num_albums'], -1)
        where_final_h = torch.cat([where_forward_h, where_backward_h], dim=2)

        pts_out = pts_out.reshape(self.N, params['max_num_albums'], params['max_num_photos'], -1, 200)
        pts_forward_h = pts_out_states[0][0, :, :].reshape(self.N, params['max_num_albums'],
                                                           params['max_num_photos'], -1)
        pts_backward_h = pts_out_states[0][1, :, :].reshape(self.N, params['max_num_albums'],
                                                            params['max_num_photos'], -1)
        pts_final_h = torch.cat([pts_forward_h, pts_backward_h], dim=3)

        choice_out = choice_out.reshape(self.N, 4, -1, 200)
        choice_forward_h = choices_out_states[0][0, :, :].reshape(self.N, 4, -1)
        choice_backward_h = choices_out_states[0][1, :, :].reshape(self.N, 4, -1)
        choice_final_h = torch.cat([choice_forward_h, choice_backward_h], dim=2)

        qq_forward_h = qq_out_states[0][0, :, :].reshape(self.N, -1)
        qq_backward_h = qq_out_states[0][1, :, :].reshape(self.N, -1)
        qq_final_h = torch.cat([qq_forward_h, qq_backward_h], dim=1)

        # Attention
        at_attn, at_s = attention_flow_layer(at_out, qq_out)
        ad_attn, ad_s = attention_flow_layer(ad_out, qq_out)
        when_attn, when_s = attention_flow_layer(when_out, qq_out)
        where_attn, where_s = attention_flow_layer(where_out, qq_out)
        pts_attn, pts_s = attention_flow_layer(pts_out, qq_out)
        image_attn, image_s = attention_flow_layer(images_out, qq_out)

        attn_concat = torch.cat([at_attn, ad_attn, when_attn, where_attn, pts_attn, image_attn], dim=1)

        # Direct Links
        at_re = at_out.reshape(self.N, -1, 200)
        ad_re = ad_out.reshape(self.N, -1, 200)
        when_re = when_out.reshape(self.N, -1, 200)
        where_re = where_out.reshape(self.N, -1, 200)
        pts_re = pts_out.reshape(self.N, -1, 200)
        image_re = images_out.reshape(self.N, -1, 200)
        full = torch.cat([at_re, ad_re, when_re, where_re, pts_re, image_re], dim=1)
        full_attn, full_s = attention_flow_layer(full, qq_out)

        final_out = torch.cat([full_attn, attn_concat], dim=1)

        # Choice Attention
        choice_attn = choice_final_h

        # Question Attention
        q_attn, q_attn_s = attention_flow_layer(qq_out, full)  # context aware query representation

        # Modelling Layer
        # concat choice and question attention
        choice_attn = self.out1(choice_attn)
        q_attn = self.out1(q_attn)

        final_out = torch.tile(torch.unsqueeze(final_out, 1), [1, self.num_choice, 1])
        q_attn = torch.tile(torch.unsqueeze(q_attn, 1), [1, self.num_choice, 1])

        xf = torch.cat([q_attn, choice_attn, (final_out * choice_attn), (q_attn * choice_attn),
                       (final_out - choice_attn) * (final_out - choice_attn)],
                      dim=2)
        logits = self.output_layer(xf)
        logits = torch.squeeze(logits)
        pred = F.softmax(logits, dim=1)
        #actual = torch.tensor(self.y, dtype=float)
        actual = self.y.clone().detach()
        actual = torch.max(actual, 1)[1]
        index = torch.tensor(batch.data['yidx'], device="cuda")
        #print(logits)
        #print('Actual', actual)
        #print('Predicted', pred)
        #print('Index', batch.data['yidx'])
        #print('Index Tensor', index)
        return pred, index
