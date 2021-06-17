import sys
sys.path.append('../')
from config import *
import torch.nn as nn
from utils.nn import LSTM, Linear


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
    def __init__(self, config, pretrained):
        super(BiDAF, self).__init__()
        self.args = params

        self.N = params.batch_size

        self.VW = params.word_vocab_size
        self.VC = params.char_vocab_size
        self.W = params.max_word_size

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(params.char_vocab_size, params.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Sequential(
            nn.Conv2d(1, params.char_channel_size, (params.char_dim, params.char_channel_width)),
            nn.ReLU()
        )

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)

        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(params.hidden_size * 2, 1)
        self.att_weight_q = Linear(params.hidden_size * 2, 1)
        self.att_weight_cq = Linear(params.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=params.hidden_size * 8,
                                   hidden_size=params.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=params.dropout)

        self.modeling_LSTM2 = LSTM(input_size=params.hidden_size * 2,
                                   hidden_size=params.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=params.dropout)

        # 6. Output Layer
        self.p1_weight_g = Linear(params.hidden_size * 8, 1, dropout=args.dropout)
        self.p1_weight_m = Linear(params.hidden_size * 2, 1, dropout=args.dropout)
        self.p2_weight_g = Linear(params.hidden_size * 8, 1, dropout=args.dropout)
        self.p2_weight_m = Linear(params.hidden_size * 2, 1, dropout=args.dropout)

        self.output_LSTM = LSTM(input_size=params.hidden_size * 2,
                                hidden_size=params.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=params.dropout)

        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, batch):
        print('Forward')

        def char_emb_layer(x):
            batch_size = x.size(0)
