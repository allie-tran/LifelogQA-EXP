##https://github.com/galsang/BiDAF-pytorch/blob/master/utils/nn.py
import os
import torch
import torch.nn as nn
from itertools import zip_longest
import random, itertools
from collections import defaultdict
from config import *
import math
import numpy as np
from copy import deepcopy


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def grouper(l, n):
    # given a list and n(batch_size), devide list into n sized chunks
    # last one will fill None
    args = [iter(l)] * n
    out = zip_longest(*args, fillvalue=None)
    out = list(out)
    return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x):
        x, x_len = x
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)
        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)

        return x, h


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class Dataset():
    def __init__(self, data, datatype, shared=None, valid_idxs=None):
        self.data = data
        self.datatype = datatype
        self.shared = shared

        self.valid_idxs = range(self.get_data_size()) if valid_idxs is None else valid_idxs
        self.num_examples = len(self.valid_idxs)

    def get_data_size(self):
        return len(next(iter(self.data.values())))

    def get_by_idxs(self, idxs):
        out = defaultdict(list)  # so the initial value is a list
        for key, val in self.data.items():  # "q",[] ; "cq", [] ;"y",[]
            out[key].extend(val[idx] for idx in idxs)  # extend with one whole list
        # so we get a batch_size of data : {"q":[] -> len() == batch_size}
        return out

    def get_batches(self, batch_size, num_steps, shuffle=True, cap=False):

        num_batches_per_epoch = int(math.ceil(self.num_examples / float(batch_size)))
        if cap and (num_steps > num_batches_per_epoch):
            num_steps = num_batches_per_epoch
        # this may be zero
        num_epochs = int(math.ceil(num_steps / float(num_batches_per_epoch)))
        # TODO: Group single-album and cross-album question to train separately?
        # shuflle
        if shuffle:
            # shuffled idx
            # but all epoch has the same order
            random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
            random_grouped = lambda: list(grouper(random_idxs, batch_size))  # all batch idxs for one epoch
            # grouper
            # given a list and n(batch_size), devide list into n sized chunks
            # last one will fill None
            grouped = random_grouped
        else:
            raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
            grouped = raw_grouped
        # grouped is a list of list, each is batch_size items make up to -> total_sample

        # all batches idxs from multiple epochs
        batch_idxs_iter = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        # print "in get batches, num_steps:%s,num_epch:%s"%(num_steps,num_epochs)
        for _ in range(num_steps):  # num_step should be batch_idxs length
            # so in the end batch, the None will not included
            batch_idxs = tuple(i for i in next(batch_idxs_iter) if i is not None)  # each batch idxs
            # so batch_idxs might not be size batch_size

            # print "batch size:%s"%len(batch_idxs)
            # a dict of {"q":[],"cq":[],"y":[]...}
            # get from dataset class:{"q":...} all the key items with idxs
            # so no album info anything
            batch_data = self.get_by_idxs(batch_idxs)  # get the actual data based on idx
            # print len(batch_data['q'])

            # go through all album to get pid2idx first,
            pid2idx = {}  # get all the pid to a index
            for albumIds in batch_data['aid']:  # each QA has album list
                for albumId in albumIds:
                    for pid in self.shared['albums'][albumId]['photo_ids']:
                        if pid not in pid2idx.keys():
                            pid2idx[pid] = len(pid2idx.keys())  # start from zero

            # fill in the image feature
            image_feats = np.zeros(
                (len(pid2idx), self.shared['pid2feat'][list(self.shared['pid2feat'].keys())[0]].shape[0]),
                dtype="float32")

            # here image_matrix idx-> feat, will replace the pid in each instance to this idx
            for pid in pid2idx:  # fill each idx with feature, -> pid
                image_feats[pid2idx[pid]] = self.shared['pid2feat'][pid]

            batch_data['pidx2feat'] = image_feats

            shared_batch_data = defaultdict(list)

            # all the shared data need for this mini batch
            for albumIds in batch_data['aid']:
                # one shared album info for one qa, could be multiple albums
                album_title = []
                album_title_c = []
                album_description = []
                album_description_c = []
                album_where = []
                album_where_c = []
                album_when = []
                album_when_c = []
                photo_titles = []
                photo_titles_c = []
                photo_idxs = []
                photo_ids = []  # for debug
                for albumId in albumIds:
                    album = self.shared['albums'][albumId]

                    album_title.append(album['title'])
                    album_title_c.append(album['title_c'])
                    album_description.append(album['description'])
                    album_description_c.append(album['description_c'])
                    album_where.append(album['where'])
                    album_when.append(album['when'])
                    album_where_c.append(album['where_c'])
                    album_when_c.append(album['when_c'])
                    photo_titles.append(album['photo_titles'])
                    photo_titles_c.append(album['photo_titles_c'])
                    photo_idxs.append([pid2idx[pid] for pid in album['photo_ids']])
                    # this will not be used, just for debug
                    photo_ids.append(album['photo_ids'])

                shared_batch_data['album_title'].append(album_title)
                shared_batch_data['album_title_c'].append(album_title_c)
                shared_batch_data['album_description'].append(album_description)
                shared_batch_data['album_description_c'].append(album_description_c)
                shared_batch_data['where'].append(album_where)
                shared_batch_data['where_c'].append(album_where_c)
                shared_batch_data['when'].append(album_when)
                shared_batch_data['when_c'].append(album_when_c)
                shared_batch_data['photo_titles'].append(photo_titles)
                shared_batch_data['photo_titles_c'].append(photo_titles_c)
                # all pid should be change to a local batch idx
                shared_batch_data['photo_idxs'].append(photo_idxs)
                # for debug
                shared_batch_data['photo_ids'].append(photo_ids)

            batch_data.update(shared_batch_data)  # combine the shared data in to the minibatch
            # so it be {"q","cq","y"...,"pidx2feat","album_info"...}

            yield batch_idxs, Dataset(batch_data, self.datatype, shared=self.shared)


def update_params(datasets, showMeta=False):
    params['max_num_albums'] = 0
    params['max_num_photos'] = 0  # max photo per album
    params['max_sent_album_title_size'] = 0  # max sentence word count for album title
    params['max_sent_photo_title_size'] = 0
    params['max_sent_des_size'] = 0
    params['max_when_size'] = 6
    params['max_where_size'] = 6
    params['max_answer_size'] = 6
    params['max_question_size'] = 0
    params['max_word_size'] = 0  # word letter count

    # go through all datasets to get the max count
    for dataset in datasets:
        for idx in dataset.valid_idxs:

            question = dataset.data['q'][idx]
            answer = dataset.data['y'][idx]
            choices = dataset.data['cs'][idx]

            params['max_question_size'] = max(params['max_question_size'], len(question))
            params['max_word_size'] = max(params['max_word_size'], max(len(word) for word in question))

            for sent in choices + [answer]:
                params['max_answer_size'] = max(params['max_answer_size'], len(sent))
                params['max_word_size'] = max(params['max_word_size'], max(len(word) for word in sent))

            albums = [dataset.shared['albums'][aid] for aid in dataset.data['aid'][idx]]

            params['max_num_albums'] = max(params['max_num_albums'], len(albums))

            for album in albums:
                params['max_num_photos'] = max(params['max_num_photos'], len(album['photo_ids']))

                # title
                # params.max_sent_title_size = max(params.max_sent_title_size,len(album['title']))
                params['max_sent_album_title_size'] = max(params['max_sent_album_title_size'], len(album['title']))

                for title in album['photo_titles']:
                    if len(title) > 0:
                        # params.max_sent_title_size = max(params.max_sent_title_size,len(title))
                        params['max_sent_photo_title_size'] = max(params['max_sent_photo_title_size'], len(title))
                        params['max_word_size'] = max(params['max_word_size'], max(len(word) for word in title))

                # description
                if len(album['description']) > 0:
                    params['max_sent_des_size'] = max(params['max_sent_des_size'], len(album['description']))
                    params['max_word_size'] = max(params['max_word_size'],
                                                  max(len(word) for word in album['description']))

                # when
                params['max_when_size'] = max(params['max_when_size'], len(album['when']))

                # got word size for all
                params['max_word_size'] = max(params['max_word_size'], max(len(word) for word in album['title']))
                params['max_word_size'] = max(params['max_word_size'], max(len(word) for word in album['when']))
                # where could be empty
                if len(album['where']) != 0:
                    params['max_word_size'] = max(params['max_word_size'], max(len(word) for word in album['where']))
                    params['max_where_size'] = max(params['max_where_size'], len(album['where']))

    if showMeta:
        params_vars = vars(params)
        print("max meta:")
        print("\t" + " ,".join(["%s:%s" % (key, params_vars[key]) for key in params.maxmeta]))

    # adjust the max based on the threshold argument input as well
    if params['is_train']:
        # album and photo counts
        params['max_num_albums'] = min(params['max_num_albums'], params['num_albums_thres'])
        params['max_num_photos'] = min(params['max_num_photos'], params['num_photos_thres'])

        # params.max_sent_title_size = min(params.max_sent_title_size,params.sent_title_size_thres)
        params['max_sent_album_title_size'] = min(params['max_sent_album_title_size'],
                                                  params['sent_album_title_size_thres'])
        params['max_sent_photo_title_size'] = min(params['max_sent_photo_title_size'],
                                                  params['sent_photo_title_size_thres'])

        params['max_sent_des_size'] = min(params['max_sent_des_size'], params['sent_des_size_thres'])

        params['max_when_size'] = min(params['max_when_size'], params['sent_when_size_thres'])
        params['max_where_size'] = min(params['max_where_size'], params['sent_where_size_thres'])

        params['max_answer_size'] = min(params['max_answer_size'], params['answer_size_thres'])

    # not cliping question
    # params.question_size_thres = max(params.max_question_size,params.question_size_thres)
    else:
        # for testing, still removing the description since it could be 2k+ tokens
        params['max_sent_des_size'] = min(params['max_sent_des_size'], params['sent_des_size_thres'])
        # also cap the photo title size
        params['max_sent_photo_title_size'] = min(params['max_sent_photo_title_size'],
                                                  params['sent_photo_title_size_thres'])

    # always clip word_size
    params['max_word_size'] = min(params['max_word_size'], params['word_size_thres'])

    # get the vocab size # the charater in the charCounter
    params['char_vocab_size'] = len(datasets[0].shared['char2idx'])
    # the word embeding's dimension
    params['word_emb_size'] = len(next(iter(datasets[0].shared['word2vec'].values())))
    # the size of word vocab not in existing glove
    params['word_vocab_size'] = len(datasets[0].shared['word2idx'])


def get_char(char, batch):
    d = batch.shared['char2idx']
    if char in d:
        return d[char]
    return 1


def get_word(word, batch):
    d = batch.shared['word2idx']  # this is for the word not in glove
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in d:
            return d[each]
    # the word in glove

    d2 = batch.shared['existing_word2idx']
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in d2:
            return d2[each] + len(d)  # all idx + len(the word to train)
    return 1  # 1 is the -UNK-


def populate_tensors(self, batch):
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

    # question and choices
    Q = batch.data['q']
    Q_c = batch.data['cq']

    C = deepcopy(batch.data['cs'])
    C_c = deepcopy(batch.data['ccs'])

    AT = batch.data['album_title']
    AT_c = batch.data['album_title_c']
    AD = batch.data['album_description']
    AD_c = batch.data['album_description_c']
    WHERE = batch.data['where']
    WHERE_c = batch.data['where_c']
    WHEN = batch.data['when']
    WHEN_c = batch.data['when_c']
    PT = batch.data['photo_titles']
    PT_c = batch.data['photo_titles_c']
    PI = batch.data['photo_idxs']

    if params['is_train']:
        Y = batch.data['y']
        Y_c = batch.data['cy']

        correctIndex = np.random.choice(self.num_choice, params['batch_size'])
        for i in range(len(batch.data['y'])):
            self.y[i, correctIndex[i]] = True
            assert len(C[i]) == (self.num_choice - 1)
            C[i].insert(correctIndex[i], Y[i])
            C_c[i].insert(correctIndex[i], Y_c[i])
            assert len(batch.data['cs'][i]) == (self.num_choice - 1)
    else:
        pass

    # the photo idx is simple
    for i, pi in enumerate(PI):
        # one batch
        for j, pij in enumerate(pi):
            # one album
            if j == params['max_num_albums']:
                break
            for k, pijk in enumerate(pij):
                if k == params['max_num_photos']:
                    break
                # print pijk
                assert isinstance(pijk, int)
                self.pis[i, j, k] = pijk
                self.pis_mask[i, j, k] = True

    # album title
    for i, ati in enumerate(AT):  # batch_sizes
        # one batch
        for j, atij in enumerate(ati):
            # one album
            if j == params['max_num_albums']:
                break
            for k, atijk in enumerate(atij):
                # each word
                if k == params['max_sent_album_title_size']:
                    break
                wordIdx = get_word(atijk, batch)
                self.at[i, j, k] = wordIdx
                self.at_mask[i, j, k] = True

    for i, cati in enumerate(AT_c):
        for j, catij in enumerate(cati):
            if j == params['max_num_albums']:
                break
            for k, catijk in enumerate(catij):
                if k == params['max_sent_album_title_size']:
                    break
                for l, catijkl in enumerate(catijk):
                    if l == params['max_word_size']:
                        break
                    self.at_c[i, j, k, l] = get_char(catijkl, batch)

    # album description
    for i, adi in enumerate(AD):  # batch_sizes
        # one batch
        for j, adij in enumerate(adi):
            # one album
            if j == params['max_num_albums']:
                break
            for k, adijk in enumerate(adij):
                # each word
                if k == params['max_sent_des_size']:
                    break
                wordIdx = get_word(adijk, batch)
                self.ad[i, j, k] = wordIdx
                self.ad_mask[i, j, k] = True

    for i, cadi in enumerate(AD_c):
        # one batch
        for j, cadij in enumerate(cadi):
            if j == params['max_num_albums']:
                break
            for k, cadijk in enumerate(cadij):
                # each word
                if k == params['max_sent_des_size']:
                    break
                for l, cadijkl in enumerate(cadijk):
                    if l == params['max_word_size']:
                        break
                    self.ad_c[i, j, k, l] = get_char(cadijkl, batch)

    # album when
    for i, wi in enumerate(WHEN):  # batch_sizes
        # one batch
        for j, wij in enumerate(wi):
            # one album
            if j == params['max_num_albums']:
                break
            for k, wijk in enumerate(wij):
                # each word
                if k == params['max_when_size']:
                    break
                wordIdx = get_word(wijk, batch)
                self.when[i, j, k] = wordIdx
                self.when_mask[i, j, k] = True

    for i, cwi in enumerate(WHEN_c):
        # one batch
        for j, cwij in enumerate(cwi):
            if j == params['max_num_albums']:
                break
            for k, cwijk in enumerate(cwij):
                # each word
                if k == params['max_when_size']:
                    break
                for l, cwijkl in enumerate(cwijk):
                    if l == params['max_word_size']:
                        break
                    self.when_c[i, j, k, l] = get_char(cwijkl, batch)

    # album where
    for i, wi in enumerate(WHERE):  # batch_sizes
        # one batch
        for j, wij in enumerate(wi):
            # one album
            if j == params['max_num_albums']:
                break
            for k, wijk in enumerate(wij):
                # each word
                if k == params['max_where_size']:
                    break
                wordIdx = get_word(wijk, batch)
                self.where[i, j, k] = wordIdx
                self.where_mask[i, j, k] = True

    for i, cwi in enumerate(WHERE_c):
        # one batch
        for j, cwij in enumerate(cwi):
            if j == params['max_num_albums']:
                break
            for k, cwijk in enumerate(cwij):
                # each word
                if k == params['max_where_size']:
                    break
                for l, cwijkl in enumerate(cwijk):
                    if l == params['max_word_size']:
                        break
                    self.where_c[i, j, k, l] = get_char(cwijkl, batch)

    # photo title
    for i, pti in enumerate(PT):  # batch_sizes
        # one batch
        for j, ptij in enumerate(pti):
            # one album
            if j == params['max_num_albums']:
                break
            for k, ptijk in enumerate(ptij):
                # each photo
                if k == params['max_num_photos']:
                    break
                for l, ptijkl in enumerate(ptijk):
                    if l == params['max_sent_photo_title_size']:
                        break
                    # each word
                    wordIdx = get_word(ptijkl, batch)
                    self.pts[i, j, k, l] = wordIdx
                    self.pts_mask[i, j, k, l] = True

    for i, pti in enumerate(PT_c):  # batch_sizes
        # one batch
        for j, ptij in enumerate(pti):
            # one album
            if j == params['max_num_albums']:
                break
            for k, ptijk in enumerate(ptij):
                # each photo
                if k == params['max_num_photos']:
                    break
                for l, ptijkl in enumerate(ptijk):
                    if l == params['max_sent_photo_title_size']:
                        break
                    # each word
                    for o, ptijklo in enumerate(ptijkl):
                        # each char
                        if o == params['max_word_size']:
                            break
                        self.pts_c[i, j, k, l, o] = get_char(ptijklo, batch)

    # Answer Choices
    for i, ci in enumerate(C):
        # one batch
        assert len(ci) == self.num_choice
        for j, cij in enumerate(ci):
            # one answer
            for k, cijk in enumerate(cij):
                # one word
                if k == params['max_answer_size']:
                    break
                wordIdx = get_word(cijk, batch)
                self.choices[i, j, k] = wordIdx
                self.choices_mask[i, j, k] = True

    for i, ci in enumerate(C_c):
        # one batch
        assert len(ci) == self.num_choice, (len(ci))
        for j, cij in enumerate(ci):
            # one answer
            for k, cijk in enumerate(cij):
                # one word
                if k == params['max_answer_size']:
                    break
                for l, cijkl in enumerate(cijk):
                    if l == params['max_word_size']:
                        break
                    self.choices_c[i, j, k, l] = get_char(cijkl, batch)

    # load the question
    # no limiting on the question word length
    for i, qi in enumerate(Q):
        # one batch
        for j, qij in enumerate(qi):
            self.q[i, j] = get_word(qij, batch)
            self.q_mask[i, j] = True

    # Load the Question Char
    for i, cqi in enumerate(Q_c):
        for j, cqij in enumerate(cqi):
            for k, cqijk in enumerate(cqij):
                if k == params['max_word_size']:
                    break
                self.q_c[i, j, k] = get_char(cqijk, batch)

    self.image_emb_mat = batch.data['pidx2feat']
    # self.existing_emb_mat = torch.tensor(batch.shared['existing_emb_mat'], device='cuda')
    self.existing_emb_mat = torch.from_numpy(batch.shared['existing_emb_mat'])


def get_eval_score(pred, gt):
    assert len(pred) == len(gt)
    assert len(pred) > 0
    total = len(pred)
    correct = 0
    for qid in pred.keys():
        if pred[qid] == gt[qid]:
            correct += 1
    return correct / float(total)


def getAnswers(yp, batch):
    id2predanswers = {}
    id2realanswers = {}
    # print yp.shape
    for qid, yidxi, ypi in zip(batch[1].data['qid'], batch[1].data['yidx'], yp):
        #print('qid', qid)
        #print('yidxi', yidxi)
        #print('ypi', ypi)
        id2predanswers[qid] = np.argmax(ypi.detach().numpy())
        id2realanswers[qid] = yidxi  # available answers
        assert yidxi < 4
        assert np.argmax(ypi.detach().numpy()) < 4
    # print q,id2answers[qid
    return id2predanswers, id2realanswers
