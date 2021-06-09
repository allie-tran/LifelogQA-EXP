import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import config
import numpy as np
from collections import defaultdict
from pprint import pprint

MEMEX_DATA = os.getenv('MEMEX_DATA')

def read_data(mode, loadExistModelShared):
    """mode = test, val, train"""

    with open(f"{MEMEX_DATA}/preprocessed/{mode}_data.p", "rb") as f:
        data = pickle.load(f)
    with open(f"{MEMEX_DATA}/preprocessed/{mode}_shared.p", "rb") as f:
        shared = pickle.load(f)  # this will be added later with word id, either new or load from exists

    num_examples = len(data['q'])

    # this is the file for the model' training, with word ID and stuff, if set load in config, will read from existing, otherwise write a new one
    # load the word2idx info into shared[]
    model_shared_path = f"{MEMEX_DATA}/preprocessed/shared.p"
    if(loadExistModelShared):
        with open(model_shared_path, "rb") as f:
            model_shared = pickle.load(f)
        for key in model_shared:
            shared[key] = model_shared[key]
    else:
        # no fine tuning of word vector

        # the word larger than word_count_thres and not in the glove word2vec
        # word2idx -> the idx is the wordCounter's item() idx
        # the new word to index
        shared['word2idx'] = {word: idx+2 for idx, word in enumerate([word for word, count in shared['wordCounter'].items(
        ) if (count > config.word_count_thres) and not word in shared['word2vec']])}
        shared['char2idx'] = {char: idx+2 for idx, char in enumerate(
            [char for char, count in shared['charCounter'].items() if count > config.char_count_thres])}

        NULL = "<NULL>"
        UNK = "<UNK>"
        shared['word2idx'][NULL] = 0
        shared['char2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][UNK] = 1

        # existing word in word2vec will be put after len(new word)+2
        pickle.dump({"word2idx": shared['word2idx'], 'char2idx': shared['char2idx']}, open(
            model_shared_path, "wb"))

    # load the word embedding for word in word2vec
    shared['existing_word2idx'] = {word: idx for idx, word in enumerate(
        [word for word in sorted(shared['word2vec'].keys()) if not word in shared['word2idx']])}

    # idx -> vector
    idx2vec = {idx: shared['word2vec'][word]
               for word, idx in shared['existing_word2idx'].items()}
    # load all this vector into a matrix
    # so you can use word -> idx -> vector
    # using range(len) so that the idx is 0,1,2,3...
    # then it could be call with embedding lookup with the correct idx

    shared['existing_emb_mat'] = np.array(
        [idx2vec[idx] for idx in range(len(idx2vec))], dtype="float32")

    return MemexQADataset(data, shared=shared)


class MemexQADataset(Dataset):
    def __init__(self, data, shared=None):
        self.data = data
        self.shared = shared

    def __len__(self):
        return len(self.data['q'])

    def __getitem__(self, idx):
        out = {}
        for key, val in self.data.items():
            out[key] = val[idx]
        return out


def collate_fn(shared, batch):
    # print "batch size:%s"%len(batch_idxs)
    # a dict of {"q":[],"cq":[],"y":[]...}
    # get from dataset class:{"q":...} all the key items with idxs
    # so no album info anything
    # get the actual data based on idx
    # print len(batch_data['q'])
    batch_data = defaultdict(lambda: [])
    for data in batch:
        for key, val in data.items():
            batch_data[key].append(val)

    # go through all album to get pid2idx first,
    pid2idx = {}  # get all the pid to a index
    for albumIds in batch_data['aid']:  # each QA has album list
        for albumId in albumIds:
            for pid in shared['albums'][albumId]['photo_ids']:
                if not pid in pid2idx: 
                    pid2idx[pid] = len(pid2idx.keys())  # start from zero

    # fill in the image feature
    image_feats = np.zeros((len(pid2idx), shared['pid2feat'][list(shared['pid2feat'].keys())[0]].shape[0]), dtype="float32")

    # here image_matrix idx-> feat, will replace the pid in each instance to this idx
    for pid in pid2idx:  # fill each idx with feature, -> pid
        image_feats[pid2idx[pid]] = shared['pid2feat'][pid]

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
            album = shared['albums'][albumId]

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

    # combine the shared data in to the minibatch
    batch_data.update(shared_batch_data)
    # so it be {"q","cq","y"...,"pidx2feat","album_info"...}

    return batch_data

dataset = read_data('val', False)
dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, collate_fn=lambda batch: collate_fn(dataset.shared, batch))
