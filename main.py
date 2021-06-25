import sys, os, argparse
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # so here won't have poll allocator info
import _pickle as pickle
import numpy as np
import math, time, json
import torch
from tqdm import tqdm
from torch import optim, nn
from config import *
from utils import Dataset
from models.BiDAF import BiDAF,EMA



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


get_model = None  # the model we will use, based on parameter in the get_args()


# def get_args():
#     global get_model
#     parser = argparse.ArgumentParser()
#     parser.add_argument("out_path", type=str)
#
#     parser.add_argument("outbasepath", type=str, help="full path will be outbasepath/modelname/runId")
#     parser.add_argument("--modelname", type=str, default="memoryqa")
#     parser.add_argument("--runId", type=int, default=0, help="used for run the same model multiple times")
#     parser.add_argument("--is_train", action="store_true", default=False, help="training mode, ")
#     parser.add_argument("--is_test", action="store_true", default=False, help="testing mode, otherwise test mode")
#
#
#     parser.add_argument('--batch_size', type=int, default=20)
#     parser.add_argument('--val_num_batches', type=int, default=100,help="eval during training, get how many batch in train/val to eval")
#     parser.add_argument("--num_epochs", type=int, default=20)  # num_step will be num_example/batch_size * epoch
#     return parser.parse_args()


def read_data(datatype, loadExistModelShared=False):
    data_path = os.path.join(config.paths['out_path'], "%s_data.p" % datatype)
    shared_path = os.path.join(config.paths['out_path'], "%s_shared.p" % datatype)

    with open(data_path, "rb")as f:
        data = pickle.load(f, encoding='latin1')
    with open(shared_path, "rb") as f:
        shared = pickle.load(f,
                             encoding='latin1')  # this will be added later with word id, either new or load from exists

    num_examples = len(data['q'])
    valid_idxs = range(num_examples)
    print("loaded %s/%s data points for %s" % (len(valid_idxs), num_examples, datatype))

    model_shared_path = os.path.join(config.paths['shared_path'], "shared.p")
    if (loadExistModelShared):
        with open(model_shared_path, "rb") as f:
            model_shared = pickle.load(f)
        for key in model_shared:
            shared[key] = model_shared[key]
    else:
        shared['word2idx'] = {word: idx + 2 for idx, word in enumerate(
            [word for word, count in shared['wordCounter'].items() if
             (count > config.threshold['word_count_thres']) and word not in shared['word2vec']])}
        shared['char2idx'] = {char: idx + 2 for idx, char in enumerate(
            [char for char, count in shared['charCounter'].items() if count > config.threshold['char_count_thres']])}

        NULL = "<NULL>"
        UNK = "<UNK>"
        shared['word2idx'][NULL] = 0
        shared['char2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][UNK] = 1

        pickle.dump({"word2idx": shared['word2idx'], 'char2idx': shared['char2idx']}, open(model_shared_path, "wb"))

    shared['existing_word2idx'] = {word: idx for idx, word in enumerate(
        [word for word in sorted(shared['word2vec'].keys()) if word not in shared['word2idx']])}
    idx2vec = {idx: shared['word2vec'][word] for word, idx in shared['existing_word2idx'].items()}
    shared['existing_emb_mat'] = np.array([idx2vec[idx] for idx in range(len(idx2vec))], dtype="float32")

    # assert config.feature_dim['image_feat_dim'] == shared['pid2feat'][shared['pid2feat'].keys()[0]].shape[0], ("image dim is not %s, it is %s" % (config.feature_dim['image_feat_dim'], shared['pid2feat'][shared['pid2feat'].keys()[0]].shape[0]))
    return Dataset(data, datatype, shared=shared, valid_idxs=valid_idxs)


def train():
    train_data = read_data('train', load)
    val_data = read_data('val', True)

    print('data loading complete')
    print(train_data)

    word2vec_dict = train_data.shared['word2vec']
    word2idx_dict = train_data.shared['word2idx']

    print(train_data.num_examples)
    print(val_data.num_examples)

    # we are not fine tuning , so this should be empty
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}

    params.word_emb_size = len(next(iter(train_data.shared['word2vec'].values())))
    # the size of word vocab not in existing glove
    params.word_vocab_size = len(train_data.shared['word2idx'])

    # random initial embedding matrix for new words
    params.emb_mat = np.array([idx2vec_dict[idx] if idx2vec_dict.has_key(idx) else np.random.multivariate_normal(
        np.zeros(params.word_emb_size), np.eye(params.word_emb_size)) for idx in range(params.word_vocab_size)],
                              dtype="float32")

    device = torch.device(f"cuda:{params.gpu}" if torch.cuda.is_available() else "cpu")
    model = BiDAF(params).to(device)

    ema = EMA(params.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adadelta(parameters, lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss, last_epoch = 0, -1
    max_dev_exact, max_dev_f1 = -1, -1


if __name__ == '__main__':
    # config = get_args()
    read_data(datatype='train', loadExistModelShared=False)
