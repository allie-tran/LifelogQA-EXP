
import os,sys,json,re
import argparse,nltk
import numpy as np
from collections import Counter
import _pickle as pickle
import random
from html.parser import HTMLParser
import ssl


def get_args():
    parser = argparse.ArgumentParser(description='MemexQA Data Preprocessing')
    parser.add_argument("datajson", type=str, help="path to the qas.json")
    parser.add_argument("albumjson", type=str, help="path to album_info.json")
    parser.add_argument("testids", type=str, help="path to test id list")
    parser.add_argument("--valids", type=str, default=None,
                        help="path to validation id list, if not set will be random 20% of the training set")
    parser.add_argument("imgfeat", action="store", type=str, help="path to img feat npz file")
    parser.add_argument("glove", action="store", type=str, help="path to glove vector file")
    parser.add_argument("outpath", type=str, help="output path")
    return parser.parse_args()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# for each token with "-" or others, remove it and split the token
def process_tokens(tokens):
    newtokens = []
    l = ("-","/", "~", '"', "'", ":","\)","\(","\[","\]","\{","\}")
    for token in tokens:
        # split then add multiple to new tokens
        newtokens.extend([one for one in re.split("[%s]"%("").join(l),token) if one != ""])
        return newtokens

def l2norm(feat):
    l2_norm = np.linalg.norm(feat,2)
    return feat/l2_norm

# word_counter words are lowered already
def get_word2vec(args,word_counter):
    word2vec_dict = {}
    import io
    with io.open(args.glove, 'r', encoding='utf-8') as fh:
        for line in fh:
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            #elif word.capitalize() in word_counter:
            #	word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            #elif word.upper() in word_counter:
            #	word2vec_dict[word.upper()] = vector

    print ("{}/{} of word vocab have corresponding vectors ".format(len(word2vec_dict), len(word_counter)))
    return word2vec_dict


from tqdm import tqdm
def prepro_each(args,data_type,question_ids,start_ratio=0.0,end_ratio=1.0):
    debug = False
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    sent_tokenize = nltk.sent_tokenize
    sent_tokenize = lambda para:[para] # right now we don't do sentence tokenization # just for album_description
    def word_tokenize(tokens):
        return process_tokens([token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)])

    qas = {str(qa['question_id']): qa for qa in args.qas}
    global_aids = {}  # all the album Id the question used, also how many question used that album

    ques_tokens, ques_chars, answer_tokens, answer_chars, album_id, ques_id,mul_choice, mul_choice_chars,idxs ,answer_index = [],[],[],[],[],[],[],[],[],[]
    word_counter,char_counter = Counter(),Counter() # lower word counter
    start_idx = int(round(len(question_ids) * start_ratio))
    end_idx = int(round(len(question_ids) * end_ratio))

    for idx,question_id in enumerate(tqdm(question_ids[start_idx:end_idx])):
        assert isinstance(question_id,str)
        qa = qas[question_id]

        # Questions
        ques_token_i = word_tokenize(qa['question'])
        ques_char_i = [[*token] for token in ques_token_i]

        for token in ques_token_i:
            word_counter[token.lower()] += 1
            for chr in token:
                char_counter[chr] += 1

        # Album ID counter
        for albumId in qa['album_ids']:
            albumId = str(albumId)
            if (albumId not in global_aids.keys()):
                global_aids[albumId] = 0
            global_aids[albumId] += 1


        # answer, choices
        answer_token_i = word_tokenize(qa['answer'])
        answer_chars_i = [[*token] for token in answer_token_i]
        for token in answer_token_i:
            word_counter[token.lower()] += 1
            for ch in token:
                char_counter[ch] += 1

        mul_choice_i = qa['multiple_choices_4'][:]  # copy it
        # remove the answer in choices
        answer_index_i = mul_choice_i.index(
        qa['answer'])  # this is for during testing, we need to reconstruct the answer in the original order
        mul_choice_i.remove(qa['answer'])  # will error if answer not in choice
        assert len(mul_choice_i) == 3
        mul_choice_chars_i = []  # char for choices
        for i, c in enumerate(mul_choice_i):
            mul_choice_i[i] = word_tokenize(c)
            mul_choice_chars_i.append([[*choices] for choices in mul_choice_i[i]])
            for choices in mul_choice_i[i]:
                word_counter[choices.lower()] += 1
                for chr in choices:
                    char_counter[chr] += 1
        if (debug):
            print("question:%s" % qa['question'])
            print(ques_token_i)
            print(ques_char_i)
            print("answer:%s" % (qa['answer']))
            print(answer_token_i)
            print(answer_chars_i)
            print("choices:%s" % ("/".join(qa['multiple_choices_4'])))
            print(mul_choice_i)
            print(mul_choice_chars_i)
            break

        ques_tokens.append(ques_token_i)
        ques_chars.append(ques_char_i)
        answer_tokens.append(answer_token_i)
        answer_chars.append(answer_chars_i)
        answer_index.append(answer_index_i)
        mul_choice.append(mul_choice_i)
        mul_choice_chars.append(mul_choice_chars_i)
        album_id.append([str(one) for one in qa['album_ids']])
        ques_id.append(question_id)
        idxs.append(idx)

    albums = {str(album['album_id']):album for album in args.albums}
    album_info = {}
    pid2feat = {}
    for albumId in tqdm(global_aids):
        album = albums[albumId]
        used = global_aids[albumId]
        temp = {'aid':album['album_id']}



        # album info
        temp['title'] = word_tokenize(album['album_title'])
        temp['title_c'] = [list(tok) for tok in temp['title']]
        #temp['description'] = list(map(word_tokenize, sent_tokenize(strip_tags(album['album_description']))))
        # treat description as one sentence
        #print('Album Desc', album['album_description'])
        temp['description'] = word_tokenize(strip_tags(album['album_description']))
        #print('Here  ',temp['description'])
        if(temp['description']):
            temp['description_c'] = [[*tok] for tok in temp['description']]

        # use _ to connect?
        if album['album_where'] is None:
            temp['where'] = []
            temp['where_c'] = []
        else:
            temp['where'] = word_tokenize(album['album_where'])
            temp['where_c'] = [[*tok] for tok in temp['where']]
        temp['when'] = word_tokenize(album['album_when'])
        temp['when_c'] = [[*tok] for tok in temp['when']]

        # photo info
        temp['photo_titles'] = [word_tokenize(title) for title in album['photo_titles']]
        for title in temp['photo_titles']:
            if(title):
                temp['photo_titles_c'] = [[*tok] for tok in title]

        temp['photo_ids'] = [str(pid) for pid in album['photo_ids']]
        assert len(temp['photo_ids']) == len(temp['photo_titles'])
        for pid in temp['photo_ids']:
            assert isinstance(pid, str)
            if pid not in pid2feat.keys():
                pid2feat[pid] = args.images[pid]

        concat_meta = []
        if(temp['title']):
            concat_meta += temp['title']
        if(temp['description']):
            concat_meta += temp['description']
        if(temp['where']):
            concat_meta += temp['where']
        if(temp['when']):
            concat_meta += temp['when']

        for title in temp['photo_titles']:
            if(title):
                for tok in title:
                    if(tok):
                        concat_meta += tok

        for t in concat_meta:
            #print(t)
            word_counter[t.lower()] += used
            for c in t:
                char_counter[c] += used
        album_info[albumId] = temp

    word2vec_dict = get_word2vec(args, word_counter)

    data = {
        'ques_tokens': ques_tokens,
        'ques_chars': ques_chars,
        'answer_tokens': answer_tokens,
        'answer_char': answer_chars,
        'answer_index': answer_index,  # the original answer idx in the choices list # this means the correct index
        'album_id': album_id,  # each is a list of aids
        'ques_id': ques_id,
        'idxs': idxs,
        'mul_choice': mul_choice,  # each is a list of wrong choices
        'mul_choice_chars': mul_choice_chars,
    }

    shared = {
        "albums" :album_info, # albumId -> photo_ids/title/when/where ...
        "pid2feat":pid2feat, # pid -> image feature
        "wordCounter":word_counter,
        "charCounter":char_counter,
        "word2vec":word2vec_dict,}

    print(shared)

    #print(word_counter)
    print("data:%s, char entry:%s, word entry:%s, word2vec entry:%s,album: %s/%s, image_feat:%s"%(data_type,len(char_counter),len(word_counter),len(word2vec_dict),len(album_info),len(albums),len(pid2feat)))
    pickle.dump(data,open(os.path.join(args.outpath,"%s_data.p"%data_type),"wb"))
    pickle.dump(shared,open(os.path.join(args.outpath,"%s_shared.p"%data_type),"wb"))


def getTrainValIds(qas, validlist, testidlist):
    testIds = [one.strip() for one in open(testidlist, "r").readlines()]

    valIds = []
    if validlist is not None:
        valIds = [one.strip() for one in open(validlist, "r").readlines()]

    trainIds = []

    for one in qas:
        qid = str(one['question_id'])
        if ((qid not in testIds) and (qid not in valIds)):
            trainIds.append(qid)

    # if validation id not provided, get from trainIds
    if validlist is None:
        valcount = int(len(trainIds) * 0.2)
        random.seed(1)
        random.shuffle(trainIds)
        random.shuffle(trainIds)
        valIds = trainIds[:valcount]
        trainIds = trainIds[valcount:]

    print("total trainId:%s,valId:%s,testId:%s, total qa:%s" % (len(trainIds), len(valIds), len(testIds), len(qas)))
    return trainIds, valIds, testIds


if __name__ == '__main__':
    args = get_args()
    mkdir(args.outpath)

    args.qas = json.load(open(args.datajson, "r"))
    args.albums = json.load(open(args.albumjson, "r"))
    if(args.imgfeat.endswith(".p")):
        print( "read pickle image feat.")
        imagedata = pickle.load(open(args.imgfeat,"r"))
        args.images = {}
        assert len(imagedata[0]) == len(imagedata[1])
        for i,pid in enumerate(imagedata[0]):
            args.images[pid] = imagedata[1][i]
    else:
        print("read npz image feat.")
        args.images = np.load(args.imgfeat)

    trainIds,valIds,testIds = getTrainValIds(args.qas,args.valids,args.testids)
    prepro_each(args, "train", trainIds, 0.0, 1.0)
    prepro_each(args, "val", valIds, 0.0, 1.0)
    prepro_each(args, "test", testIds, 0.0, 1.0)