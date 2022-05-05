import jieba
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torchtext.transforms import VocabTransform
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelEncoder
import torch
import scipy.io as io
import numpy as np


def word_split(text):
    print("正在进行分词....")
    f = open('dataset/data/stop_word.txt', 'r')
    stop_word = [i.strip() for i in f.readlines()]
    f.close()
    words = []
    for i in text:
        ws = [i for i in jieba.lcut(i) if len(i) > 1 and i not in stop_word]
        words.append(ws)

    return words


def word2vector():

    file = open('dataset/data/toutiao_cat_data.txt', 'r')
    text = file.readlines()[:10000]
    file.close()
    label = [i.split('_')[5] for i in text]
    text = [i.split('_')[7] for i in text]
    words = word_split(text)
    print("合并词list...")
    ws = sum(words, [])
    set_ws = set(ws)
    print("建立词典...")
    ds = dict(zip(set_ws, list(range(1, 1+len(set_ws)))))
    ordered_dict = OrderedDict(ds)
    my_vocab = vocab(ordered_dict, specials=['<UNK>', '<SEP>'])
    vocab_transform = VocabTransform(my_vocab)
    vector = vocab_transform(words)
    vector = [torch.tensor(i) for i in vector]
    lengths = np.array([len(i) for i in vector])
    pad_vector = pad_sequence(vector, batch_first=True)
    label_encoder = LabelEncoder()
    x = pad_vector
    y = label_encoder.fit_transform(label)
    zeros_index = lengths == 0
    x = x[~zeros_index]
    y = y[~zeros_index]
    lengths = lengths[~zeros_index]
    data = {'x': x.numpy(), 'y':y, 'length':lengths}
    print("保存数据....")
    io.savemat('dataset/data/w2v.mat', data)
    print("保存成功!!!")
