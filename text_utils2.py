from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import re
from nltk.corpus import stopwords
import string
import itertools
from collections import Counter
from itertools import count
import torch
from tqdm import tqdm

stop_words = set(stopwords.words('english'))

def get_char_vector():

    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    assert len(chars) == 26
    vocab = {}


def get_word_vector(vocab, emb='glove'):

    if emb == 'glove':
        fname = 'glove.6B.300d.txt'

        with open(fname, 'rt', encoding='utf8') as fi:
            full_content = fi.read().strip().split('\n')

        data = {}
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            if i_word not in vocab.keys():
                continue
            i_embeddings = [float(val)
                            for val in full_content[i].split(' ')[1:]]
            data[i_word] = i_embeddings

    elif emb == 'fasttext':
        fname = 'wiki-news-300d-1M.vec'

        fin = io.open(fname, 'r', encoding='utf-8',
                      newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}

        for line in tqdm(fin, total=1000000, desc='loading fasttext vocabs...'):
            tokens = line.rstrip().split(' ')
            if tokens[0] not in vocab.keys():
                continue
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)

    else:
        raise Exception('emb not implemented')

    w = []
    find = 0
    for word in vocab.keys():
        try:
            w.append(torch.tensor(data[word]))
            find += 1
        except:
            w.append(torch.rand(300))

    print('found', find, 'words in', emb)
    return torch.stack(w, dim=0)


def data_preprocessing(text, remove_stopword=False):

    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]
    text = ' '.join(text)
    return text


def create_vocab(corpus, vocab_size=30000):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    if vocab_size > len(sorted_words):
        v = len(sorted_words)
    else:
        v = vocab_size - 4

    vocab_to_int = {w: i + 4 for i, (w, c) in enumerate(sorted_words[:v])}

    vocab_to_int['<pad>'] = 0
    vocab_to_int['<unk>'] = 1
    vocab_to_int['<cls>'] = 2
    vocab_to_int['<sep>'] = 3
    print('vocab size', len(vocab_to_int))

    return vocab_to_int


class Textset(Dataset):
    def __init__(self, prem, hyp, label, vocab, max_len):
        super().__init__()

        new_text = []
        for t in prem:
            if len(t) > max_len:
                t = t[:max_len]
                new_text.append(t)
            else:
                new_text.append(t)
        self.x1 = new_text
        new_text = []
        for t in hyp:
            if len(t) > max_len:
                t = t[:max_len]
                new_text.append(t)
            else:
                new_text.append(t)
        self.x2 = new_text
        self.y = label
        self.vocab = vocab

    def collate(self, batch):

        x1 = [torch.tensor(x1) for x1, x2, y in batch]
        x2 = [torch.tensor(x2) for x1, x2, y in batch]
        y = [y for x1, x2, y in batch]
        x1_tensor = pad_sequence(x1, True)
        x2_tensor = pad_sequence(x2, True)
        y = torch.tensor(y)
        pad_mask1 = ~(x1_tensor == 0).to(x1_tensor.device)
        pad_mask2 = ~(x2_tensor == 0).to(x2_tensor.device)

        return x1_tensor, pad_mask1, x2_tensor, pad_mask2, y

    def convert2id(self, text):
        r = []
        for word in text.split():
            if word in self.vocab.keys():
                r.append(self.vocab[word])
            else:
                r.append(self.vocab['<unk>'])
        return r

    def __getitem__(self, idx):
        text1 = self.x1[idx]
        word_id1 = self.convert2id(text1)
        text2 = self.x2[idx]
        word_id2 = self.convert2id(text2)
        return word_id1, word_id2, self.y[idx]

    def __len__(self):
        return len(self.x1)