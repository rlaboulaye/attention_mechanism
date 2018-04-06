# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import math
import io
import collections
from itertools import izip
import numpy as np
import pickle
from HTMLParser import HTMLParser

from torch.utils.data import Dataset


class SentenceTranslationDataset(Dataset):

    UNKNOWN = "UNKNOWN"
    EOS = "EOS"

    def __init__(
        self,
        src_lang_vocab_path="./data/en-de/vocab.50K.en",
        targ_lang_vocab_path="./data/en-de/vocab.50K.de",
        src_lang_embedding_path="./data/fastText/wiki.en.vec",
        targ_lang_embedding_path="./data/fastText/wiki.de.vec",
        src_lang_path="./data/en-de/train.en",
        targ_lang_path="./data/en-de/train.de",
        max_vocab_size=None,
        max_n_sentences=None,
        max_src_sentence_len=None,
        prune_by_vocab=False,
        prune_by_embedding=False
    ):
        if max_vocab_size is not None:
            max_vocab_size = int(max_vocab_size)
        if max_n_sentences is not None:
            max_n_sentences = int(max_n_sentences)
        if max_src_sentence_len is not None:
            max_src_sentence_len = int(max_src_sentence_len)
        if prune_by_vocab != False:
            prune_by_vocab = True
        if prune_by_embedding != False:
            prune_by_embedding = True

        self.src_lang_vocab_path = src_lang_vocab_path
        self.targ_lang_vocab_path = targ_lang_vocab_path
        self.src_lang_embedding_path = src_lang_embedding_path
        self.targ_lang_embedding_path = targ_lang_embedding_path
        self.src_lang_path = src_lang_path
        self.targ_lang_path = targ_lang_path
        self.max_vocab_size = max_vocab_size
        self.max_n_sentences = max_n_sentences
        self.max_src_sentence_len = max_src_sentence_len
        self.prune_by_vocab = prune_by_vocab
        self.prune_by_embedding = prune_by_embedding

        self.html_parser = HTMLParser()

        self.cache_dir = "./.cache/"
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        self._init_vocab()
        self._init_embedding()
        self._init_text()
        self._init_batching()

    def _init_vocab(self):
        '''
        line 0 - unknown token
        line 2 - end of sentence token
        '''
        # todo consider deleting src_vocab, it is not used anywhere else
        self.src_vocab, self.src_word_2_encoding = self._read_vocab(self.src_lang_vocab_path, self.max_vocab_size)
        self.src_vocab_set = set(self.src_vocab)
        self.targ_vocab, self.targ_word_2_encoding = self._read_vocab(self.targ_lang_vocab_path, self.max_vocab_size)
        self.targ_vocab_set = set(self.targ_vocab)

        self.special_tokens = {
            self.UNKNOWN: self.src_vocab[0],
            self.EOS: self.src_vocab[2]
        }

    def _init_targ_encoding(self):
        self.targ_word_2_encoding = {}
        targ_word_encoding_index = 0
        for word in self.targ_vocab_set:
            self.targ_word_2_encoding[word] = targ_word_encoding_index
            targ_word_encoding_index += 1

    def _read_vocab(self, path, max_vocab_size):
        vocab = []
        word_2_encoding = {}
        with io.open(path, "r", encoding="utf-8") as f:
        # with open(path, "r") as f:
            for i, line in enumerate(f):
                word = line.strip().lower() # one word per line
                vocab.append(word)
                word_2_encoding[word] = i
                if max_vocab_size is not None and i >= max_vocab_size:
                    break
        return vocab, word_2_encoding

    def _init_embedding(self):
        cache_src_emb_path = self._get_cache_src_emb_path()
        if os.path.isfile(cache_src_emb_path):
            self.src_word_2_embedding = pickle.load(open(cache_src_emb_path, "rb"))
        else:
            self.src_word_2_embedding = self._read_embedding(self.src_lang_embedding_path)
            pickle.dump(self.src_word_2_embedding, open(cache_src_emb_path, "wb"))

        cache_targ_emb_path = self._get_cache_targ_emb_path()
        if os.path.isfile(cache_targ_emb_path):
            self.targ_word_2_embedding = pickle.load(open(cache_targ_emb_path, "rb"))
        else:
            self.targ_word_2_embedding = self._read_embedding(self.targ_lang_embedding_path)
            pickle.dump(self.targ_word_2_embedding, open(cache_targ_emb_path, "wb"))

    def _get_cache_src_emb_path(self):
        hash_str = str(abs(hash(
            self.src_lang_embedding_path + self.src_lang_vocab_path + str(self.max_vocab_size)
        )))
        return self.cache_dir + "src_emb_" + hash_str + ".p"

    def _get_cache_targ_emb_path(self):
        hash_str = str(abs(hash(
            self.targ_lang_embedding_path + self.src_lang_vocab_path + str(self.max_vocab_size)
        )))
        return self.cache_dir + "targ_emb_" + hash_str + ".p"

    def _read_embedding(self, path):
        word_2_embedding = {}
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i != 0 and line != "":
                    # print line
                    # if i == 50:
                    #     exit(1)
                    word, str_embedding = line.strip().split(" ", 1)
                    word = word.lower()
                    if word in self.src_vocab_set:
                        word_2_embedding[word] = np.array(str_embedding.split(" "), dtype=float)
                    # else:
                    #     print word

        embedding_size = word_2_embedding.itervalues().next().shape[0]
        word_2_embedding[self.special_tokens[self.UNKNOWN]] = np.zeros(embedding_size)
        return word_2_embedding

    def _init_text(self):
        self.known_src_word_count = 0
        self.unknown_src_vocab_count = 0
        self.unknown_src_embedding_count = 0
        self.known_targ_word_count = 0
        self.unknown_targ_vocab_count = 0
        self.unknown_targ_embedding_count = 0
        self.pruned_sentence_count = 0

        self.src_data = []
        self.targ_data = []
        self.src_data_by_seq_len_indices = []
        with io.open(self.src_lang_path, "r", encoding='utf8') as f_src, io.open(self.targ_lang_path, "r", encoding='utf8') as f_targ:
        # with open(self.src_lang_path, "r") as f_src, open(self.targ_lang_path, "r") as f_targ:
            for i, (src_line, targ_line) in enumerate(izip(f_src, f_targ)):
                src_sentence = self._parse_line(src_line, "src")
                targ_sentence = self._parse_line(targ_line, "targ")

                if None in [src_sentence, targ_sentence]: # it was pruned
                    self.pruned_sentence_count += 1
                    # print i
                else:
                    self.src_data.append(src_sentence)
                    self.targ_data.append(targ_sentence)

                    while len(self.src_data_by_seq_len_indices) <= len(src_sentence):
                        self.src_data_by_seq_len_indices.append([])
                    self.src_data_by_seq_len_indices[len(src_sentence)].append(len(self.src_data)-1)

                if self.max_n_sentences is not None and len(self.src_data) >= self.max_n_sentences:
                    break

    def _parse_line(self, line, line_type):
        if line_type == "src":
            vocab_set = self.src_vocab_set
            word_2_embedding = self.src_word_2_embedding
        elif line_type == "targ":
            vocab_set = self.targ_vocab_set
            word_2_embedding = self.targ_word_2_embedding
            # line = line.replace(u"\xfc", u"\xc3\xbc")# "ü")
            # "Ä","ä","Ö","ö","Ü",,"ß"
        else:
            raise ValueError("invalid line_type")

        line = line.lower()
        line = self.html_parser.unescape(line)

        unknown_vocab_count = 0
        unknown_embedding_count = 0
        sentence = []
        for word in line.strip().split(" "):
            if word not in vocab_set:
                unknown_vocab_count += 1
                word = self.special_tokens[self.UNKNOWN]
            elif word not in word_2_embedding:
                # if line_type == "targ":
                #     print word
                unknown_embedding_count += 1
                word = self.special_tokens[self.UNKNOWN]
            sentence.append(word)
        sentence.append(self.special_tokens[self.EOS])

        if line_type == "src":
            self.known_src_word_count += len(sentence) - unknown_vocab_count - unknown_embedding_count
            self.unknown_src_vocab_count += unknown_vocab_count
            self.unknown_src_embedding_count += unknown_embedding_count
        elif line_type == "targ":
            self.known_targ_word_count += len(sentence) - unknown_vocab_count - unknown_embedding_count
            self.unknown_targ_vocab_count += unknown_vocab_count
            self.unknown_targ_embedding_count += unknown_embedding_count

        if self.prune_by_vocab == True and unknown_vocab_count > 0:
            return None
        elif self.prune_by_embedding == True and unknown_embedding_count > 0:
            return None
        else:
            return sentence

    def _init_batching(self):
        self.batch_start_indices = [0 for _ in xrange(len(self.src_data_by_seq_len_indices))]

    def batch(self, src_seq_len, batch_size):
        if src_seq_len >= len(self.batch_start_indices):
            raise ValueError("src_seq_len is too long")

        batch_start_index = self.batch_start_indices[src_seq_len]
        batch_end_index = min(batch_start_index + batch_size, len(self.src_data_by_seq_len_indices))
        if batch_end_index == len(self.src_data_by_seq_len_indices):
            self.batch_start_indices[src_seq_len] = 0
        else:
            self.batch_start_indices[src_seq_len] = batch_end_index

        data_indices = self.src_data_by_seq_len_indices[src_seq_len][batch_start_index:batch_end_index]

        batch_src_data = np.array([self._embed_src_sentence(self.src_data[i]) for i in data_indices])
        batch_src_data = np.swapaxes(batch_src_data, 0, 1)

        # get max targ seq len
        max_targ_seq_len = 0
        jagged_batch_targ_data = []
        for i in data_indices:
            jagged_batch_targ_data.append(self._encode_targ_sentence(self.targ_data[i]))
            max_targ_seq_len = max(max_targ_seq_len, len(jagged_batch_targ_data[-1]))

        # pad targ batch to max seq len and embed
        batch_targ_data = []
        for targ_data in jagged_batch_targ_data:
            pad_width = (0, max_targ_seq_len - len(targ_data))
            if pad_width != (0,0):
                targ_data = np.pad(targ_data, pad_width, mode="constant", constant_values=self.targ_word_2_encoding[self.special_tokens[self.EOS]])
            batch_targ_data.append(targ_data)

        batch_targ_data = np.array(batch_targ_data)
        batch_targ_data = np.swapaxes(batch_targ_data, 0, 1)

        return batch_src_data, batch_targ_data

    def _embed_src_sentence(self, sentence):
        return np.array([self.src_word_2_embedding[word] for word in sentence])

    def _encode_targ_sentence(self, sentence):
        return np.array([self.targ_word_2_encoding[word] for word in sentence])

dataset = SentenceTranslationDataset(max_n_sentences=1e5, max_vocab_size=None, max_src_sentence_len=40)#, prune_by_embedding=True)
print dataset.unknown_src_vocab_count / dataset.known_src_word_count, dataset.unknown_src_embedding_count / dataset.known_src_word_count
print dataset.unknown_targ_vocab_count / dataset.known_targ_word_count, dataset.unknown_targ_embedding_count / dataset.known_targ_word_count
print dataset.pruned_sentence_count
print len(dataset.src_data), len(dataset.targ_data)

# for i, item in enumerate(dataset.src_data_by_seq_len_indices):
#     print i, len(item)

for i in xrange(10):
    dataset.batch(10, 25)[:2]


# emb_path = "./data/fastText/wiki.de.vec"
# word_2_embedding = {}
# with io.open(emb_path, "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         if i != 0 and line != "":
#             word, str_embedding = line.strip().split(" ", 1)
#             word = word.lower()
#             word_2_embedding[word] = np.array(str_embedding.split(" "), dtype=float)
#         if i == 100: 
#             break
# print word_2_embedding.keys()

# lang_path = "./data/en-de/"
# with io.open(lang_path, "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         print line
#         if i == 10:
#             break
