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

    UNKNOWN_TOKEN = "<unk>"
    EOS_TOKEN = "</s>"

    # # default
    # def __init__(
    #     self,
    #     src_lang_vocab_path="./data/en-de/vocab.50K.en",
    #     targ_lang_vocab_path="./data/en-de/vocab.50K.de",
    #     src_lang_embedding_path="./data/fastText/wiki.en.vec",
    #     targ_lang_embedding_path="./data/fastText/wiki.de.vec",
    #     src_lang_path="./data/en-de/train.en",
    #     targ_lang_path="./data/en-de/train.de",
    #     max_vocab_size=None,
    #     max_n_sentences=None,
    #     max_src_sentence_len=None,
    #     prune_by_vocab=False,
    #     prune_by_embedding=False
    # ):

    # # embedding as vocab
    # def __init__(
    #     self,
    #     src_lang_vocab_path="./data/fastText/wiki.en.vec.vocab",
    #     targ_lang_vocab_path="./data/fastText/wiki.de.vec.vocab",
    #     src_lang_embedding_path="./data/fastText/wiki.en.vec",
    #     targ_lang_embedding_path="./data/fastText/wiki.de.vec",
    #     src_lang_path="./data/en-de/train.en",
    #     targ_lang_path="./data/en-de/train.de",
    #     max_vocab_size=None,
    #     max_n_sentences=None,
    #     max_src_sentence_len=None,
    #     prune_by_vocab=False,
    #     prune_by_embedding=False
    # ):

    # english to english
    def __init__(
        self,
        src_lang_vocab_path="./data/fastText/wiki.en.vec.vocab",
        targ_lang_vocab_path="./data/fastText/wiki.en.vec.vocab",
        src_lang_embedding_path="./data/fastText/wiki.en.vec",
        targ_lang_embedding_path="./data/fastText/wiki.en.vec",
        src_lang_path="./data/en-de/train.en",
        targ_lang_path="./data/en-de/train.en",
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

    def _open_file(self, path, mode, encoding="utf-8"):
        return io.open(path, mode, encoding=encoding)

    def _parse_word(self, word):
        word = word.strip().lower()
        word = self.html_parser.unescape(word)
        return word

    def _init_vocab(self):
        self.src_vocab, self.src_vocab_2_encoding = self._read_vocab(self.src_lang_vocab_path)
        self.targ_vocab, self.targ_vocab_2_encoding = self._read_vocab(self.targ_lang_vocab_path)

    def _read_vocab(self, path):
        vocab = []
        word_2_encoding = {}
        with self._open_file(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line == "" or (self.max_vocab_size is not None and i >= self.max_vocab_size):
                        break
                word = self._parse_word(line) # one word per line
                vocab.append(word)
                word_2_encoding[word] = i

        if self.UNKNOWN_TOKEN not in word_2_encoding:
            vocab.append(self.UNKNOWN_TOKEN)
            word_2_encoding[self.UNKNOWN_TOKEN] = i
            i += 1

        if self.EOS_TOKEN not in word_2_encoding:
            vocab.append(self.EOS_TOKEN)
            word_2_encoding[self.EOS_TOKEN] = i
            i += 1

        return vocab, word_2_encoding

    def _init_embedding(self):
        cache_src_emb_path = self._get_cache_src_emb_path()
        if os.path.isfile(cache_src_emb_path):
            self.src_word_2_embedding = pickle.load(open(cache_src_emb_path, "rb"))
        else:
            self.src_word_2_embedding = self._read_embedding(self.src_lang_embedding_path, self.src_vocab_2_encoding)
            pickle.dump(self.src_word_2_embedding, open(cache_src_emb_path, "wb"))

        cache_targ_emb_path = self._get_cache_targ_emb_path()
        if os.path.isfile(cache_targ_emb_path):
            self.targ_word_2_embedding = pickle.load(open(cache_targ_emb_path, "rb"))
        else:
            self.targ_word_2_embedding = self._read_embedding(self.targ_lang_embedding_path, self.targ_vocab_2_encoding)
            pickle.dump(self.targ_word_2_embedding, open(cache_targ_emb_path, "wb"))

    def _get_cache_src_emb_path(self):
        hash_str = str(abs(hash(
            self.src_lang_embedding_path + self.src_lang_vocab_path + str(self.max_vocab_size)
        )))
        return self.cache_dir + "src_emb_" + hash_str + ".p"

    def _get_cache_targ_emb_path(self):
        hash_str = str(abs(hash(
            self.targ_lang_embedding_path + self.targ_lang_vocab_path + str(self.max_vocab_size)
        )))
        return self.cache_dir + "targ_emb_" + hash_str + ".p"

    def _read_embedding(self, path, vocab_lookup):
        word_2_embedding = {}
        with self._open_file(path, "r") as f:
            for i, line in enumerate(f):
                if i != 0 and line != "":
                    word, str_embedding = line.strip().split(" ", 1)
                    word = self._parse_word(word)
                    if word in vocab_lookup:
                        word_2_embedding[word] = np.array(str_embedding.split(" "), dtype=float)
                if len(word_2_embedding) == len(vocab_lookup): # will not get any more words, no matter how much we search
                    break
        embedding_size = word_2_embedding.itervalues().next().shape[0]
        word_2_embedding[self.UNKNOWN_TOKEN] = np.zeros(embedding_size)
        if self.EOS_TOKEN not in word_2_embedding:
            raise ValueError(path + " does not contain an end of sequence embedding")
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
        with self._open_file(self.src_lang_path, "r") as f_src, self._open_file(self.targ_lang_path, "r") as f_targ:
            for i, (src_line, targ_line) in enumerate(izip(f_src, f_targ)):
                src_sentence = self._parse_text_line(src_line, "src")
                targ_sentence = self._parse_text_line(targ_line, "targ")

                if None in [src_sentence, targ_sentence]: # it was pruned
                    self.pruned_sentence_count += 1
                else:
                    self.src_data.append(src_sentence)
                    self.targ_data.append(targ_sentence)

                    # store indices to src data by sequence length
                    while len(self.src_data_by_seq_len_indices) <= len(src_sentence):
                        self.src_data_by_seq_len_indices.append([])
                    self.src_data_by_seq_len_indices[len(src_sentence)].append(len(self.src_data)-1)

                if self.max_n_sentences is not None and len(self.src_data) >= self.max_n_sentences:
                    break

    def _parse_text_line(self, line, line_type):
        if line_type == "src":
            vocab_lookup = self.src_vocab_2_encoding
            word_2_embedding = self.src_word_2_embedding
        elif line_type == "targ":
            vocab_lookup = self.targ_vocab_2_encoding
            word_2_embedding = self.targ_word_2_embedding
        else:
            raise ValueError("invalid line_type")

        unknown_vocab_count = 0
        unknown_embedding_count = 0
        sentence = []
        for word in line.strip().split(" "):
            word = self._parse_word(word)
            if word not in vocab_lookup:
                unknown_vocab_count += 1
                word = self.UNKNOWN_TOKEN
            elif word not in word_2_embedding:
                unknown_embedding_count += 1
                word = self.UNKNOWN_TOKEN
            sentence.append(word)
        sentence.append(self.EOS_TOKEN)

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
        self.batch_start_indices = [0] * len(self.src_data_by_seq_len_indices)

    def batch(self, src_seq_len, batch_size, shuffle=True):
        if src_seq_len >= len(self.batch_start_indices):
            raise ValueError("src_seq_len is too long")

        batch_start_index = self.batch_start_indices[src_seq_len]
        batch_end_index = min(batch_start_index + batch_size, len(self.src_data_by_seq_len_indices))
        if batch_end_index == len(self.src_data_by_seq_len_indices):
            self.batch_start_indices[src_seq_len] = 0
        else:
            self.batch_start_indices[src_seq_len] = batch_end_index

        data_indices = self.src_data_by_seq_len_indices[src_seq_len][batch_start_index:batch_end_index]
        print self.src_data[data_indices[0]]
        print self.targ_data[data_indices[0]]
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
                targ_data = np.pad(targ_data, pad_width, mode="constant", constant_values=self.targ_vocab_2_encoding[self.EOS_TOKEN])
            batch_targ_data.append(targ_data)

        batch_targ_data = np.array(batch_targ_data)
        batch_targ_data = np.swapaxes(batch_targ_data, 0, 1)

        return batch_src_data, batch_targ_data

    def _embed_src_sentence(self, sentence):
        return np.array([self.src_word_2_embedding[word] for word in sentence])

    def _encode_targ_sentence(self, sentence):
        return np.array([self.targ_vocab_2_encoding[word] for word in sentence])

dataset = SentenceTranslationDataset(max_n_sentences=1e6, max_vocab_size=1e4, max_src_sentence_len=40, prune_by_vocab=True, prune_by_embedding=True)#, prune_by_embedding=True)
print dataset.unknown_src_vocab_count / dataset.known_src_word_count, dataset.unknown_src_embedding_count / dataset.known_src_word_count
print dataset.unknown_targ_vocab_count / dataset.known_targ_word_count, dataset.unknown_targ_embedding_count / dataset.known_targ_word_count
print dataset.pruned_sentence_count
print len(dataset.src_data), len(dataset.targ_data)

# for i, item in enumerate(dataset.src_data_by_seq_len_indices):
#     print i, len(item)

for i in xrange(7):
    dataset.batch(13, 17)


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
