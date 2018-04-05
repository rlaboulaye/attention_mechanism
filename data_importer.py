import os
import time
import math
import io
import collections
from itertools import izip
import numpy as np
import pickle

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
        prune=False
    ):
        if max_vocab_size is not None:
            max_vocab_size = int(max_vocab_size)
        if max_n_sentences is not None:
            max_n_sentences = int(max_n_sentences)
        if max_src_sentence_len is not None:
            max_src_sentence_len = int(max_src_sentence_len)
        if prune != False:
            prune = True

        self.src_lang_vocab_path = src_lang_vocab_path
        self.targ_lang_vocab_path = targ_lang_vocab_path
        self.src_lang_embedding_path = src_lang_embedding_path
        self.targ_lang_embedding_path = targ_lang_embedding_path
        self.src_lang_path = src_lang_path
        self.targ_lang_path = targ_lang_path
        self.max_vocab_size = max_vocab_size
        self.max_n_sentences = max_n_sentences
        self.max_src_sentence_len = max_src_sentence_len
        self.prune = prune

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
            for i, line in enumerate(f):
                word = line.strip()
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

        cache_targ_emb_path = self._get_cache_text_data_path()
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
            for line in f:
                if line != "":
                    word, str_embedding = line.strip().split(" ", 1)
                    if word in self.src_vocab_set:
                        word_2_embedding[word] = np.array(str_embedding.split(" "), dtype=float)

        embedding_size = word_2_embedding.itervalues().next().shape[0]
        word_2_embedding[self.special_tokens[self.UNKNOWN]] = np.zeros(embedding_size)
        return word_2_embedding

    def _init_text(self):
        self.unknown_src_word_count = 0
        self.unknown_targ_word_count = 0
        self.known_src_word_count = 0
        self.known_targ_word_count = 0

        hash_str = self.src_lang_vocab_path +\
            self.targ_lang_vocab_path +\
            self.src_lang_embedding_path +\
            self.targ_lang_embedding_path +\
            self.src_lang_path +\
            self.targ_lang_path +\
            str(self.max_vocab_size) +\
            str(self.max_n_sentences) +\
            str(self.max_src_sentence_len) +\
            str(self.prune)

        cache_text_data_path = self.cache_dir + str(abs(hash(hash_str))) + ".p"
        if os.path.isfile(cache_text_data_path):
            self.src_data, self.targ_data, self.src_data_by_seq_len_indices = pickle.load(open(cache_text_data_path, "rb"))
        else:
            self.src_data = []
            self.targ_data = []
            self.src_data_by_seq_len_indices = []
            with io.open(self.src_lang_path, "r", encoding='utf8') as f_src, io.open(self.targ_lang_path, "r", encoding='utf8') as f_targ:
                for src_line, targ_line in izip(f_src, f_targ):
                    src_sentence = self._parse_line(src_line, "src")
                    targ_sentence = self._parse_line(targ_line, "targ")

                    if (self.prune == False or None not in [src_sentence, targ_sentence]) and (self.max_src_sentence_len is None or len(src_sentence) <= self.max_src_sentence_len):
                        self.src_data.append(src_sentence)
                        self.targ_data.append(targ_sentence)

                        while len(self.src_data_by_seq_len_indices) <= len(src_sentence):
                            self.src_data_by_seq_len_indices.append([])
                        self.src_data_by_seq_len_indices[len(src_sentence)].append(len(self.src_data)-1)

                    if self.max_n_sentences is not None and len(self.src_data) >= self.max_n_sentences:
                        break
            pickle.dump([self.src_data, self.targ_data, self.src_data_by_seq_len_indices], open(cache_text_data_path, "wb"))

    def _get_cache_text_data_path(self):
        hash_str = str(abs(hash(
            self.src_lang_vocab_path +
            self.targ_lang_vocab_path +
            self.src_lang_embedding_path +
            self.targ_lang_embedding_path +
            self.src_lang_path +
            self.targ_lang_path +
            str(self.max_vocab_size) +
            str(self.max_n_sentences) +
            str(self.max_src_sentence_len) +
            str(self.prune)
        )))
        return self.cache_dir + "text_data_" + hash_str + ".p"


    def _parse_line(self, line, line_type):
        if line_type == "src":
            vocab_set = self.src_vocab_set
            word_2_embedding = self.src_word_2_embedding
        elif line_type == "targ":
            vocab_set = self.targ_vocab_set
            word_2_embedding = self.targ_word_2_embedding
        else:
            raise ValueError("invalid line_type")

        unknown_word_count = 0
        sentence = []
        for word in line.strip().split(" "):
            if word in vocab_set and word in word_2_embedding:
                sentence.append(word)
            else:
                unknown_word_count += 1
                sentence.append(self.special_tokens[self.UNKNOWN])
        sentence.append(self.special_tokens[self.EOS])

        if line_type == "src":
            self.unknown_src_word_count += unknown_word_count
            self.known_src_word_count += len(sentence) - unknown_word_count
        elif line_type == "targ":
            self.unknown_targ_word_count += unknown_word_count
            self.known_targ_word_count += len(sentence) - unknown_word_count
        else:
            raise ValueError("invalid line_type")

        if self.prune and unknown_word_count > 0:
            return None
        else:
            return sentence

    def _init_batching(self):
        self.batch_start_indices = [0 for _ in xrange(len(self.src_data_by_seq_len_indices))]

    def batch(self, sequence_length, batch_size):
        if sequence_length >= len(self.batch_start_indices):
            raise ValueError("sequence_length is too long")

        batch_start_index = self.batch_start_indices[sequence_length]
        batch_end_index = min(batch_start_index + batch_size, len(self.src_data_by_seq_len_indices))
        if batch_end_index == len(self.src_data_by_seq_len_indices):
            self.batch_start_indices[sequence_length] = 0
        else:
            self.batch_start_indices[sequence_length] = batch_end_index

        data_indices = self.src_data_by_seq_len_indices[sequence_length][batch_start_index:batch_end_index]

        batch_src_data = np.array([self._embed_src_sentence(self.src_data[i]) for i in data_indices])
        batch_src_data = np.swapaxes(batch_src_data, 0, 1)
        batch_targ_data = np.array([self._encode_targ_sentence(self.targ_data[i]) for i in data_indices])
        batch_targ_data = np.swapaxes(batch_targ_data, 0, 1)

        return batch_src_data, batch_targ_data

    def _embed_src_sentence(self, sentence):
        return np.array([self.src_word_2_embedding[word] for word in sentence])

    def _encode_targ_sentence(self, sentence):
        return np.array([self.targ_word_2_encoding[word] for word in sentence])

dataset = SentenceTranslationDataset(max_n_sentences=1e4, max_vocab_size=None, max_src_sentence_len=40)
# print dataset.known_src_word_count, dataset.unknown_src_word_count
# print dataset.known_targ_word_count, dataset.unknown_targ_word_count
print len(dataset.src_data), len(dataset.targ_data)
for i, item in enumerate(dataset.src_data_by_seq_len_indices):
    print i, len(item)

for i in xrange(10):
    print(dataset.batch(10, 25)[:2])
