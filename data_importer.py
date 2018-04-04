import time
import math
import io
import collections
from itertools import izip
import numpy as np

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
        max_n_sentences=None,
        max_vocab_size=None,
        prune=False
    ):
        if max_n_sentences is not None:
            max_n_sentences = int(max_n_sentences)
        if max_vocab_size is not None:
            max_vocab_size = int(max_vocab_size)
        if prune != False:
            prune = True

        self._init_vocab(src_lang_vocab_path, targ_lang_vocab_path, max_vocab_size)
        self._init_embedding(src_lang_embedding_path, targ_lang_embedding_path)
        self._init_text(src_lang_path, targ_lang_path, max_n_sentences, prune)
        self._init_batching()

    def _init_vocab(self, src_lang_vocab_path, targ_lang_vocab_path, max_vocab_size):
        '''
        line 0 - unknown token
        line 2 - end of sentence token
        '''
        # todo consider deleting src_vocab, it is not used anywhere else
        self.src_vocab, self.src_word_2_encoding = self._read_vocab(src_lang_vocab_path, max_vocab_size)
        self.src_vocab_set = set(self.src_vocab)
        self.targ_vocab, self.targ_word_2_encoding = self._read_vocab(targ_lang_vocab_path, max_vocab_size)
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

    def _init_embedding(self, src_lang_embedding_path, targ_lang_embedding_path):
        self.src_word_2_embedding = self._read_embedding(src_lang_embedding_path)
        self.targ_word_2_embedding = self._read_embedding(targ_lang_embedding_path)

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

    def _init_text(self, src_lang_path, targ_lang_path, max_n_sentences, prune):
        self.unknown_src_word_count = 0
        self.unknown_targ_word_count = 0
        self.known_src_word_count = 0
        self.known_targ_word_count = 0

        self.src_data = []
        self.targ_data = []
        with io.open(src_lang_path, "r", encoding='utf8') as f_src, io.open(targ_lang_path, "r", encoding='utf8') as f_targ:
            for src_line, targ_line in izip(f_src, f_targ):
                src_sentence = self._parse_line(src_line, "src", prune)
                targ_sentence = self._parse_line(targ_line, "targ", prune)

                if prune == False or None not in [src_sentence, targ_sentence]:
                    self.src_data.append(src_sentence)
                    self.targ_data.append(targ_sentence)

                if max_n_sentences is not None and len(self.src_data) >= max_n_sentences:
                    break

    def _parse_line(self, line, line_type, prune):
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

        if prune and unknown_word_count > 0:
            return None
        else:
            return sentence

    def _init_batching(self):
        pass

    def batch(self, sequence_length, batch_size):
        pass

    # def embed_src_sentence(sentence):
    #     result = []
    #     for word in sentence:
    #         if word == self.special_tokens[self.UNKNOWN] or word not in src_word_2_str_embedding:
    #             embedding = self.special_token_str_embedding[self.UNKNOWN]
    #         elif word == self.special_tokens[self.EOS]:
    #             embedding = self.special_token_str_embedding[self.EOS]
    #         else:
    #             embedding = src_word_2_str_embedding[word] + " 0"
    #         embedding = np.array(embedding.split(" "), dtype=float)
    #         result.append(embedding)
    #     return result

    # self.src_data = map(embed_src_sentence, self.src_data)



dataset = SentenceTranslationDataset(max_n_sentences=1e3, max_vocab_size=1e3)
print dataset.known_src_word_count, dataset.unknown_src_word_count
print dataset.known_targ_word_count, dataset.unknown_targ_word_count
