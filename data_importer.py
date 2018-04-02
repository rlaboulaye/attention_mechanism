import time
import math
import io
import collections

import numpy as np

from torch.utils.data import Dataset


class SentenceTranslationDataset(Dataset):

    def __init__(
        self,
        src_lang_path="./data/fr-en/europarl-v7.fr-en.en",
        targ_lang_path="./data/fr-en/europarl-v7.fr-en.fr",
        src_lang_embedding_path="./glove.6B/glove.6B.50d.txt",
        max_n_sentences=1e4,
        max_vocab_size=1e5
    ):
        max_n_sentences = int(max_n_sentences)
        max_vocab_size = int(max_vocab_size)

        self._init_src_embedding(src_lang_embedding_path)
        self._init_targ_encoding(max_vocab_size)
        self._load_embedding_safe_data(src_lang_path, targ_lang_path, max_n_sentences)
        self._prune_data_by_counts_and_encode(max_vocab_size)

    def _init_src_embedding(self, path):
        self.src_word_2_embedding = {}
        with open(path, "r") as f:
            for line in f:
                if line != "":
                    word, str_embedding = line.strip().split(" ", 1)
                    embedding = np.array(str_embedding.split(" "), dtype=float)
                    self.src_word_2_embedding[word] = embedding
        self.src_embedding_size = self.src_word_2_embedding.itervalues().next().shape[0]

    def _init_targ_encoding(self, max_vocab_size):
        self.targ_word_2_encoding = {}
        self.targ_word_encoding_index = 0

    def _load_embedding_safe_data(self, src_lang_path, targ_lang_path, max_n_sentences):
        self.src_data = []
        self.targ_data = []

        with io.open(src_lang_path, "r", encoding='utf8') as f_src, io.open(targ_lang_path, "r", encoding='utf8') as f_targ:
            for src_line, targ_line in zip(f_src, f_targ):
                src_sentence, targ_sentence = src_line.strip().split(" "), targ_line.strip().split(" ")
                # skip sentences which cannot be embedded
                prune_sentence = False
                for i, word in enumerate(src_sentence):
                    word = word.strip()
                    src_sentence[i] = word
                    if word not in self.src_word_2_embedding:
                        prune_sentence = True
                        break
                if prune_sentence == True:
                    continue

                targ_sentence = map(lambda x: x.strip(), targ_sentence)

                self.src_data.append(src_sentence)
                self.targ_data.append(targ_sentence)

                if max_n_sentences is not None and len(self.src_data) >= max_n_sentences:
                    break

    def _prune_data_by_counts_and_encode(self, max_vocab_size):
        self.src_vocab_counts = collections.Counter()
        for src_sentence in self.src_data:
            self.src_vocab_counts.update(src_sentence)
        self.targ_vocab_counts = collections.Counter()
        for targ_sentence in self.targ_data:
            self.targ_vocab_counts.update(targ_sentence)

        most_common_src_words = self.src_vocab_counts.most_common(max_vocab_size)
        print type(most_common_src_words)
        input("here")
        most_common_targ_words = set(self.targ_vocab_counts.most_common(max_vocab_size))
        self.targ_embedding_size = min(max_vocab_size, len(most_common_targ_words))

        old_src_data = self.src_data
        self.src_data = []
        old_targ_data = self.targ_data
        self.targ_data = []
        for src_sentence, targ_sentence in zip(old_src_data, old_targ_data):
            if set(src_sentence).issubset(most_common_src_words) and set(targ_sentence).issubset(most_common_targ_words):
                print "here"
                self.src_data.append(self._embed_src_sentence(src_sentence))
                self.targ_data.append(self.embed_targ_sentence(targ_sentence, max_vocab_size))

    def _embed_src_sentence(self, src_sentence):
        result = np.empty((len(src_sentence), self.src_embedding_size))
        for i, word in enumerate(src_sentence):
            result[i] = self.src_word_2_embedding[word]
        return result

    def _embed_targ_sentence(self, targ_sentence):
        result = np.zeros((len(targ_sentence), self.targ_embedding_size))
        for i, word in enumerate(targ_sentence):
            word_encoding = self._get_targ_word_encoding(word)
            result[i][word_encoding] = 1
        return result

    def _get_targ_word_encoding(self, word):
        if word not in self.targ_word_2_encoding:
            self.targ_word_2_encoding[word] = self.targ_word_encoding_index
            self.targ_word_encoding_index += 1
            if self.targ_word_encoding_index > self.targ_embedding_size:
                raise Exception("There are more target words than can be embedded.")
        return self.targ_word_2_encoding[word]

    def __len__(self):
        return src_lang.shape[0]

    def __getitem__(self, index):
        return self.src_lang[index], self.targ_lang[index]

dataset = SentenceTranslationDataset()
print dataset.src_data[0].shape
print dataset.targ_data[0].shape

class BookSentences(Dataset):

    def __init__(self, max_length = 20, min_length = 1):
        self.max_length = max_length
        self.min_length = min_length
        self.data = []
        self.token_counts = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return BookSentences.parse(self.data[idx])

    @staticmethod
    def parse(sentence):
        result = []
        for token in sentence.strip().split():
            if not token == u'.':
                result.append(token)
        result.append(u'.')
        return result

    def append(self, sentence):
        self.data.append(sentence)

    # 74004229 rows, 8 GB memory, ~60s to load entire file with no embeddings
    # 67334174 rows, 8 GB memory, ~430s to load entire file with glove embeddings
    @staticmethod
    def load_from_file(file_name = "books_in_sentences.txt", max_rows = 1e5, max_length = None):
        book_sentences = BookSentences()
        count = 0
        with io.open(file_name, 'r', encoding = 'utf8') as data_file:
            for sentence in data_file:
                book_sentences.append(sentence)
                count += 1
                if max_rows is not None and count >= max_rows:
                    break
        return book_sentences

    @staticmethod
    def load_by_length(file_name = "books_in_sentences.txt", max_rows = 1e5, min_length = 1, max_length = 20):
        start_time = time.time()
        data = [BookSentences(max_length=x, min_length=x) for x in range(min_length, max_length + 1)]

        count = 0
        with io.open(file_name, 'r', encoding = 'utf8') as data_file:
            for sentence in data_file:
                parsed_sentence = BookSentences.parse(sentence)
                length = len(parsed_sentence)

                if max_length is not None and length > max_length:
                    continue

                if min_length is not None and length < min_length:
                    continue

                data[length - min_length].append(' '.join(parsed_sentence))
                count += 1

                if max_rows is not None and count >= max_rows:
                    break

        print("Loaded "+ str(len(data)) +" datasets in {0:.2f} seconds".format(time.time() - start_time))
        return data

    @staticmethod
    def load_most_common_tokens(load_file = "books_in_sentences.txt", save_file = "most_common_tokens.txt", max_vocab_size = 1e3):
        start_time = time.time()
        try:
            with io.open(save_file, 'r', encoding = 'utf8') as sf:
                common_tokens = []
                for line in sf:
                    common_tokens.append(line.strip())
                if len(common_tokens) == 0:
                    raise ValueError("Save file doesn't exist")
        except:
            token_counts = {}
            with io.open(load_file, 'r', encoding = 'utf8') as lf:
                for sentence in lf:
                    for token in sentence.strip().split():
                        if token not in token_counts:
                            token_counts[token] = 1
                        else:
                            token_counts[token] += 1
            token_order = sorted([(token_counts[token], token) for token in token_counts], reverse=True)
            common_tokens = [x[1] for x in token_order]

            with io.open(save_file, 'w', encoding = 'utf8') as sf:
                for token_count, token in common_tokens:
                    sf.write(token + '\n')

        if max_vocab_size is None:
            print("Loaded "+ str(len(common_tokens)) +" tokens in {0:.2f} seconds".format(time.time() - start_time))
            return common_tokens
        else:
            print("Loaded "+ str(max_vocab_size) +" tokens in {0:.2f} seconds".format(time.time() - start_time))
            return common_tokens[:max_vocab_size]


# 2196016 rows, 6 GB memory, ~25 seconds to load entire file
class GloveEmbeddings():

    def __init__(self, file_name = "glove.txt", vocabulary = None):
        start_time = time.time()
        self.file_name = file_name
        self.data = {}
        self.index_to_token_map = {}
        self.count = 0
        self.vocabulary = vocabulary
        if self.vocabulary is not None:
            self.vocabulary = set(self.vocabulary)
            self.vocabulary.add(".") # use period for end of sentence
            self.vocabulary.add("0") # use zero for numbers
            self.vocabulary.add("unknown") # use unknown for oov words

        with io.open(file_name, 'r', encoding = 'utf8') as data_file:
            for line in data_file:
                line = line.split(" ", 1)
                self.add(line[0], line[1])
        print("Loaded "+ str(len(self.data)) +" embeddings in {0:.2f} seconds".format(time.time() - start_time))


    def add(self, word, embedding):
        if self.vocabulary is not None:
            if word not in self.vocabulary:
                return False
        self.data[word] = (embedding, self.count)
        self.index_to_token_map[self.count] = word
        self.count += 1
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return [float(x) for x in self.data[key][0].split(" ")]

    def get_word(self, index):
        return self.index_to_token_map[index]

    def get_index(self, token):
        if token.isnumeric():
            return self.data["0"][1]
        elif token not in self.data:
            return self.data["unknown"][1]
        else:
            return self.data[token][1]

    def get_indexes(self, tokens):
        indexed_tokens = []
        for token in tokens:
            indexed_tokens.append(self.get_index(token))
        return indexed_tokens

    def __contains__(self, key):
        return key in self.data

    def embed(self, batch):
        embedded_batch = []
        for sentence in batch:
            embedded_sentence = []
            for token in sentence:
                if token.isnumeric():
                    embedded_sentence.append(self["0"])
                elif token not in self.data:
                    embedded_sentence.append(self["unknown"])
                else:
                    embedded_sentence.append(self[token])
            embedded_batch.append(embedded_sentence)
        return embedded_batch

    def save(self, save_file = None):
        if save_file is None:
            save_file = "glove_" + str(len(self.data)) + ".txt"
        with io.open(save_file, 'w', encoding = 'utf8') as sf:
            for token in self.data:
                sf.write(token + " " + self.data[token][0])



class GloveDataset(Dataset):
    def __init__(self, file_name = "glove.txt", max_rows = 1e5):
        self.max_rows = max_rows
        self.file_name = file_name
        self.data = []

        with io.open(file_name, 'r', encoding = 'utf8') as data_file:
            for line in data_file:
                self.data.append(line)
                if self.max_rows is not None and len(self.data) >= self.max_rows:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].split(" ")
        line = [line[0]] + [float(x) for x in line[1:]]
        return line

class CharacterEmbeddings():

    def __init__(self, file_name = "character_embedding_weights.txt"):
        with io.open(file_name, 'r') as f:
            character_to_index = {}
            index_to_character = {}
            entries = f.read().split()
            embedding_size  =300
            row_size = embedding_size + 1
            embedding = None
            for i in range(math.ceil(len(entries) / float(row_size))):
                character = entries[i * row_size]
                row = np.array(entries[i * row_size + 1 : i * row_size + row_size], dtype=float).reshape(1,-1)
                character_to_index[character] = i
                index_to_character[i] = character
                if embedding is None:
                    embedding = row
                else:
                    embedding = np.append(embedding, row, axis=0)
            self.embedding = embedding
            self.character_to_index = character_to_index
            self.index_to_character = index_to_character

    def to_indices(self, string):
        if len(string) > 20:
            return []
        return [self.character_to_index[x] for x in string if x in self.character_to_index]