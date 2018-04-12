from __future__ import division
import os
import io
from itertools import izip
import numpy as np
import pickle
from HTMLParser import HTMLParser

from torch.utils.data import Dataset

html_parser = HTMLParser()
def parse_word(word):
    word = word.strip().lower()
    word = html_parser.unescape(word)
    return word

def open_file(path, mode, encoding="utf-8"):
    return io.open(path, mode, encoding=encoding)

def vocab_intersect(
    lang_vocab_path = "./data/en-es/vocab.50K.es",
    embedding_vocab_path = "./data/fastText/wiki.es.vec.vocab",
    dest_path = "./processed_data/en-es/vocab.es"
):
    lang_word_index = {}
    with open_file(lang_word_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line != "":
                word = parse_word(line) # one word per line
                lang_word_index[word] = len(lang_word_index)

    vocab = [None] * len(lang_word_index)
    with open_file(embedding_vocab_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line != "":
                word = parse_word(line) # one word per line
                if word in lang_word_index:
                    index = lang_word_index[word]
                    vocab[index] = word

    with open_file(dest_path, "w") as f:
        for word in vocab:
            if word is not None:
                f.write(word + "\n")

def vocab_embeddings(
    vocab_path = "./processed_data/en-es/vocab.es",
    embedding_path = "./data/fastText/wiki.es.vec",
    dest_path = "./processed_data/en-es/embedding.vocab.es"
):
    vocab = []
    word_index = {}
    with open_file(vocab_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line != "":
                word = parse_word(line) # one word per line
                vocab.append(word)
                word_index[word] = len(word_index)

    embeddings = [None] * len(word_index)
    with open_file(embedding_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                embedding_size = line.strip().split(" ")[1]
            elif line != "": # first line as metadata and skip blank lines (last line)
                word, str_embedding = line.split(" ", 1)
                word = parse_word(word)
                index = word_index.get(word, None)
                if index is not None:
                    embeddings[index] = str_embedding

    with open_file(dest_path, "w") as f:
        f.write(u"{} {}\n".format(len(vocab), embedding_size))
        for word, embedding in izip(vocab, embeddings):
            if embedding is None:
                raise ValueError("invalid vocab at word {}".format(word))
            f.write(u"{} {}\n".format(word, embedding))

def split_data(
    src_lang_path="./data/en-es/train.en",
    targ_lang_path="./data/en-es/train.es",
    dest_src_lang_path="./processed_data/en-es/text.en",
    dest_targ_lang_path="./processed_data/en-es/text.es",
    split_ratio=.9
):
    with \
        open_file(src_lang_path, "r") as f_src,\
        open_file(targ_lang_path, "r") as f_targ,\
        open_file(dest_src_lang_path+".train", "w") as f_src_train,\
        open_file(dest_src_lang_path+".test", "w") as f_src_test,\
        open_file(dest_targ_lang_path+".train", "w") as f_targ_train,\
        open_file(dest_targ_lang_path+".test", "w") as f_targ_test\
    :
        for src_line, targ_line in izip(f_src, f_targ):
            if np.random.rand() < split_ratio:
                f_src_train.write(src_line)
                f_targ_train.write(targ_line)
            else:
                f_src_test.write(src_line)
                f_targ_test.write(targ_line)


class SentenceTranslationDataset(Dataset):
    # todo train/test split
    PAD = -1
    # default, english to spanish
    def __init__(
        self,
        src_lang_vocab_path="./processed_data/en-es/vocab.en",
        targ_lang_vocab_path="./processed_data/en-es/vocab.es",
        src_lang_embedding_path="./processed_data/en-es/embedding.vocab.en",
        targ_lang_embedding_path="./processed_data/en-es/embedding.vocab.es",
        src_lang_text_path="./processed_data/en-es/text.en.train",
        targ_lang_text_path="./processed_data/en-es/text.es.train",
        max_vocab_size=None,
        max_n_sentences=None,
        max_src_sentence_len=None,
        max_targ_sentence_len=None,
        eos_token="</s>"
    ):
    # # english to english
    # def __init__(
    #     self,
    #     src_lang_vocab_path="./processed_data/en-es/vocab.en",
    #     targ_lang_vocab_path="./processed_data/en-es/vocab.en",
    #     src_lang_embedding_path="./processed_data/en-es/embedding.vocab.en",
    #     targ_lang_embedding_path="./processed_data/en-es/embedding.vocab.en",
    #     src_lang_text_path="./data/en-es/train.en",
    #     targ_lang_text_path="./data/en-es/train.en",
    #     max_vocab_size=None,
    #     max_n_sentences=None,
    #     max_src_sentence_len=None,
    #     max_targ_sentence_len=None,
    #     eos_token="</s>"
    # ):
        if max_vocab_size is not None:
            max_vocab_size = int(max_vocab_size)
        if max_n_sentences is not None:
            max_n_sentences = int(max_n_sentences)
        if max_src_sentence_len is not None:
            max_src_sentence_len = int(max_src_sentence_len)
        if max_targ_sentence_len is not None:
            max_targ_sentence_len = int(max_targ_sentence_len)

        self.src_lang_vocab_path = src_lang_vocab_path
        self.targ_lang_vocab_path = targ_lang_vocab_path
        self.src_lang_embedding_path = src_lang_embedding_path
        self.targ_lang_embedding_path = targ_lang_embedding_path
        self.src_lang_text_path = src_lang_text_path
        self.targ_lang_text_path = targ_lang_text_path
        self.max_vocab_size = max_vocab_size
        self.max_n_sentences = max_n_sentences
        self.max_src_sentence_len = max_src_sentence_len
        self.max_targ_sentence_len = max_targ_sentence_len
        self.EOS_TOKEN = eos_token

        self.cache_dir = "./.cache/"
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        self._init_vocab()
        self._init_embedding()
        self._init_text()
        self._init_batching()

    def _init_vocab(self):
        self.src_vocab, self.src_word_2_encoding = self._read_vocab(self.src_lang_vocab_path)
        self.targ_vocab, self.targ_word_2_encoding = self._read_vocab(self.targ_lang_vocab_path)

    def _read_vocab(self, path):
        vocab = []
        word_2_encoding = {}
        with open_file(path, "r") as f:
            for i, line in enumerate(f):
                word = parse_word(line) # one word per line
                if word != "":
                    word_2_encoding[word] = len(vocab)
                    vocab.append(word)
                if self.max_vocab_size is not None and len(vocab) >= self.max_vocab_size:
                    break

        if self.EOS_TOKEN not in word_2_encoding:
            raise ValueError("EOS token not in vocabulary")

        return np.array(vocab), word_2_encoding

    def _get_src_emb_cache_path(self):
        hash_str = str(abs(hash(
            self.src_lang_embedding_path + self.src_lang_vocab_path
        )))
        return self.cache_dir + "emb_"  + str(self.max_vocab_size) + "_" + hash_str + ".p"

    def _get_targ_emb_cache_path(self):
        hash_str = str(abs(hash(
            self.targ_lang_embedding_path + self.targ_lang_vocab_path
        )))
        return self.cache_dir + "emb_"  + str(self.max_vocab_size) + "_" + hash_str + ".p"

    def _init_embedding(self):
        cache_src_emb_path = self._get_src_emb_cache_path()
        self.src_encoding_2_embedding = self._read_embedding(
            self.src_lang_embedding_path,
            self.src_vocab,
            self.src_word_2_encoding,
            cache_src_emb_path
        )

        cache_targ_emb_path = self._get_targ_emb_cache_path()
        self.targ_encoding_2_embedding = self._read_embedding(
            self.targ_lang_embedding_path,
            self.targ_vocab,
            self.targ_word_2_encoding,
            cache_targ_emb_path
        )

    def _read_embedding(self, path, vocab, word_2_encoding, cache_path=None):
        if cache_path is not None and os.path.isfile(cache_path):
            return pickle.load(open(cache_path, "rb"))

        encoding_2_embedding = None
        with open_file(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    self.embedding_size = int(line.strip().split(" ")[1])
                    encoding_2_embedding = np.empty((len(vocab), self.embedding_size)) * np.nan
                elif line != "":
                    word, str_embedding = line.strip().split(" ", 1)
                    word = parse_word(word)
                    encoding = word_2_encoding.get(word, None)
                    if encoding is not None:
                        embedding = np.array(str_embedding.split(" "), dtype=float)
                        encoding_2_embedding[encoding] = embedding

        self._validate_vocab(path, vocab, word_2_encoding, encoding_2_embedding)

        if cache_path is not None:
            pickle.dump(encoding_2_embedding, open(cache_path, "wb"))

        return encoding_2_embedding

    def _validate_vocab(self, path, vocab, word_2_encoding, encoding_2_embedding):
        if np.any(np.isnan(encoding_2_embedding[word_2_encoding[self.EOS_TOKEN]])):
            raise ValueError(path + " does not contain an end of sequence embedding")

        for encoding, embedding in enumerate(encoding_2_embedding):
            try:
                word = vocab[encoding]
                if word_2_encoding[word] != encoding:
                    raise Exception()
            except:
                raise ValueError("invalid vocab {} {}".format(word, encoding))

    def _get_src_text_cache_path(self):
        hash_str = str(abs(hash(
            self.src_lang_vocab_path + self.src_lang_embedding_path + self.src_lang_text_path
        )))
        return "{}text_v{}_n{}_l{}_{}.p".format(
            self.cache_dir,
            self.max_vocab_size,
            self.max_n_sentences,
            self.max_src_sentence_len,
            hash_str
        )

    def _get_targ_text_cache_path(self):
        hash_str = str(abs(hash(
            self.targ_lang_vocab_path + self.targ_lang_embedding_path + self.targ_lang_text_path
        )))
        return "{}text_v{}_n{}_l{}_{}.p".format(
            self.cache_dir,
            self.max_vocab_size,
            self.max_n_sentences,
            self.max_targ_sentence_len,
            hash_str
        )

    def _get_text_index_cache_path(self):
        hash_str = str(abs(hash(
            self.src_lang_vocab_path + self.src_lang_embedding_path + self.src_lang_text_path
        )))
        return "{}text_index_v{}_n{}_l{}_{}.p".format(
            self.cache_dir,
            self.max_vocab_size,
            self.max_n_sentences,
            self.max_src_sentence_len,
            hash_str
        )

    def _init_text(self):
        self.known_src_word_count = 0
        self.unknown_src_word_count = 0
        self.known_targ_word_count = 0
        self.unknown_targ_word_count = 0
        self.pruned_sentence_count = 0

        src_text_cache_path = self._get_src_text_cache_path()
        targ_text_cache_path = self._get_targ_text_cache_path()
        text_index_cache_path = self._get_text_index_cache_path()

        if os.path.isfile(src_text_cache_path) and os.path.isfile(targ_text_cache_path) and os.path.isfile(text_index_cache_path):
            self.src_data = pickle.load(open(src_text_cache_path, "rb"))
            self.targ_data = pickle.load(open(targ_text_cache_path, "rb"))
            self.src_data_by_seq_len_indices = pickle.load(open(text_index_cache_path, "rb"))
        else:
            self.src_data = []
            self.targ_data = []
            self.src_data_by_seq_len_indices = []
            with open_file(self.src_lang_text_path, "r") as f_src, open_file(self.targ_lang_text_path, "r") as f_targ:
                for i, (src_line, targ_line) in enumerate(izip(f_src, f_targ)):
                    src_sentence, unknown_src_word_count = self._parse_text_line(
                        src_line,
                        self.src_word_2_encoding,
                        self.src_encoding_2_embedding
                    )
                    self.known_src_word_count += len(src_sentence) - unknown_src_word_count
                    self.unknown_src_word_count += unknown_src_word_count

                    targ_sentence, unknown_targ_word_count = self._parse_text_line(
                        targ_line,
                        self.targ_word_2_encoding,
                        self.targ_encoding_2_embedding
                    )
                    self.known_targ_word_count += len(targ_sentence) - unknown_targ_word_count
                    self.unknown_targ_word_count += unknown_targ_word_count

                    if unknown_src_word_count > 0 or unknown_targ_word_count > 0:
                        self.pruned_sentence_count += 1
                    elif self.max_src_sentence_len is not None and len(src_sentence) > self.max_src_sentence_len:
                        self.pruned_sentence_count += 1
                    elif self.max_targ_sentence_len is not None and len(targ_sentence) > self.max_targ_sentence_len:
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

            pickle.dump(self.src_data, open(src_text_cache_path, "wb"))
            pickle.dump(self.targ_data, open(targ_text_cache_path, "wb"))
            pickle.dump(self.src_data_by_seq_len_indices, open(text_index_cache_path, "wb"))

    def _parse_text_line(self, line, word_2_encoding, encoding_2_embedding):
        unknown_word_count = 0
        sentence = []
        for word in line.strip().split(" "):
            word = parse_word(word)
            encoding = word_2_encoding.get(word, None)
            if encoding is None:
                unknown_word_count += 1
            elif np.any(np.isnan(encoding_2_embedding[encoding])):
                raise ValueError("encoding not in embedding")
            sentence.append(encoding)
        sentence.append(word_2_encoding[self.EOS_TOKEN])

        return sentence, unknown_word_count

    def _init_batching(self):
        self.batch_indices_start_indices = [0] * len(self.src_data_by_seq_len_indices)
        self.valid_src_sequence_lens = np.array([i for i, item in enumerate(self.src_data_by_seq_len_indices) if len(item) > 0])
        self.src_sequence_len_probs = self.valid_src_sequence_lens / float(self.valid_src_sequence_lens.sum())
        if self.src_sequence_len_probs.sum() > 1.0:
            self.src_sequence_len_probs *= .99999

    def get_valid_src_seq_lens(self):
        return self.valid_src_sequence_lens

    def get_src_seq_len_probs(self):
        return self.src_sequence_len_probs

    def sample_src_seq_len(self):
        src_seq_len_index = np.argmax(np.random.multinomial(1, self.src_sequence_len_probs))
        return self.valid_src_sequence_lens[src_seq_len_index]

    def _validate_batch_params(self, src_seq_len, batch_size, drop_last):
        if src_seq_len >= len(self.batch_indices_start_indices):
            raise ValueError("src_seq_len is too long")
        if drop_last == True and batch_size > len(self.src_data_by_seq_len_indices[src_seq_len]):
            raise ValueError("not enough data of sequence length {} for a batch of size {}".format(src_seq_len, batch_size))

    def get_n_batches(self, src_seq_len, batch_size, drop_last=False):
        try:
            self._validate_batch_params(src_seq_len, batch_size, drop_last)
        except ValueError:
            return 0

        n_data = len(self.src_data_by_seq_len_indices[src_seq_len])
        n_batches = n_data // batch_size
        if drop_last == False and batch_size * n_batches < n_data:
            n_batches += 1
        return n_batches

    def batch(self, src_seq_len, batch_size, drop_last=True, shuffle=True):
        self._validate_batch_params(src_seq_len, batch_size, drop_last)

        batch_indices = self._get_batch_indices(src_seq_len, batch_size, drop_last, shuffle)
        batch_src_data = self._src_batch(batch_indices)
        batch_targ_data = self._targ_batch(batch_indices)

        return batch_src_data, batch_targ_data

    def _get_batch_indices(self, src_seq_len, batch_size, drop_last, shuffle):
        # get index indices - yes double indexing
        batch_indices_start_index = self.batch_indices_start_indices[src_seq_len]
        batch_indices_end_index = batch_indices_start_index + batch_size
        if drop_last == True and batch_indices_end_index > len(self.src_data_by_seq_len_indices[src_seq_len]):
            batch_indices_start_index = 0
            batch_indices_end_index = batch_size
        else:
            batch_indices_end_index = min(batch_indices_end_index, len(self.src_data_by_seq_len_indices[src_seq_len]))

        # shuffle if needed
        if batch_indices_start_index == 0 and shuffle == True:
            self._shuffle_src_data_indices(src_seq_len)

        self.batch_indices_start_indices[src_seq_len] += batch_size
        # update index indices
        if batch_indices_start_index == 0:
            self.batch_indices_start_indices[src_seq_len] = batch_size
        if batch_indices_end_index == len(self.src_data_by_seq_len_indices[src_seq_len]):
            self.batch_indices_start_indices[src_seq_len] = 0

        return self.src_data_by_seq_len_indices[src_seq_len][batch_indices_start_index:batch_indices_end_index]

    def _shuffle_src_data_indices(self, src_seq_len):
        np.random.shuffle(self.src_data_by_seq_len_indices[src_seq_len])

    def _src_batch(self, batch_indices):
        batch_src_data = np.array([self.src_data[i] for i in batch_indices])
        batch_src_data = np.swapaxes(batch_src_data, 0, 1)
        return batch_src_data

    def embed_src_batch(self, batch_src_data):
        return np.array([self.src_encoding_2_embedding[batch_t] for batch_t in batch_src_data])

    def _targ_batch(self, batch_indices):
        # get max targ seq len
        jagged_batch_targ_data = []
        batch_max_targ_sentence_len = 0
        for i in batch_indices:
            jagged_batch_targ_data.append(self.targ_data[i])
            batch_max_targ_sentence_len = max(batch_max_targ_sentence_len, len(self.targ_data[i]))

        # pad targ batch to max seq len and encode
        batch_targ_data = []
        for targ_data in jagged_batch_targ_data:
            pad_width = (0, batch_max_targ_sentence_len - len(targ_data))
            if pad_width != (0,0):
                targ_data = np.pad(targ_data, pad_width, mode="constant", constant_values=self.PAD)
            batch_targ_data.append(targ_data)

        batch_targ_data = np.swapaxes(batch_targ_data, 0, 1)
        return batch_targ_data

    def decode_src_batch(self, batch_src_data):
        batch_src_data = np.swapaxes(batch_src_data, 0, 1)
        return [self.src_vocab[sentence] for sentence in batch_src_data]

    def decode_targ_batch(self, batch_targ_data):
        batch_targ_data = np.swapaxes(batch_targ_data, 0, 1)
        return np.array([self.targ_vocab[batch_t[batch_t != self.PAD]] for batch_t in batch_targ_data])


if __name__ == '__main__':
    dataset = SentenceTranslationDataset(
        max_vocab_size=3e3,
        max_n_sentences=1e6,
        max_src_sentence_len=30,
        max_targ_sentence_len=30
    )
    try:
        print dataset.unknown_src_word_count / dataset.known_src_word_count
        print dataset.unknown_targ_word_count / dataset.known_targ_word_count
        print dataset.pruned_sentence_count
    except:
        pass
    print len(dataset.src_data), len(dataset.targ_data)

    for seq_len in dataset.get_valid_src_seq_lens():
        print seq_len
        for _ in xrange(100):
            batch_x, batch_y = dataset.batch(seq_len, 100, True, True)
        # print dataset.embed_src_batch(batch_x)
        print dataset.decode_src_batch(batch_x)[0]
        print dataset.decode_targ_batch(batch_y)[0]
        print "passed drop last"
        for _ in xrange(100):
            batch = dataset.batch(seq_len, 100, False, True)
        print "passed include last"

    # print dataset.get_valid_src_seq_lens()

    # for i, item in enumerate(dataset.src_data_by_seq_len_indices):
    #     print i, len(item), dataset.get_n_batches(i, 32, True), dataset.get_n_batches(i, 32, False)
