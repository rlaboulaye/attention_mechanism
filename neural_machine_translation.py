import time
import itertools

import numpy as np
import torch
from torch import nn

from data_importer import SentenceTranslationDataset
from encoder import Encoder
from decoder import Decoder
from attention_based_decoder import AttentionBasedDecoder
from loss import SequenceLoss
from error_rate import ErrorRate
from gru_cell import GRUCell
from context_enhanced_gru_cell_a import ContextEnhancedGRUCellA
from context_enhanced_gru_cell_b import ContextEnhancedGRUCellB
from variable import get_variable

class NeuralMachineTranslation(nn.Module):

    def __init__(
        self,
        data_loader,
        vocab_size,

        batch_size=16,

        n_encoder_layers=2,
        enc_input_dimension_size=300,
        enc_hidden_dimension_size=256,

        n_decoder_layers=2,
        dec_input_dimension_size=300,
        dec_hidden_dimension_size=512,

        encoder_weights=None,
        decoder_weights=None,

        use_attention_mechanism=False,

        bottom_time_cell=GRUCell,
        stacked_time_cell=GRUCell
    ):
        super(NeuralMachineTranslation, self).__init__()
        self.data_loader = data_loader
        if encoder_weights is None:
            self.encoder = Encoder(enc_input_dimension_size, enc_hidden_dimension_size, n_encoder_layers, batch_size)

        else:
            if torch.cuda.is_available():
                self.encoder = torch.load(encoder_weights)
            else:
                self.encoder = torch.load(encoder_weights, map_location=lambda storage, loc: storage)

        if decoder_weights is None:
            if use_attention_mechanism == True:
                self.decoder = AttentionBasedDecoder(
                    dec_input_dimension_size,
                    dec_hidden_dimension_size,
                    enc_hidden_dimension_size*2,
                    vocab_size,
                    n_decoder_layers,
                    batch_size,
                    bottom_time_cell=bottom_time_cell,
                    stacked_time_cell=stacked_time_cell
                )
            else:
                self.decoder = Decoder(
                    dec_input_dimension_size,
                    dec_hidden_dimension_size,
                    vocab_size,
                    n_decoder_layers,
                    batch_size,
                    bottom_time_cell=bottom_time_cell,
                    stacked_time_cell=stacked_time_cell
                )
        else:
            if torch.cuda.is_available():
                self.decoder = torch.load(decoder_weights)
            else:
                self.decoder = torch.load(decoder_weights, map_location=lambda storage, loc: storage)

        self.loss = SequenceLoss()
        self.error_rate = ErrorRate()

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.loss = self.loss.cuda()
            self.error_rate = self.error_rate.cuda()

        self.batch_size = batch_size
        self.use_attention_mechanism = use_attention_mechanism

    def train(self, num_epochs=10, epoch_size=10, learning_rate=1e-5, encoder_path='weights/encoder_weights', decoder_path='weights/decoder_weights', loss_path='losses/losses.npy', error_rate_path='losses/error_rates.npy'):
        optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=learning_rate)
        train_losses = []
        train_error_rates = []
        start_time = time.time()
        for e in xrange(num_epochs):
            print('Epoch {}'.format(e))
            train_loss, train_error_rate = self._epoch(epoch_size, optimizer)
            train_losses += train_loss
            train_error_rates += train_error_rate
            # test_losses += self._epoch(epoch_size) # todo
            print('Elapsed Time: {}'.format(time.time() - start_time))
            torch.save(self.encoder, encoder_path)
            torch.save(self.decoder, decoder_path)
            np.save(loss_path, np.array(train_losses))
            np.save(error_rate_path, np.array(train_error_rates))

    def _epoch(self, epoch_size, optimizer=None):
        src_seq_lens = self.data_loader.get_valid_src_seq_lens()
        src_seq_len_probs = src_seq_lens[:,1] / float(np.sum(src_seq_lens[:,1]))

        losses = []
        error_rates = []
        for i in xrange(epoch_size):
            src_seq_len_index = np.argmax(np.random.multinomial(1, .9999 * src_seq_len_probs))
            src_seq_len = src_seq_lens[src_seq_len_index, 0]
            batch_x_values, batch_y_values = self.data_loader.batch(src_seq_len, self.batch_size)
            targ_seq_len = batch_y_values.shape[0]
            batch_x_variables = [get_variable(torch.FloatTensor(x)) for x in batch_x_values]
            batch_y_variables = [get_variable(torch.LongTensor(y)) for y in batch_y_values]
            encoding = self.encoder(batch_x_variables, self.use_attention_mechanism)
            predictions, logits = self.decoder(
                encoding,
                self.data_loader.targ_encoding_2_embedding,
                self.data_loader.targ_word_2_encoding[self.data_loader.EOS_TOKEN],
                targ_seq_len
            )
            loss = self.loss(logits, batch_y_variables)
            losses.append(loss.cpu().data.numpy())
            error_rate = self.error_rate(np.array(predictions), batch_y_values)
            error_rates.append(error_rate)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Mean Loss: {}'.format(np.mean(losses)))
        print('Mean Error Rate: {}'.format(np.mean(error_rates)))
        return losses, error_rates

    def translate(self):
        # todo
        pass
