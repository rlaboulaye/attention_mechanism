import torch
from torch import nn

from data_importer import SentenceTranslationDataset
from encoder import Encoder
from decoder import Decoder
from attention_based_decoder import AttentionBasedDecoder

class NeuralMachineTranslation(nn.Module):

	def __init__(self, vocab_size=, encoder_weights=None, decoder_weights=None):
		self.vocab_size = vocab_size
		if encoder_weights is None:
		else:
		if decoder_weights is None:
		else:

	def train(self, num_epochs=10, epoch_size=10, learning_rate=1e-5):


	def epoch(self, epoch_size, data_loader):
		self.encoder
		self.decoder

	def translate(self):