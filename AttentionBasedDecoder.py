import numpy as np
import torch
from torch import nn

from Decoder import Decoder
from AttentionMechanism import AttentionMechanism
from GRUCell import GRUCell
from ContextEnhancedGRUCellA import ContextEnhancedGRUCellA
from ContextEnhancedGRUCellB import ContextEnhancedGRUCellB
from Variable import get_variable


class AttentionBasedDecoder(Decoder):

	def __init__(self, input_dimension=300, hidden_dimension=1024, context_dimension=512, output_dimension=5000, num_layers=1, batch_size=1, max_sequence_length=50, bottom_time_cell=ContextEnhancedGRUCellA, stacked_time_cell=GRUCell):
		self.context_dimension = context_dimension
		super(AttentionBasedDecoder, self).__init__(input_dimension, hidden_dimension, output_dimension, num_layers, batch_size, max_sequence_length, bottom_time_cell, stacked_time_cell)
		self.attention = AttentionMechanism(hidden_dimension, context_dimension)

	def _get_initial_hidden_state(self, sequence_embedding):
		return get_variable(torch.FloatTensor(np.zeros((self.num_layers, self.batch_size, self.hidden_dimension))))

	def _get_hidden_state(self, x, h_tm1, sequence_embedding):
		context = self.attention(h_tm1[0], sequence_embedding)
		return self.rnn(x, h_tm1, context)
