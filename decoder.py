import numpy as np
import torch
from torch import nn

from rnn import RNN
from variable import get_variable
from gru_cell import GRUCell
from initialization import initialize_weights


class Decoder(nn.Module):

	def __init__(self, input_dimension=512, hidden_dimension=1024, output_dimension=5000, num_layers=1, batch_size=1, max_sequence_length=50, bottom_time_cell=GRUCell, stacked_time_cell=GRUCell):
		super(Decoder, self).__init__()
		self.input_dimension = int(input_dimension)
		self.hidden_dimension = int(hidden_dimension)
		self.output_dimension = int(output_dimension)
		self.num_layers = int(num_layers)
		self.batch_size = int(batch_size)
		if hasattr(self, 'context_dimension'):
			self.rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers, bottom_time_cell, stacked_time_cell, self.context_dimension)
		else:
			self.rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers, bottom_time_cell, stacked_time_cell)
		self.fc = nn.Linear(self.hidden_dimension, self.output_dimension)
		self.output_activation = nn.Softmax()
		self.max_sequence_length = max_sequence_length
		self.initialize_modules()

	def initialize_modules(self):
		for module in self.modules():
			module.apply(initialize_weights)

	def _get_initial_hidden_state(self, sequence_embedding):
		return torch.cat([sequence_embedding.unsqueeze(0)] * self.num_layers)

	def _get_hidden_state(self, x, h_tm1, sequence_embedding):
		return self.rnn(x, h_tm1)

	def _get_output(self, hidden_state):
		logits = self.fc(hidden_state)
		probabilities = self.output_activation(logits)
		indices = [np.random.multinomial(1, probability_distribution.cpu().data.numpy()).argmax() for probability_distribution in probabilities]
		return indices, logits

	def forward(self, sequence_embedding, embedding_dict, eos_index, training_sequence_length=None):
		sequence_of_indices = []
		sequence_of_logits = []
		x = get_variable(torch.FloatTensor([embedding_dict[eos_index]] * self.batch_size))
		h_tm1 = self._get_initial_hidden_state(sequence_embedding)
		word_indices = [-1] * self.batch_size
		while (training_sequence_length is None and np.any(np.array(word_indices) != eos_index) and len(sequence_of_indices) < self.max_sequence_length) or (training_sequence_length is not None and len(sequence_of_indices) < training_sequence_length):
			h_tm1 = self._get_hidden_state(x, h_tm1, sequence_embedding)
			word_indices, logits = self._get_output(h_tm1[-1])
			sequence_of_indices.append(word_indices)
			sequence_of_logits.append(logits)
			x = get_variable(torch.FloatTensor([embedding_dict[word_index] for word_index in word_indices]))
		return sequence_of_indices, sequence_of_logits
