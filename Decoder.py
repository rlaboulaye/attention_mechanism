import numpy as np
import torch
from torch import nn

from Variable import Variable
from RNN import RNN
from Initialization import initialize_weights


class Decoder(nn.Module):

	def __init__(self, input_dimension=512, hidden_dimension=1024, output_dimension=5000, num_layers=1, batch_size=1):
		super(Decoder, self).__init__()
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.output_dimension = output_dimension
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers)
		self.fc = nn.Linear(self.hidden_dimension, self.output_dimension)
		self.output_activation = nn.Softmax()
		self.x_0 = Variable(torch.FloatTensor(np.zeros((self.batch_size, self.hidden_dimension))))

	def initialize_modules(self):
		for module in self.modules():
			module.apply(initialize_weights)

	def _get_output(self, hidden_state, generate):
		logits = self.fc(hidden_state)
		if generate:
			return self.output_activation(logits)
		else:
			return logits

	def forward(self, sequence_embedding, generate=False):
		sequence = []
		x = self.x_0
		h_tm1 = sequence_embedding
		for input_embedding in input_sequence:
			h_tm1 = self.rnn(x, h_tm1)
			x = self._get_output(h_tm1[-1], generate)
			sequence.append(x)
		return sequence
