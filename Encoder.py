import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from RNN import RNN

class Encoder(nn.Module):

	def __init__(self, input_dimension=300, hidden_dimension=512, num_layers=1, batch_size=1):
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.forward_rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers)
		self.h0 = Variable(torch.FloatTensor(np.zeros((self.num_layers, self.batch_size, self.hidden_dimension))))

	def forward(self, input_sequence):
		h_tm1 = self.h0
		for input_embedding in input_sequence:
			h_tm1 = self.forward_rnn(input_embedding, h_tm1)
		return h_tm1[-1]
