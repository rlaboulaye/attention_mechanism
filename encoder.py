import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from rnn import RNN

class Encoder(nn.Module):

	# TODO: cuda

	def __init__(self, input_dimension=300, hidden_dimension=512, num_layers=1, batch_size=1):
		super(Encoder, self).__init__()
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.forward_rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers)
		self.backward_rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers)
		self.h_0 = Variable(torch.FloatTensor(np.zeros((self.num_layers, self.batch_size, self.hidden_dimension))))
		if torch.cuda.is_available():
			self.h_0.cuda()

	def forward(self, input_sequence, retain_sequence=False):
		forward_h_tm1 = self.h_0
		backward_h_tm1 = self.h_0
		sequence_length = len(input_sequence)
		embeddings = [[None, None]] * sequence_length
		for i in range(sequence_length):
			forward_input_embedding = input_sequence[i]
			backward_input_embedding = input_sequence[sequence_length - i - 1]
			forward_h_tm1 = self.forward_rnn(forward_input_embedding, forward_h_tm1)
			backward_h_tm1 = self.forward_rnn(backward_input_embedding, backward_h_tm1)
			embeddings[i][0] = forward_h_tm1[-1]
			embeddings[sequence_length - i - 1][1] = backward_h_tm1[-1]
		if retain_sequence:
			embeddings = [torch.cat((embedding[0], embedding[1]), dim=-1) for embedding in embeddings]
			return embeddings
		else:
			return torch.cat((embeddings[0][0], embeddings[-1][1]), dim=-1)
