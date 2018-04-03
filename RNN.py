from torch import nn

from GRUCell import GRUCell


class RNN(nn.Module):

	def __init__(self, input_dimension, hidden_dimension, num_layers, time_cell=GRUCell):
		super(RNN, self).__init__()
		if num_layers < 1:
			raise ValueError('num_layers must be 1 or greater')
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.layers = []
		for i in range(num_layers):
			if i == 0:
				input_dimension = self.input_dimension
			else:
				input_dimension = self.hidden_dimension
			self.layers.append(time_cell(input_dimension, self.hidden_dimension))

	def forward(self, x_t, h_tm1):
		h_t = []
		for i, layer in enumerate(self.layers):
			h_t.append(layer(x_t, h_tm1[i]))
			x_t = h_t[i]
		return h_t
