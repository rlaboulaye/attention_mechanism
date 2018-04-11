from torch import nn

from gru_cell import GRUCell


class RNN(nn.Module):

	def __init__(self, input_dimension, hidden_dimension, num_layers, bottom_time_cell=GRUCell, stacked_time_cell=GRUCell, context_dimension=None):
		super(RNN, self).__init__()
		if num_layers < 1:
			raise ValueError('num_layers must be 1 or greater')
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.context_dimension = context_dimension
		self.layers = []
		for i in range(num_layers):
			if i == 0:
				if self.context_dimension is not None:
					self.layers.append(bottom_time_cell(self.input_dimension, self.hidden_dimension, self.context_dimension))
				else:
					self.layers.append(bottom_time_cell(self.input_dimension, self.hidden_dimension))
			else:
				self.layers.append(stacked_time_cell(self.hidden_dimension, self.hidden_dimension))
		if torch.cuda.is_available():
			for i in range(num_layers):
				self.layers[i] = self.layers[i].cuda()

	def forward(self, x_t, h_tm1, context=None):
		h_t = []
		for i, layer in enumerate(self.layers):
			if context is not None and i == 0:
				h_t.append(layer(x_t, h_tm1[i], context))
			else:
				h_t.append(layer(x_t, h_tm1[i]))
			x_t = h_t[i]
		return h_t
