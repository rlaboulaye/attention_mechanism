import numpy as np
import torch
from torch import nn

class GRUCell(nn.Module):

	def __init__(self, input_dimension, hidden_dimension, output_activation=nn.Tanh(), gating_activation=nn.Sigmoid()):
		super(GRUCell, self).__init__()
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.output_activation = output_activation
		self.gating_activation = gating_activation
		self.W_z = nn.Linear(input_dimension, hidden_dimension, bias=True)
		self.U_z = nn.Linear(hidden_dimension, hidden_dimension, bias=True)
		self.W_r = nn.Linear(input_dimension, hidden_dimension, bias=True)
		self.U_r = nn.Linear(hidden_dimension, hidden_dimension, bias=True)
		self.W_h = nn.Linear(input_dimension, hidden_dimension, bias=True)
		self.U_h = nn.Linear(hidden_dimension, hidden_dimension, bias=True)

	def initialize_parameters(self):
		for parameter in self.parameters:
			parameter.weight.data.normal_(0, np.sqrt(2. / (parameter.in_features + parameter.out_features)))
			parameter.bias.data.fill_(0.)

	def forward(self, x_t, h_tm1):
		z_t = self.gating_activation(self.W_z(x_t) + self.U_z(h_tm1))
		r_t = self.gating_activation(self.W_r(x_t) + self.U_r(h_tm1))
		h_t = (1 - z_t) * h_tm1 + z_t * self.output_activation(self.W_h(x_t) + self.U_h(r_t * h_tm1))
		return h_t
