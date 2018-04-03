import torch
from torch import autograd

class Variable(autograd.Variable):

	def __init__(self, data, *args, **kwargs):
		if torch.cuda.is_available():
			data = data.cuda()
		super(Variable, self).__init__(data, *args, **kwargs)
