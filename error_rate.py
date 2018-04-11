import numpy as np
import torch
from torch import nn

class ErrorRate(nn.Module):

	def __init__(self):
		super(ErrorRate, self).__init__()
		self.activation = nn.Softmax()

	def forward(self, sequence_of_logits, sequence_of_targets):
		error_rates = []
		for logits, targets in zip(sequence_of_logits, sequence_of_targets):
			probabilities = self.activation(logits)
			indices = np.array([np.random.multinomial(1, .9999 * probability_distribution.cpu().data.numpy()).argmax() for probability_distribution in probabilities])
			targets = targets.cpu().data.numpy()
			errors = (indices != targets)[(targets != -1)]
			error_rates.append(errors.mean())
		return np.mean(error_rates)