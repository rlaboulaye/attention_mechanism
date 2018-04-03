import numpy as np
import torch
from torch import nn

from Variable import Variable
from Initialization import initialize_weights


class AttentionMechanism(nn.Module):

	# TODO: add another fc; cuda

	def __init__(self, state_dimension, embedding_dimension):
		super(AttentionMechanism, self).__init__()
		self.fc = nn.Linear(state_dimension + embedding_dimension, 1)
		self.probability = nn.Softmax()

	def initialize_modules(self):
		for module in self.modules():
			module.apply(initialize_weights)

	def forward(self, state_tm1, embeddings):
		unnorm_alphas = []
		for embedding in embeddings:
			concatenated_embedding = torch.cat((state_tm1, embedding), dim=1)
			unnorm_alphas.append(self.fc(concatenated_embedding))
		alphas = self.probability(torch.cat([torch.unsqueeze(alpha, dim=0) for alpha in unnorm_alphas], dim=0))
		weighted_combination = None
		for i in range(len(embeddings)):
			weighted_embedding = alphas[i] * embeddings[i]
			if weighted_combination is None:
				weighted_combination = weighted_embedding
			else:
				weighted_combination += weighted_embedding
		return weighted_combination
