import torch
from torch import nn
from torch.autograd import Variable

class AttentionMechanism(nn.Module):

	def __init__(self, state_dimension, embedding_dimension):
		self.fc = nn.Linear(state_dimension + embedding_dimension, 1)
		self.probability = nn.Softmax()

	def forward(self, state_tm1, embeddings):
		unnorm_alphas = []
		for embedding in embeddings:
			concatenated_embedding = torch.cat((state_tm1, embedding), dim=1)
			unnorm_alphas.append(self.fc(concatenated_embedding))
		alphas = self.probability(Variable(torch.FloatTensor(unnorm_alphas)))
		return alphas * embeddings
