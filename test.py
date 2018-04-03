import numpy as np
import torch
from Variable import Variable

from Encoder import Encoder
from AttentionMechanism import AttentionMechanism

input_dimension = 3
hidden_dimension = 5
num_layers = 2
batch_size = 3

sequence_length = 6
input_sequence = []
for i in range(sequence_length):
	x = Variable(torch.FloatTensor(np.random.rand(batch_size, input_dimension)))
	input_sequence.append(x)

encoder = Encoder(input_dimension, hidden_dimension, num_layers, batch_size)
embeddings = encoder(input_sequence, True)

state = Variable(torch.FloatTensor(np.random.rand(batch_size, hidden_dimension)))
attention = AttentionMechanism(hidden_dimension, hidden_dimension * 2)
print(attention(state, embeddings))
