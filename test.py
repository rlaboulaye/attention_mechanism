import numpy as np
import torch
from Variable import Variable

from Encoder import Encoder
from Decoder import Decoder
from AttentionMechanism import AttentionMechanism

input_dimension = 8
hidden_dimension = 5
num_layers = 2
batch_size = 3

embedding_dict = {}
vocabulary = ['a', 'b', 'c', 'd', 'eos']
eos_index = vocabulary.index('eos')
for index in range(len(vocabulary)):
	embedding_dict[index] = np.random.rand(input_dimension)

sequence_length = 6
input_embeddings = []
for i in range(sequence_length):
	batch_words = np.random.choice(range(len(vocabulary)), batch_size)
	batch_embeddings = np.concatenate([embedding_dict[word].reshape(1,-1) for word in batch_words])
	x = Variable(torch.FloatTensor(batch_embeddings))
	input_embeddings.append(x)
input_embeddings.append(Variable(torch.FloatTensor(np.concatenate([embedding_dict[eos_index].reshape(1, -1) for i in range(batch_size)]))))

encoder = Encoder(input_dimension, hidden_dimension, num_layers, batch_size)
sequence_embeddings = encoder(input_embeddings, True)

state = Variable(torch.FloatTensor(np.random.rand(batch_size, hidden_dimension)))
attention = AttentionMechanism(hidden_dimension, hidden_dimension * 2)
print(attention(state, sequence_embeddings))

sequence_embedding = encoder(input_embeddings, False)
print(sequence_embedding.size())
decoder = Decoder(input_dimension, hidden_dimension * 2, len(vocabulary), num_layers, batch_size)
output_sequence, logits = decoder(sequence_embedding, embedding_dict, eos_index, sequence_length)
print(output_sequence)
print(logits)
