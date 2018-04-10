import numpy as np
import torch
from Variable import get_variable

from Encoder import Encoder
from Decoder import Decoder
from AttentionMechanism import AttentionMechanism
from AttentionBasedDecoder import AttentionBasedDecoder

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
	x = get_variable(torch.FloatTensor(batch_embeddings))
	input_embeddings.append(x)
input_embeddings.append(get_variable(torch.FloatTensor(np.concatenate([embedding_dict[eos_index].reshape(1, -1) for i in range(batch_size)]))))

encoder = Encoder(input_dimension, hidden_dimension, num_layers, batch_size)
sequence_embeddings = encoder(input_embeddings, True)

state = get_variable(torch.FloatTensor(np.random.rand(batch_size, hidden_dimension)))
attention = AttentionMechanism(hidden_dimension, hidden_dimension * 2)
print(attention(state, sequence_embeddings))

print('\nDecoder')
sequence_embedding = encoder(input_embeddings, False)
decoder = Decoder(input_dimension, hidden_dimension * 2, len(vocabulary), num_layers, batch_size)
output_sequence, logits = decoder(sequence_embedding, embedding_dict, eos_index, sequence_length)
print(output_sequence)
print(logits)

print('\nAttention Based Decoder')
sequence_embedding = encoder(input_embeddings, True)
attention_based_decoder = AttentionBasedDecoder(input_dimension, hidden_dimension, hidden_dimension * 2, len(vocabulary), num_layers, batch_size)
output_sequence, logits = attention_based_decoder(sequence_embedding, embedding_dict, eos_index, sequence_length)
print(output_sequence)
print(logits)
