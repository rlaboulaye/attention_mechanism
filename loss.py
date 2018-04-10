import numpy as np
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss

class SequenceLoss(nn.Module):

	def __init__(self, element_loss=CrossEntropyLoss):
		super(SequenceLoss, self).__init__()
		self.element_loss = element_loss()

	def forward(self, sequence_of_logits, sequence_of_targets):
		losses = []
		for logits, target in zip(sequence_of_logits, sequence_of_targets):
			losses.append(self.element_loss(logits, target))
		return np.mean(losses)
