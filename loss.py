import numpy as np
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss

class SequenceLoss(nn.Module):

	def __init__(self, element_loss=CrossEntropyLoss):
		super(SequenceLoss, self).__init__()
		self.element_loss = element_loss(reduce=False)

	def forward(self, sequence_of_logits, sequence_of_targets):
		losses = []
		for logits, targets in zip(sequence_of_logits, sequence_of_targets):
			mask = targets != -1
			batch_loss = self.element_loss(logits, targets)
			masked_batch_loss = mask.float() * batch_loss
			losses.append(masked_batch_loss.mean())
		return np.mean(losses)
