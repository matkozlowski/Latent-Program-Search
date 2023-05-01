from autoencoder.autoencoder_model import SyntaxChecker
from typing import List
import torch


class ArithmeticSyntaxChecker(SyntaxChecker):
	batch_size: int
	empty: List[bool]
	open_count: List[int]
	is_left: List[bool]
	depth: List[int]

	max_depth: int
	vocab_size: int

	def __init__(self, device, max_depth, inf_mask: bool = False):
		super().__init__(device)

		self.max_depth = max_depth
		self.vocab_size = 11
		self.mask_val = float('-inf') if inf_mask else -100
		
		self.allow_terminals    = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.bool).to(self.device)
		self.allow_nonterminals = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool).to(self.device)
		self.allow_eos          = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool).to(self.device)

		self.reset_state(1)
	

	def reset_state(self, batch_size):
		self.batch_size = batch_size
		self.empty = [True] * batch_size
		self.open_count = [0] * batch_size
		self.is_left = [True] * batch_size
		self.depth = [0] * batch_size


	def get_syntax_mask(self):
		masks = torch.empty(self.batch_size, self.vocab_size).fill_(self.mask_val).to(self.device)
		mask_inf = torch.empty(self.vocab_size).fill_(self.mask_val).to(self.device)
		mask_zeros = torch.zeros(self.vocab_size).to(self.device)

		for i in range(self.batch_size):
			if self.empty[i]:
				masks[i] = torch.where(torch.logical_or(self.allow_terminals, self.allow_nonterminals), torch.zeros_like(mask_zeros), mask_inf)
			elif self.open_count[i] == 0:
				masks[i] = torch.where(self.allow_eos, torch.zeros_like(mask_zeros), mask_inf)
			elif self.depth[i] == self.max_depth:
				masks[i] = torch.where(self.allow_terminals, torch.zeros_like(mask_zeros), mask_inf)
			else:
				masks[i] = torch.where(torch.logical_or(self.allow_terminals, self.allow_nonterminals), torch.zeros_like(mask_zeros), mask_inf)

		return masks
	

	def update_state(self, tokens):
		#tokens = [batch size]

		for i in range(self.batch_size):
			token = tokens[i].item()
			if self.allow_terminals[token].item():
				self.empty[i] = False
				if self.is_left[i]:
					self.is_left[i] = False
				else:
					self.open_count[i] -= 1
					self.depth[i] -= 1
			elif self.allow_nonterminals[token].item():
				self.empty[i] = False
				if self.is_left[i]:
					self.open_count[i] += 1
					self.is_left[i] = True
				else:
					self.is_left[i] = True
				self.depth[i] += 1
				
		


if __name__ == '__main__':
	sc = ArithmeticSyntaxChecker('cpu', 2, True)
	sc.reset_state(1)
	sc.update_state(torch.tensor([6]))
	sc.update_state(torch.tensor([7]))
	sc.update_state(torch.tensor([6]))
	sc.update_state(torch.tensor([6]))

	print(sc.get_syntax_mask())