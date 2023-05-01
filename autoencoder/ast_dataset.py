import torch
import ast
from torch.utils.data import Dataset, DataLoader, Sampler
import random

class ASTDataset(Dataset):
    def __init__(self, tokenizations_file, pad_token, device):
        self.pad_token = pad_token
        self.device = device

        self.tokenizations = []
        # max_tokenization_length = 0
        # with open(tokenizations_file, 'r') as tok_file:
        #     file_toks = tok_file.readlines()
        #     for tok_str in file_toks:
        #         tok_list = ast.literal_eval(tok_str)
        #         if len(tok_list) > max_tokenization_length:
        #             max_tokenization_length = len(tok_list)
        #         self.tokenizations.append(tok_list)

        
        # Need to add padding to ensure all token lists are the same length, necessary for batching
        # for tokenization in self.tokenizations:
        #     while len(tokenization) < max_tokenization_length:
        #         tokenization.append(self.pad_token)


        with open(tokenizations_file, 'r') as tok_file:
            file_toks = tok_file.readlines()
            for tok_str in file_toks:
                tok_list = ast.literal_eval(tok_str)
                self.tokenizations.append(tok_list)
        
    def __len__(self):
        return len(self.tokenizations)

    def __getitem__(self, index):
        return torch.tensor(self.tokenizations[index]).to(self.device)
    

class SameLengthBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices_by_length = {}

        for idx, item in enumerate(dataset):
            length = len(item)
            if length not in self.indices_by_length:
                self.indices_by_length[length] = []
            self.indices_by_length[length].append(idx)

    def __iter__(self):
        length_groups = list(self.indices_by_length.values())

        if self.shuffle:
            for indices in length_groups:
                random.shuffle(indices)  # Shuffle indices within each length group
            random.shuffle(length_groups)  # Shuffle the order of length groups

        for indices in length_groups:
            batch_indices = []
            for idx in indices:
                batch_indices.append(idx)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    batch_indices = []

            # Include remaining sequences in the last batch
            if batch_indices:
                yield batch_indices

    def __len__(self):
        total_batches = 0
        for indices in self.indices_by_length.values():
            full_batches, remaining_sequences = divmod(len(indices), self.batch_size)
            total_batches += full_batches
            if remaining_sequences > 0:
                total_batches += 1  # Account for the last smaller batch
        return total_batches