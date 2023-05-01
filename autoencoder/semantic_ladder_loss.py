from torch.utils.data import Dataset
from autoencoder.ast_dataset import ASTDataset
from autoencoder.autoencoder_model import Encoder
from grammars.arithmetic_grammar import ArithmeticGrammar
from astlib.ast_util import ASTBuilder
from typing import List, Any, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import pickle



class SemanticLadderLossHelper:
    program_tensor_to_behavior_dist_matrix_idx: Dict[Tuple, int]
    behavior_dist_matrix_idx_to_program_tensor: Dict[int, Tuple]

    def __init__(self,  behavior_dist_matrix_file: str, behavior_vecs_file: str, behavior_dist_mat_idx_dicts_file: str, 
                 programs: ASTDataset, inputs: List[Any] | None, grammar: ArithmeticGrammar):
        self.grammar = grammar
        self.ast_builder = ASTBuilder(grammar)

        
        self.program_tensor_to_behavior_dist_matrix_idx = {}
        self.behavior_dist_matrix_idx_to_program_tensor = {}
        if os.path.isfile(behavior_vecs_file) and os.path.isfile(behavior_dist_mat_idx_dicts_file):
            self.behavior_vecs = np.load(behavior_vecs_file)
            with open(behavior_dist_mat_idx_dicts_file, 'rb') as f:
                dict_list = pickle.load(f)
            self.program_tensor_to_behavior_dist_matrix_idx = dict_list[0]
            self.behavior_dist_matrix_idx_to_program_tensor = dict_list[1]
        else:
            assert(inputs is not None)
            self.behavior_vecs = np.zeros((len(programs), len(inputs)))
            self._build_behavior_vecs(programs, inputs)
            np.save(behavior_vecs_file, self.behavior_vecs)
            dict_list = [self.program_tensor_to_behavior_dist_matrix_idx, self.behavior_dist_matrix_idx_to_program_tensor]
            with open(behavior_dist_mat_idx_dicts_file, 'wb+') as f:
                pickle.dump(dict_list, f)
        
        self.behavior_dist_matrix = np.zeros((len(programs), len(programs)))
        if os.path.isfile(behavior_dist_matrix_file):
            self.behavior_dist_matrix = np.load(behavior_dist_matrix_file)
        else:
            self._fill_behavior_dist_matrix()
            np.save(behavior_dist_matrix_file, self.behavior_dist_matrix)


    def _build_behavior_vecs(self, programs: ASTDataset, inputs: List[Any]):
        for program_i in tqdm(range(len(programs))):
            program = programs[program_i]

            # Add program to dict so it can be used to index into dist matrix later
            self.program_tensor_to_behavior_dist_matrix_idx[tuple(program.tolist())] = program_i
            self.behavior_dist_matrix_idx_to_program_tensor[program_i] = tuple(program.tolist())

            program_ast = self.ast_builder.tokenization_to_AST(program.tolist())
            for input_i, input in enumerate(inputs):
                program_ast.apply_func_to_ast(self.grammar.get_ast_apply_func_set_x_val(input))
                output = program_ast.evaluate()
                self.behavior_vecs[program_i, input_i] = output


    def _fill_behavior_dist_matrix(self):
        for i in tqdm(range(len(self.behavior_dist_matrix))):
            for j in range(i+1, len(self.behavior_dist_matrix)):
                behavior_vec_a = self.behavior_vecs[i]
                behavior_vec_b = self.behavior_vecs[j]
                behavior_dist = np.linalg.norm(behavior_vec_a - behavior_vec_b)

                self.behavior_dist_matrix[i,j] = behavior_dist
                self.behavior_dist_matrix[j,i] = behavior_dist

        # normalize_to_range(self.behavior_dist_matrix, new_min=0, new_max=1)
        

    # Returns a tuple, first item is a positive program tensor, second item is a list of negative ones, in decreasing
    # order of relevance, such that retval[1][0] is the most relevant negative program and retval[1][-1] the least relevant
    # anchor_program should be a tensor of shape [seq_len]
    def get_comparison_programs(self, anchor_program: torch.Tensor, ladder_levels: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        anchor_idx = self.program_tensor_to_behavior_dist_matrix_idx[tuple(anchor_program.tolist())]

        comparison_behavior_dists = self.behavior_dist_matrix[anchor_idx]

        # Get list of positive indices except for anchor if there are other options, sample a random one, and retrieve its corresponding program
        positive_indices = np.where(comparison_behavior_dists <= 1e-5)[0]
        if len(positive_indices) > 1:
            positive_indices = np.delete(positive_indices, np.where(positive_indices == anchor_idx))
        
        positive_idx = np.random.choice(positive_indices)
        positive_program = torch.tensor(self.behavior_dist_matrix_idx_to_program_tensor[positive_idx], device=device)
        
        # Choose ladder_levels negative programs, where the total space of negative programs is evenly divided into 
        # ladder_levels pieces and a negative program is sampled from each piece
        negative_programs: List[torch.Tensor] = []
        sampled_negative_indices = []
        negative_behavior_indices = np.where(comparison_behavior_dists >= 1e-5)[0]
        sorted_negative_behavior_indices = negative_behavior_indices[np.argsort(comparison_behavior_dists[negative_behavior_indices])]

        num_negative_behavior_indices = len(sorted_negative_behavior_indices)
        split_size = num_negative_behavior_indices // ladder_levels
        remainder = num_negative_behavior_indices % ladder_levels

        splits = []
        prev_end = 0
        for i in range(ladder_levels):
            end = prev_end + split_size + (1 if remainder > 0 else 0)
            splits.append(sorted_negative_behavior_indices[prev_end:end])
            prev_end = end
            remainder -= 1

        sampled_negative_indices = [np.random.choice(split) for split in splits]
        
        for sampled_negative_idx in sampled_negative_indices:
            negative_programs.append(torch.tensor(self.behavior_dist_matrix_idx_to_program_tensor[sampled_negative_idx], device=device))

        return positive_program, negative_programs



class SemanticLadderLoss(torch.nn.Module):
    def __init__(self, margins: List[float], ladder_levels: int, ladder_weights: List[float], helper: SemanticLadderLossHelper):
        super(SemanticLadderLoss, self).__init__()
        self.ladder_levels = ladder_levels
    
        assert(len(margins)) == ladder_levels
        assert(len(ladder_weights)) == ladder_levels

        self.margins = margins
        self.ladder_weights = ladder_weights
        self.helper = helper


    # Distance should be 0-1, with 0 being exact match and 1 being maximally far apart
    def _distance_calc(self, a: torch.Tensor, b: torch.Tensor):
        cos_sim = F.cosine_similarity(a, b)
        cos_sim_scaled = (cos_sim + 1) / 2
        return 1 - cos_sim_scaled
    

    def _triplet_loss(self, anchor, positive, negative, margin):
        positive_dist = self._distance_calc(anchor, positive)    
        negative_dist = self._distance_calc(anchor, negative)
        losses = torch.relu(positive_dist - negative_dist + margin)
        return losses


    # x is the programs that the autoencoder is trying to encode: [batch size, seq len]
    # anchor is the context vector output by the encoder when run on x: [1, batch size, hid dim]
    def forward(self, x: torch.Tensor, anchor: torch.Tensor, encoder: Encoder, device: torch.device, use_mu: bool = False):
        # Get batch_size comparison pairs from the helper
        #comparison_latent_reps = []
        batched_positives = []
        batched_negatives_list = []

        positive_programs = []
        negative_programs_list = []
        for i in range(x.shape[0]):
            current_program = x[i]
            positive_program, negative_programs = self.helper.get_comparison_programs(current_program, self.ladder_levels, device)

            positive_latent_rep, positive_mu, positive_log_var = encoder(positive_program)
            positive_mu = positive_mu.squeeze()
            positive_latent_rep = positive_latent_rep.squeeze()

            negative_latent_reps = []
            negative_mus = []
            for negative_program in negative_programs:
                negative_latent_rep, negative_mu, negative_log_var = encoder(negative_program)
                negative_mu = negative_mu.squeeze()
                negative_latent_rep = negative_latent_rep.squeeze()
                negative_latent_reps.append(negative_latent_rep)
                negative_mus.append(negative_mu)

            if use_mu:
                batched_positives.append(positive_mu)
                batched_negatives_list.append(negative_mus)
            else:
                batched_positives.append(positive_latent_rep)
                batched_negatives_list.append(negative_latent_reps)

            positive_programs.append(positive_program)
            negative_programs_list.append(negative_programs)

        batched_positives = torch.stack(batched_positives)
        batched_negatives_list = [torch.stack([sublist[i] for sublist in batched_negatives_list]) for i in range(len(batched_negatives_list[0]))]

        # First item in batched_negative_programs_list is tensor containing tensor representation of most relevant 
        # negative program for each batch item in x
        # batched_positive_programs = torch.stack(positive_programs)
        # batched_negative_programs_list = [torch.stack([sublist[i] for sublist in negative_programs_list]) for i in range(len(negative_programs_list[0]))]

        # with torch.no_grad():
        #     batched_negative_latent_reps_list = []
        #     batched_positive_latent_reps = encoder(batched_positive_programs).squeeze()
        #     for batched_negative_programs in batched_negative_programs_list:
        #         batched_negative_latent_reps_list.append(encoder(batched_negative_programs).squeeze())
        
        total_loss = 0
        for level in range(self.ladder_levels):
            if level == 0:
                positive = batched_positives
                negative = batched_negatives_list[0]
            else:
                positive = batched_negatives_list[level - 1]
                negative = batched_negatives_list[level]

            losses = self._triplet_loss(anchor.squeeze(dim=0), positive, negative, self.margins[level])
            total_loss += losses.mean() * self.ladder_weights[level]
        
        return total_loss
        