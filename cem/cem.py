import torch
import torch.nn.functional as F
from typing import List, Tuple
from grammars.arithmetic_grammar import ArithmeticGrammar
from astlib.ast_util import ASTBuilder
from autoencoder.autoencoder_model import SyntaxChecker, Autoencoder, load_autoencoder
import pickle

PICKLE_FILE = 'cem/latent_reps.pkl'

def calc_error(observed, target):
    return abs(observed - target)

def normalize_to_range(tensor, new_min=-1, new_max=1):
    old_min = torch.min(tensor)
    old_max = torch.max(tensor)
    old_range = old_max - old_min
    new_range = new_max - new_min

    # Normalize the numbers to the new range while preserving their relative ratios
    normalized_tensor = (tensor - old_min) / old_range * new_range + new_min
    return normalized_tensor

# Adapted from this CEM implementation: https://github.com/clvrai/leaps/blob/a313aa14e45868f30cae2c1027d2f814e8a3bba3/pretrain/CEM.py
@torch.no_grad()
def cem_learn(n_iters: int, beam_width: int, sigma: float, popsize: int, n_elite: int, stop_loss: float,
              training_examples: List[Tuple[int, int]], grammar: ArithmeticGrammar, model: Autoencoder, 
              initial_program_tokenization: List[int], latent_space_normalization=False,):
    
    ast_builder = ASTBuilder(grammar)

    model.eval()

    initial_program = torch.tensor([initial_program_tokenization]).to(model.device)
    curr_program_latent_rep, _, _ = model.encoder(initial_program)
    curr_program_latent_rep = curr_program_latent_rep.squeeze()
    curr_sigma = sigma

    mean_latent_reps = []
    mean_latent_reps.append(curr_program_latent_rep.tolist())

    for cem_iter in range(n_iters):
        # Draw popsize samples from normal distribution centered around current latent rep, with std dev of sigma
        new_population = [curr_program_latent_rep + (curr_sigma * torch.randn_like(curr_program_latent_rep)) for _ in range(popsize)]

        # Get losses for each member in the population
        pop_losses_and_latent_reps = [(0, torch.tensor([0]))] * popsize
        for i, member_latent_rep in enumerate(new_population):
            member_program = model.decoder.beam_search(member_latent_rep, beam_width=1)
            member_ast = ast_builder.tokenization_to_AST(member_program)
            member_loss = 0
            for x, y in training_examples:
                member_ast.apply_func_to_ast(grammar.get_ast_apply_func_set_x_val(x))
                member_program_out = member_ast.evaluate()
                member_loss += calc_error(member_program_out, y)
            pop_losses_and_latent_reps[i] = (member_loss, member_latent_rep)
        
        # Sort the population members by loss and choose the n_elite of them with lowest loss to be the elite
        pop_losses_and_latent_reps_sorted = sorted(pop_losses_and_latent_reps, key=lambda x: x[0])
        elite_members = torch.stack([loss_and_lr[1] for loss_and_lr in pop_losses_and_latent_reps_sorted[:n_elite]])
        elite_losses = torch.stack([torch.tensor(loss_and_lr[0], dtype=torch.float32, device=model.device) for loss_and_lr in pop_losses_and_latent_reps_sorted[:n_elite]])


        # Take the weighted mean of the elite latent reps to get the new latent rep
        # Elite members with lower loss will be weighted more heavily
        weights = F.softmax(-elite_losses, dim=0)
        curr_sigma = torch.sqrt(torch.matmul(weights.T, (elite_members - curr_program_latent_rep).pow(2)).squeeze())
        curr_program_latent_rep = torch.matmul(weights.T, elite_members).squeeze()





        mean_latent_reps.append(curr_program_latent_rep.tolist())


        # Change the new latent rep into a program to determine the new loss
        curr_program = model.decoder.beam_search(curr_program_latent_rep, beam_width=beam_width)
        curr_ast = ast_builder.tokenization_to_AST(curr_program)
        curr_loss = 0
        for x,y in training_examples:
            curr_ast.apply_func_to_ast(grammar.get_ast_apply_func_set_x_val(x))
            out = curr_ast.evaluate()
            curr_loss += calc_error(out, y)
        print('Loss:', curr_loss, '\n', curr_ast.pretty_print() + '\n\n')

        # Place the latent representation in the -1 to 1 range to keep it in the a range the decoder is familiar with
        if latent_space_normalization:
            curr_program_latent_rep, _, _ = model.encoder(torch.tensor(curr_program, device=model.device))
            curr_program_latent_rep = curr_program_latent_rep.squeeze()


        if curr_loss <= stop_loss:
            print(f'Found program that meets loss requirement after {cem_iter + 1} iterations!')
            with open(PICKLE_FILE, 'wb+') as f:
                pickle.dump(mean_latent_reps, f)
            return True, cem_iter + 1
            break

    with open(PICKLE_FILE, 'wb+') as f:
        pickle.dump(mean_latent_reps, f)

    return False, n_iters

    