from astlib.ast_util import AST, ASTBuilder
from astlib.grammar import Grammar
from grammars.arithmetic_grammar import ArithmeticGrammar
from autoencoder.autoencoder_model import load_autoencoder, SyntaxChecker, AutoencoderConfig
from autoencoder.configs.standard_conf import standard_autoencoder_config
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker
import torch
from tqdm import tqdm
from typing import List, Tuple
import random



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

@torch.no_grad()
def finite_differences_sgd(n_epochs: int, batch_size: int, nudge_tries: int, beam_width: int, learning_rate: float, 
                           training_examples: List[Tuple[int, int]], grammar: ArithmeticGrammar, config: AutoencoderConfig, 
                           model_path: str, initial_program_tokenization: List[int],
                           syntax_checker: SyntaxChecker | None = None,):

    ast_builder = ASTBuilder(grammar)

    config.syntax_checker = syntax_checker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_autoencoder(model_path, config, device)
    model.eval()

    initial_program = torch.tensor([initial_program_tokenization]).to(device)
    curr_program_latent_rep, _, _ = model.encoder(initial_program)
    curr_program_latent_rep = curr_program_latent_rep.squeeze()


    num_examples = len(training_examples)

    last_program = None
    # Each entry in nudge_programs will be a tuple with the following elements:
    # tokenization of the program, amount the program was nudged by
    nudged_programs = []

    for epoch_i in range(n_epochs):
        print(f"\n\n\nEPOCH {epoch_i}")
        epoch_loss = 0
        
        # Shuffle the training examples to add stochasticity
        shuffled_training_examples = training_examples.copy()
        random.shuffle(shuffled_training_examples)

        # Perform an epoch of SGD
        print('\tGoing through training batches...')
        for i in range(0, num_examples, batch_size):
            # Create current program AST from latent representation
            curr_program = model.decoder.beam_search(curr_program_latent_rep, beam_width=beam_width)
            curr_ast = ast_builder.tokenization_to_AST(curr_program)
            # print('AST: \n' + curr_ast.pretty_print())
            # print(curr_program_latent_rep)

            
            # Create nudged programs by modifying each dimension of the current latent representation until a new program is found
            if last_program is None or curr_program != last_program:
                nudged_programs = []
                print('\tNudging latent representation...')
                for nudge_idx in tqdm(range(config.latent_rep_dim)):
                    nudged_latent_rep = curr_program_latent_rep.clone()

                    found_nudged_program = False
                    for i in range(nudge_tries):
                        nudged_latent_rep[nudge_idx] += 1
                        nudged_program = model.decoder.beam_search(nudged_latent_rep, beam_width=beam_width)
                        if nudged_program != curr_program:
                            nudged_programs.append((nudged_program, i + 1))
                            found_nudged_program = True
                            break

                    if not found_nudged_program:
                        nudged_programs.append((curr_program, 0))

            # Select the next mini-batch from the shuffled training examples
            batch_examples = shuffled_training_examples[i:i + batch_size]
            batch_epoch_loss = 0
            batch_partial_derivatives = [0] * len(curr_program_latent_rep)

            # Iterate over training examples in batch, calculating partial derivatives
            for x, y in batch_examples:
                # Calculate error of current program. Involves setting x value in AST and evaluating it
                curr_ast.apply_func_to_ast(grammar.get_ast_apply_func_set_x_val(x))
                curr_program_output = curr_ast.evaluate()
                curr_program_error = calc_error(curr_program_output, y)
                batch_epoch_loss += curr_program_error

                # Calculate Error of each nudged program to get each partial derivative, which gives an overall gradient
                for j, nudged_program in enumerate(nudged_programs):
                    if nudged_program[1] == 0:
                        batch_partial_derivatives[j] += 0
                    else:
                        nudged_program_ast = ast_builder.tokenization_to_AST(nudged_program[0])
                        nudged_program_ast.apply_func_to_ast(grammar.get_ast_apply_func_set_x_val(x))
                        nudged_program_output = nudged_program_ast.evaluate()
                        nudged_program_error = calc_error(nudged_program_output, y)
                        delta_error = nudged_program_error - curr_program_error
                        delta_latent_rep = nudged_program[1]
                        batch_partial_derivatives[j] += delta_error / delta_latent_rep

            # Average the batch epoch loss and partial derivatives
            batch_epoch_loss /= len(batch_examples)
            print(f'\tLoss: {batch_epoch_loss}')
            epoch_loss += batch_epoch_loss
            avgeraged_partial_derivatives = [partial / len(batch_examples) for partial in batch_partial_derivatives]

            # Take gradient descent step
            for i, partial in enumerate(avgeraged_partial_derivatives):
                curr_program_latent_rep[i] -= partial * learning_rate

            # Place the latent representation in the -1 to 1 range to keep it in the a range the decoder is familiar with
            #curr_program_latent_rep = normalize_to_range(curr_program_latent_rep)

            last_program = curr_program.copy()
        
        print(curr_ast.pretty_print())
        print(f'\t Epoch Loss: {epoch_loss / len(training_examples)}')

    final_program = model.decoder.beam_search(curr_program_latent_rep, beam_width=beam_width)
    final_ast = ast_builder.tokenization_to_AST(final_program)
    print(final_ast.pretty_print())

    
        

        