from cem.cem import cem_learn
from autoencoder.autoencoder_model import load_autoencoder
from grammars.arithmetic_grammar import ArithmeticGrammar
from autoencoder.configs.standard_conf import standard_autoencoder_config
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker
import torch
from typing import List, Tuple


def target_func(x):
    return 12 * x


def generate_training_examples():
    training_examples: List[Tuple[int, int]] = []
    for i in range(100):
        training_examples.append((i, target_func(i)))
    return training_examples

MODEL_PATH = 'autoencoder/models/best_model_2.pt'

def main():
    training_examples = generate_training_examples()
    grammar = ArithmeticGrammar()
    config = standard_autoencoder_config
    initial_program = [1, 6, 2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = ArithmeticSyntaxChecker(device, max_depth=2, inf_mask=True)

    config.syntax_checker = sc
    config.is_variational = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_autoencoder(MODEL_PATH, config, device)

    cem_learn(n_iters=20, beam_width=1, sigma=1, popsize=100, n_elite=5, stop_loss=0, 
              training_examples=training_examples, grammar=grammar, model=model, 
              initial_program_tokenization=initial_program, latent_space_normalization=False)

if __name__ == '__main__':
    main()