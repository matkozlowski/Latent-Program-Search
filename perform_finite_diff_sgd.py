from finite_differences.finite_diff import finite_differences_sgd
from grammars.arithmetic_grammar import ArithmeticGrammar
from autoencoder.configs.standard_conf import standard_autoencoder_config
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker
import torch
from typing import List, Tuple

# x + x
def target_func(x):
    return x + 5


def generate_training_examples():
    training_examples: List[Tuple[int, int]] = []
    for i in range(100):
        training_examples.append((i, target_func(i)))
    return training_examples


def main():
    training_examples = generate_training_examples()
    grammar = ArithmeticGrammar()
    config = standard_autoencoder_config
    initial_program = [1, 7, 2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = ArithmeticSyntaxChecker(device, max_depth=2, inf_mask=True)

    finite_differences_sgd(n_epochs=10, batch_size=16, nudge_tries=50, beam_width=1, learning_rate=0.00005, 
                           training_examples=training_examples, grammar=grammar, config=config, 
                           model_path='autoencoder/models/best_model_3.pt', initial_program_tokenization=initial_program, 
                           syntax_checker=sc)

if __name__ == '__main__':
    main()