from cem.cem import cem_learn
from autoencoder.autoencoder_model import load_autoencoder, Autoencoder
from grammars.arithmetic_grammar import ArithmeticGrammar
from autoencoder.configs.standard_conf import standard_autoencoder_config
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker
import torch
from typing import List, Tuple, Callable
import inspect
from tqdm import tqdm



TOTAL_TRIES = 25

def measure_CEM_performance(models: List[Autoencoder], target_funcs:List[Callable[[int], int]]):
    for target_func in target_funcs:
        training_examples = generate_training_examples(target_func)
        for model_i, model in enumerate(models):
            print()
            target_func_str = inspect.getsource(target_func).split('x: ', 1)[1][:-2]
            print(f'Model {model_i}, {target_func_str}')
            
            found_tries = 0
            avg_iters_to_find = 0
            for i in range(TOTAL_TRIES):
                found, iters = cem_learn(n_iters=20, beam_width=1, sigma=1, popsize=100, n_elite=5, stop_loss=0, 
                                        training_examples=training_examples, grammar=grammar, model=model, 
                                        initial_program_tokenization=initial_program, latent_space_normalization=False)
                
                if found:
                    avg_iters_to_find += iters
                    found_tries += 1

            if found_tries != 0:
                avg_iters_to_find = avg_iters_to_find / found_tries

            print(f'Found in {found_tries} of the tries, took {avg_iters_to_find} iters to find on average\n')
        
        print('-------\n')
            



def generate_training_examples(target_func: Callable[[int], int]):
    training_examples: List[Tuple[int, int]] = []
    for i in range(100):
        training_examples.append((i, target_func(i)))
    return training_examples


MODEL_PATHS = ['autoencoder/models/best_model_1.pt', 'autoencoder/models/best_model_2.pt']

if __name__ == '__main__':
    grammar = ArithmeticGrammar()
    config = standard_autoencoder_config
    initial_program = [1, 7, 2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = ArithmeticSyntaxChecker(device, max_depth=2, inf_mask=True)

    config.syntax_checker = sc
    config.is_variational = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = [load_autoencoder(model_path, config, device) for model_path in MODEL_PATHS]

    target_funcs: List[Callable[[int], int]] = []
    target_funcs.append(lambda x: x + 1)
    target_funcs.append(lambda x: x + 5)
    target_funcs.append(lambda x: x + 9)

    target_funcs.append(lambda x: x - 1)
    target_funcs.append(lambda x: x - 5)
    target_funcs.append(lambda x: x - 9)

    target_funcs.append(lambda x: 2 * x)
    target_funcs.append(lambda x: 2 * x + 1)
    target_funcs.append(lambda x: 2 * x - 5)
    target_funcs.append(lambda x: 2 * x + 9)

    target_funcs.append(lambda x: 5 * x)
    # target_funcs.append(lambda x: 5 * x - 11)

    measure_CEM_performance(models, target_funcs)
  
    
    