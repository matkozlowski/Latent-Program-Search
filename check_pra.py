from metrics.perfect_reconstruction_acc import get_perfect_reconstruction_accuracy
from astlib.ast_util import ASTBuilder
from grammars.arithmetic_grammar import ArithmeticGrammar
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker
from autoencoder.autoencoder_model import load_autoencoder
from autoencoder.configs.standard_conf import standard_autoencoder_config
from autoencoder.ast_dataset import ASTDataset
import torch



MODEL_PATH = "autoencoder/models/best_model_3.pt"
TEST_PROGRAMS_FILE = "autoencoder/data/valid_tokenizations.txt"


def get_test_programs(device):
    dataset = ASTDataset(TEST_PROGRAMS_FILE, 0, device)
    programs = []
    for program_i in range(len(dataset)):
        program = dataset[program_i]
        programs.append(program)
    return programs

def main():
    grammar = ArithmeticGrammar()
    ast_builder = ASTBuilder(grammar)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = ArithmeticSyntaxChecker(device, max_depth=2, inf_mask=True)

    config = standard_autoencoder_config
    config.syntax_checker = sc
    config.is_variational = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_autoencoder(MODEL_PATH, config, device)

    get_perfect_reconstruction_accuracy(model, get_test_programs(device), beam_width=1)



if __name__ == '__main__':
    main()