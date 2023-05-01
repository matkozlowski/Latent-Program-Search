from autoencoder.train_autoencoder import train
from autoencoder.configs.standard_conf import standard_autoencoder_config
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker

def main():
    syntax_checker = ArithmeticSyntaxChecker("cuda", max_depth=2, inf_mask=False)
    # syntax_checker=None
    train(n_epochs=40, save_path="autoencoder/models/best_model.pt", model_config=standard_autoencoder_config, 
          train_data_path="autoencoder/data/train_tokenizations.txt", valid_data_path="autoencoder/data/valid_tokenizations.txt", 
          batch_size=32, syntax_checker=syntax_checker, use_semantic_ladder_loss=True, use_variational=False)

if __name__ == '__main__':
    main()