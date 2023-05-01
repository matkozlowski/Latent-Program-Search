from autoencoder.semantic_ladder_loss import SemanticLadderLossHelper
from autoencoder.ast_dataset import ASTDataset
from grammars.arithmetic_grammar import ArithmeticGrammar
from astlib.ast_util import ASTBuilder
import torch


dataset = ASTDataset('autoencoder/data/train_tokenizations.txt', 0, "cpu")
inputs = [i for i in range(1, 101)]
grammar = ArithmeticGrammar()

sllh = SemanticLadderLossHelper('autoencoder/data/semantic_loss_data/train_behavior_dist_mat.npy', 
                                'autoencoder/data/semantic_loss_data/train_behavior_vecs.npy', 
                                'autoencoder/data/semantic_loss_data/train_behavior_dist_mat_idx.pkl',
                                dataset, inputs, grammar, ladder_levels=10)
positive, negatives = sllh.get_comparison_programs(torch.tensor([1, 3, 9, 6, 2]))

astbuilder = ASTBuilder(grammar)

print(positive)
print(negatives)
print(astbuilder.tokenization_to_AST(positive.tolist()).pretty_print(), '\n\n')

for neg in negatives:
    print(astbuilder.tokenization_to_AST(neg.tolist()).pretty_print(), '\n\n')
