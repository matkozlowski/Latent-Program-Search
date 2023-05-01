import numpy as np
import pickle
from grammars.arithmetic_grammar import ArithmeticGrammar
from astlib.ast_util import ASTBuilder

EPSILON = 1e-5
#EPSILON = 100
#EPSILON = 1000

def find_equivalence_classes(behavior_dist_matrix_file, behavior_dist_mat_idx_dicts_file, equivalence_class_file):
    behavior_dist_matrix = np.load(behavior_dist_matrix_file)
    with open(behavior_dist_mat_idx_dicts_file, 'rb') as f:
        dict_list = pickle.load(f)
        behavior_dist_matrix_idx_to_program_tensor = dict_list[1]
    
    indices_to_check = [i for i in range(len(behavior_dist_matrix))]

    equivalence_classes = []
    curr_idx = indices_to_check[0]
    while len(indices_to_check) > 0:
        
        comparison_behavior_dists = behavior_dist_matrix[curr_idx]

        equivalent_indices = np.where(comparison_behavior_dists <= EPSILON)[0]

        equivalence_class = []
        for eq_idx in equivalent_indices:
            equivalence_class.append(behavior_dist_matrix_idx_to_program_tensor[eq_idx])
        equivalence_classes.append(equivalence_class)

        indices_to_check = [idx for idx in indices_to_check if idx not in equivalent_indices]

        if len(indices_to_check) > 0:
            curr_idx = indices_to_check.pop(0)

    equivalence_classes = sorted(equivalence_classes, key=lambda x: len(x), reverse=True)
    print(len(equivalence_classes))
    print(equivalence_classes[0][0])
    print(equivalence_classes[1][0])
    print(equivalence_classes[2][0])
    print(equivalence_classes[1:5])


    behavior_funcs = []
    behavior_funcs.append(lambda x: 0)
    behavior_funcs.append(lambda x: x)
    behavior_funcs.append(lambda x: x + 1)
    behavior_funcs.append(lambda x: x + 2)
    behavior_funcs.append(lambda x: x + 3)
    behavior_funcs.append(lambda x: x + 4)
    behavior_funcs.append(lambda x: x + 5)


    grammar = ArithmeticGrammar()
    ast_builder = ASTBuilder(grammar)
    matching_equivalence_classes = []
    for ec in equivalence_classes:
        ec_ast = ast_builder.tokenization_to_AST(ec[0])
        num_matching = 0
        for behavior_f in behavior_funcs:
            for x in range(100):
                ec_ast.apply_func_to_ast(grammar.get_ast_apply_func_set_x_val(x))
                if behavior_f(x) == ec_ast.evaluate():
                    num_matching += 1
            if num_matching == 100:
                matching_equivalence_classes.append(ec)
                behavior_funcs.remove(behavior_f)
                break
        

    # with open(equivalence_class_file, 'wb+') as f:
    #     pickle.dump(equivalence_classes[:5], f)

    with open(equivalence_class_file, 'wb+') as f:
        pickle.dump(matching_equivalence_classes, f)

    # with open(equivalence_class_file, 'wb+') as f:
    #     pickle.dump(equivalence_classes[0:41:10], f)
        

BEHAVIOR_DIST_MATRIX_FILE = 'autoencoder/data/semantic_loss_data/valid_behavior_dist_mat.npy'
BEHAVIOR_DIST_MAT_IDX_DICTS_FILE = 'autoencoder/data/semantic_loss_data/valid_behavior_dist_mat_idx.pkl'
EQUIVALENCE_CLASSES_FILE = 'metrics/equivalence_classes.pkl'
if __name__ == '__main__':
    find_equivalence_classes(BEHAVIOR_DIST_MATRIX_FILE, BEHAVIOR_DIST_MAT_IDX_DICTS_FILE, EQUIVALENCE_CLASSES_FILE)