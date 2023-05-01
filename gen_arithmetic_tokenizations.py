from astlib.ast_util import ASTBuilder
from grammars.arithmetic_grammar import ArithmeticGrammar, Input_x
from grammars.mountaincar_grammar import MountaincarGrammar


NUM_TOKENIZATIONS_TO_GEN = 15000
AST_DEPTH = 2
MUST_CONTAIN_INPUT = True
OUTPUT_FILE = "tokenizations.txt"

def main():
    mcg = MountaincarGrammar()
    astb = ASTBuilder(mcg)
    asts = astb.generate_unique_asts(10, 10)
    for ast in asts:
        print(ast.pretty_print(), '\n\n')
    exit()

    arithmetic_grammar = ArithmeticGrammar()

    ast_builder = ASTBuilder(arithmetic_grammar)

    asts = ast_builder.generate_unique_asts(NUM_TOKENIZATIONS_TO_GEN, AST_DEPTH)

    with open(OUTPUT_FILE, 'w') as file:
        for ast in asts:
            tokenization = ast.get_tokenization()
            if MUST_CONTAIN_INPUT:
                 # Only save tokenizations of ASTs that contain at least one instance of the Input variable x
                for token in tokenization:
                    if token == arithmetic_grammar.get_token(Input_x):
                        file.write(f"{tokenization}\n")
                        break
            else:
                file.write(f"{tokenization}\n")
                

if __name__ == '__main__':
    main()