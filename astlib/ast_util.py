from typing import List, Set, Type, Callable, Any
from astlib.grammar import Grammar, Rule, Nonterminal, Terminal
import random
from tqdm import tqdm

class AST:
    root: Nonterminal | Terminal
    grammar: Grammar

    def __init__(self, root: Nonterminal | Terminal, grammar: Grammar):
        self.grammar = grammar
        self.root = root


    def evaluate(self) -> Any:
        return self.root.evaluate()


    def get_tokenization(self) -> List[int]:
        return [self.grammar.sos_token] + self._get_tokenization_recursive(self.root, 0) + [self.grammar.eos_token]
    

    def _get_tokenization_recursive(self, node: Nonterminal | Terminal, depth: int) -> List[int]:
        if isinstance(node, Nonterminal):
            tokenization: List[int] = []

            tokenization.append(self.grammar.get_token(type(node)))

            child_names = self.grammar.get_child_names(type(node))

            for child_name in child_names:
                tokenization.extend(self._get_tokenization_recursive(getattr(node, child_name), depth + 1))

            return tokenization
        
        else:
            return [self.grammar.get_token(type(node))]
    

    def apply_func_to_ast(self, apply_func: Callable[[Terminal | Nonterminal], None]):
        self._apply_func_to_ast_recursive(self.root, apply_func)

    def _apply_func_to_ast_recursive(self, node: Nonterminal | Terminal, apply_func: Callable[[Terminal | Nonterminal], None]):
        if isinstance(node, Nonterminal):
            apply_func(node)

            child_names = self.grammar.get_child_names(type(node))

            for child_name in child_names:
                self._apply_func_to_ast_recursive(getattr(node, child_name), apply_func)
        
        else:
            apply_func(node)



    def pretty_print(self) -> str:
        return self._pretty_print_recursive(self.root, 0).rstrip()


    def _pretty_print_recursive(self, node: Nonterminal | Terminal, depth: int) -> str:
        if isinstance(node, Nonterminal):
            tree_str = ''

            tree_str += '  ' * depth + node.print_symbol + '\n'

            child_names = self.grammar.get_child_names(type(node))
            
            for child_name in child_names:
                tree_str += self._pretty_print_recursive(getattr(node, child_name), depth + 1)

            return tree_str
        
        else:
            return '  ' * depth + node.print_symbol + '\n'



class ASTBuilder:
    grammar: Grammar

    def __init__(self, grammar: Grammar):
        self.grammar = grammar


    # Take a tokenization, which should be a list of tokens as created by AST.get_tokenization(), and turn it back
    # into an AST
    # tokenization param can include one, both, or neither of SOS / EOS tokens
    def tokenization_to_AST(self, tokenization: List[int]) -> AST:
        # Remove SOS and EOS tokens from tokenization if present
        if tokenization[0] == self.grammar.sos_token:
            tokenization = tokenization[1:]
        if tokenization[-1] == self.grammar.eos_token:
            tokenization = tokenization[:-1]

        root = self._tokenization_to_AST_recursive(tokenization, [0])

        ast = AST(root, self.grammar)
        return ast


    # idx is List just so that it is passed by reference, should only ever contain one value
    def _tokenization_to_AST_recursive(self, tokenization: List[int], idx: List[int]) -> Nonterminal | Terminal:
        token_type = self.grammar.get_type_from_token(tokenization[idx[0]])

        if issubclass(token_type, Nonterminal):
            num_children = len(self.grammar.get_child_names(token_type))
            children: List[Nonterminal | Terminal] = []

            for i in range(num_children):
                idx[0] += 1
                children.append(self._tokenization_to_AST_recursive(tokenization, idx))

            return token_type(*children)
        else:
            # Terminal Case
            return token_type()



    def generate_unique_asts(self, n: int, max_depth: int) -> List[AST]:
        unique_asts: List[AST] = []
        unique_ast_tokenizations: Set[str] = set()

        pbar = tqdm(total=n)
        while len(unique_asts) < n:
            ast = AST(self._handle_node(self.grammar.start, 0, max_depth), self.grammar)
            prev_len = len(unique_ast_tokenizations)
            unique_ast_tokenizations.add(str(ast.get_tokenization()))
            if len(unique_ast_tokenizations) != prev_len:
                unique_asts.append(ast)
                pbar.update(1)
        pbar.close()
        
        return unique_asts
    

    def _handle_node(self, node: Type[Nonterminal] | Type[Terminal] | Type[Rule], depth: int, depth_limit: int) -> Nonterminal | Terminal:
        if issubclass(node, Nonterminal):
            assert(depth < depth_limit)

            child_types = self.grammar.get_child_types(node)

            node_init_args: List[Nonterminal | Terminal] = []

            for child_type in child_types:
                node_init_args.append(self._handle_node(child_type, depth + 1, depth_limit))
            
            return node(*node_init_args)
        
        elif issubclass(node, Terminal):
            return node()
        
        else:
            # Rule case
            return self._handle_node(self._expand_rule(node(), depth_limit - depth), depth, depth_limit)
    

    def _expand_rule(self, rule: Rule, terminate_within:int | None =None) -> Type[Rule] | Type[Nonterminal] | Type[Terminal]:
        if terminate_within is not None:
            terminal_options: List[Type[Rule] | Type[Nonterminal] | Type[Terminal]] = []
            for option in rule.options:
                if self._can_node_terminate_within_n(option, terminate_within):
                    terminal_options.append(option)
            chosen_option = random.choice(terminal_options)
        else:
            chosen_option = random.choice(rule.options)

        return chosen_option


    # TODO:: memoize this
    def _can_node_terminate_within_n(self, node: Type[Rule] | Type[Nonterminal] | Type[Terminal], n: int) -> bool:
        if n < 0:
            return False

        if issubclass(node, Terminal):
            return True
        
        elif issubclass(node, Rule):
            rule = node()
            for option in rule.options:
                if self._can_node_terminate_within_n(option, n):
                    return True
            return False

        else:
            # Nonterminal case
            child_types = self.grammar.get_child_types(node)

            for child_type in child_types:
                if self._can_node_terminate_within_n(child_type, n - 1):
                    return True
                
            return False





