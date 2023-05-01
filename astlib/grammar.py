from __future__ import annotations 
import inspect
from typing import List, Dict, Type, get_type_hints, Any
from types import UnionType



# Rule = UnionType | type

# Each subclass of Rule must contain a member called "options"
# "options" should be a list that contains subclasses of Rule, Nonterminal, or Terminal
# This list represents the possible things the rule can be expanded into
class Rule:
    options: List[Type[Rule] | Type[Nonterminal] | Type[Terminal]]

    def __init__(self):
        self.options = []

# Each subclass of Nonterminal should have all the arguments to the __init__ function be subclasses of Rule, Nonterminal, or Terminal
# These arguments represent the children of the Nonterminal node in the AST
# For each such argument, there must be a member of the subclass with the same name and type, although other unrelated members are allowed as well
# These members will store the AST nodes that correspond to them
# It should also contain a member called "print_symbol" that represents what the node will be printed as in the AST
# The evaluate function should return the value of evaluating the Nonterminal as well as all of its children
class Nonterminal:
    print_symbol: str

    def __init__(self):
        pass

    def evaluate(self) -> Any:
        pass

# Each subclass of Terminal should contain a member called "print_symbol" that represents what the node will be printed as in the AST
# The evaluate function should return the value of the Terminal
class Terminal:
    print_symbol: str

    def __init__(self):
        pass

    def evaluate(self) -> Any:
        pass


class Grammar:
    start: Type[Rule]
    rules: List[Type[Rule]]
    nonterminals: List[Type[Nonterminal]]
    terminals: List[Type[Terminal]]
    tokenizations: Dict[Type[Nonterminal] | Type[Terminal], int]
    reverse_tokenizations: Dict[int, Type[Nonterminal] | Type[Terminal]]
    sos_token: int
    eos_token: int

    def __init__(self, start: Type[Rule], rules: List[Type[Rule]], nonterminals: List[Type[Nonterminal]], terminals: List[Type[Terminal]]):
        self.start = start
        self.rules = rules
        self.nonterminals = nonterminals
        self.terminals = terminals

        # Build simple tokenization
        self.sos_token = 1
        self.eos_token = 2

        self.tokenizations = {}
        self.reverse_tokenizations = {}
        next_unused_token = 3
        for nt in self.nonterminals:
            self.tokenizations[nt] = next_unused_token
            self.reverse_tokenizations[next_unused_token] = nt
            next_unused_token += 1
        for t in self.terminals:
            self.tokenizations[t] = next_unused_token
            self.reverse_tokenizations[next_unused_token] = t
            next_unused_token += 1

    def get_token(self, node: Type[Nonterminal] | Type[Terminal]) -> int:
        return self.tokenizations[node]
    
    def get_type_from_token(self, token: int) -> Type[Nonterminal] | Type[Terminal]:
        return self.reverse_tokenizations[token]

    def get_child_types(self, node: Type[Nonterminal]) -> List[Type[Rule] | Type[Nonterminal] | Type[Terminal]]:
        init_signature = inspect.signature(node.__init__)
        init_type_hints = get_type_hints(node.__init__)
        return [init_type_hints[param.name] for param in init_signature.parameters.values() if param.name != "self"]
    
    def get_child_names(self, node: Type[Nonterminal]) -> List[str]:
        init_signature = inspect.signature(node.__init__)
        return [param.name for param in init_signature.parameters.values() if param.name != "self"]