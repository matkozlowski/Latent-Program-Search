from __future__ import annotations
from astlib.grammar import Grammar, Rule, Nonterminal, Terminal
from types import UnionType


"""
<E>: <E> + <E> | <E> - <E> | <E> * <E> | <In> | <C>
<In>: x
<C>: 1 | 2 | 3 | 4
"""

class ArithmeticGrammar(Grammar):
    def __init__(self):
        super().__init__(start=Expression,
                         rules=[Expression, Constant, Input], 
                         nonterminals=[Addition, Subtraction, Multiplication],
                         terminals=[Input_x, Constant_1, Constant_2, Constant_3, Constant_4])
        
    @staticmethod
    def get_ast_apply_func_set_x_val(x_val: int):
        def apply_func(node: Terminal | Nonterminal):
            if isinstance(node, Input_x):
                node.x_val = x_val

        return apply_func
            



class Expression(Rule):
    def __init__(self):
        super().__init__()
        self.options = [Addition, Subtraction, Multiplication, Input, Constant]


class Addition(Nonterminal):
    left: Expression
    right: Expression

    def __init__(self, left: Expression, right: Expression):
        self.print_symbol = "+"

        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() + self.right.evaluate() # type: ignore
    


class Subtraction(Nonterminal):
    left: Expression
    right: Expression

    def __init__(self, left: Expression, right: Expression):
        self.print_symbol = "-"

        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() - self.right.evaluate() # type: ignore

        
class Multiplication(Nonterminal):
    left: Expression
    right: Expression

    def __init__(self, left: Expression, right: Expression):
        self.print_symbol = "*"

        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() * self.right.evaluate() # type: ignore
        
# class Division(Nonterminal):
#     left: Expression
#     right: Expression

#     def __init__(self, left: Expression, right: Expression):
#         self.print_symbol = "/"

#         self.left = left
#         self.right = right

#     def evaluate(self):
#         right_val = self.right.evaluate() # type: ignore
#         if right_val == 0:
#             return 0
#         return self.left.evaluate() // right_val # type: ignore


class Input(Rule):
    def __init__(self):
        super().__init__()
        self.options = [Input_x]

class Input_x(Terminal):
    x_val: int

    def __init__(self):
        self.print_symbol = "x"

        self.x_val = 0

    def evaluate(self):
        return self.x_val


class Constant(Rule):
    def __init__(self):
        super().__init__()
        self.options = [Constant_1, Constant_2, Constant_3, Constant_4]

class Constant_1(Terminal):
    def __init__(self):
        self.print_symbol = "1"

    def evaluate(self):
        return 1

class Constant_2(Terminal):
    def __init__(self):
        self.print_symbol = "2"
    
    def evaluate(self):
        return 2

class Constant_3(Terminal):
    def __init__(self):
        self.print_symbol = "3"

    def evaluate(self):
        return 3

class Constant_4(Terminal):
    def __init__(self):
        self.print_symbol = "4"

    def evaluate(self):
        return 4

# class Constant_5(Terminal):
#     def __init__(self):
#             self.print_symbol = "5"

#     def evaluate(self):
#         return 5



# RULES
# Input = Input_x
# Constant = Constant_1 | Constant_2 | Constant_3 | Constant_4 | Constant_5
# Expression = Addition | Subtraction | Multiplication | Division | Input | Constant
