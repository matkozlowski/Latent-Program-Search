from __future__ import annotations
from astlib.grammar import Grammar, Rule, Nonterminal, Terminal
from types import UnionType


"""
<E>: if <B> then <Action> else <Action>   
<B>: <NumP> > <NumP> | <NumV> > <NumV> | <B2> and <B2> | <B2> or <B2>
<B2>: <NumP> > <NumP> | <NumV> > <NumV>
<NumP>: <NumP2> | <NumP2> + <NumP2> | <NumP2> - <NumP2>
<NumP2>: <Position> | <Cp>
<NumV>: <NumV2> | <NumV2> + <NumV2> | <NumV2> - <NumV2>
<NumV2>: <Velocity> | <CV>
<Position>: position
<Velocity>: velocity
<Cp>: (-1.2, -1) | (-1, -0.8) | (-0.8, -0.6) | (-0.6, -0.4) | (-0.4, -0.2) | (-0.2, 0) | (0, 0.2) | (0.2, 0.4) | (0.4, 0.5)
<Cv>: (-0.07, -0.06) | (-0.06, -0.05) | (-0.05, -0.04) | (-0.04, -0.03) | (-0.03, -0.02) | (-0.02, -0.01) etc...
<Action> left | right
"""

class MountaincarGrammar(Grammar):
    def __init__(self):
        super().__init__(start=Expression,
                         rules=[Expression, BoolExpr, BoolExpr2, NumP, NumP2, NumV, NumV2, Cp, Cv, Action], 
                         nonterminals=[IfThenElse, GreaterThanP, GreaterThanV, And, Or, AddP, SubtractP, AddV, SubtractV],
                         terminals=[Position, Velocity, ActionLeft, ActionRight, Cp_n1_2, Cp_n1_0, Cp_n0p8, Cp_n0p6, 
                                    Cp_n0p4, Cp_n0p2, Cp_0p0, Cp_0p2, Cp_0p4, Cv_n0p07, Cv_n0p06, Cv_n0p05, Cv_n0p04, 
                                    Cv_n0p03, Cv_n0p02, Cv_n0p01, Cv_0p00, Cv_0p01, Cv_0p02, Cv_0p03, Cv_0p04, Cv_0p05, Cv_0p06]
                        )                   
            



class Expression(Rule):
    def __init__(self):
        super().__init__()
        self.options = [IfThenElse]


class IfThenElse(Nonterminal):
    condition: BoolExpr
    action1: Action
    action2: Action

    def __init__(self, condition: BoolExpr, action1: Action, action2: Action):
        self.print_symbol = "if"
        self.condition = condition
        self.action1 = action1
        self.action2 = action2

    def evaluate(self) -> str:
        if self.condition.evaluate():
            return self.action1.evaluate()
        else:
            return self.action2.evaluate()
        
    
class BoolExpr(Rule):
    def __init__(self):
        super().__init__()
        self.options = [GreaterThanP, GreaterThanV, And, Or]


class BoolExpr2(Rule):
    def __init__(self):
        super().__init__()
        self.options = [GreaterThanP, GreaterThanV]


class GreaterThanP(Nonterminal):
    left: NumP
    right: NumP

    def __init__(self, left: NumP, right: NumP):
        self.print_symbol = ">"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() > self.right.evaluate()
    

class GreaterThanV(Nonterminal):
    left: NumV
    right: NumV

    def __init__(self, left: NumV, right: NumV):
        self.print_symbol = ">"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() > self.right.evaluate()


class And(Nonterminal):
    left: BoolExpr2
    right: BoolExpr2

    def __init__(self, left: BoolExpr2, right: BoolExpr2):
        self.print_symbol = "and"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() and self.right.evaluate()
    

class Or(Nonterminal):
    left: BoolExpr2
    right: BoolExpr2

    def __init__(self, left: BoolExpr2, right: BoolExpr2):
        self.print_symbol = "or"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() or self.right.evaluate()
    

class NumP(Rule):
    def __init__(self):
        super().__init__()
        self.options = [NumP2, AddP, SubtractP]


class NumP2(Rule):
    def __init__(self):
        super().__init__()
        self.options = [Position, Cp]


class NumV(Rule):
    def __init__(self):
        super().__init__()
        self.options = [NumV2, AddV, SubtractV]


class NumV2(Rule):
    def __init__(self):
        super().__init__()
        self.options = [Velocity, Cv]


class AddP(Nonterminal):
    left: NumP2
    right: NumP2

    def __init__(self, left: NumP2, right: NumP2):
        self.print_symbol = "+"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() + self.right.evaluate()


class SubtractP(Nonterminal):
    left: NumP2
    right: NumP2

    def __init__(self, left: NumP2, right: NumP2):
        self.print_symbol = "-"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() - self.right.evaluate()


class AddV(Nonterminal):
    left: NumV2
    right: NumV2

    def __init__(self, left: NumV2, right: NumV2):
        self.print_symbol = "+"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() + self.right.evaluate()


class SubtractV(Nonterminal):
    left: NumV2
    right: NumV2

    def __init__(self, left: NumV2, right: NumV2):
        self.print_symbol = "-"
        self.left = left
        self.right = right

    def evaluate(self):
        return self.left.evaluate() - self.right.evaluate()


class Position(Terminal):
    position: float

    def __init__(self):
        self.print_symbol = "position"

        self.position = 0

    def evaluate(self) -> float:
        return self.position


class Velocity(Terminal):
    velocity: float

    def __init__(self):
        self.print_symbol = "velocity"

        self.velocity = 0

    def evaluate(self) -> float:
        return self.velocity


class Cp(Rule):
    def __init__(self):
        super().__init__()
        self.options = [Cp_n1_2, Cp_n1_0, Cp_n0p8, Cp_n0p6, Cp_n0p4, Cp_n0p2, Cp_0p0, Cp_0p2, Cp_0p4]


class Cp_n1_2(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-1.2, -1.0)"

        self.value = 0

    def evaluate(self):
        assert(self.value >= -1.2 and self.value <= -1.0)
        return self.value

class Cp_n1_0(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-1.0, -0.8)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -1.0 and self.value <= -0.8)
        return self.value

class Cp_n0p8(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.8, -0.6)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.8 and self.value <= -0.6)
        return self.value

class Cp_n0p6(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.6, -0.4)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.6 and self.value <= -0.4)
        return self.value

class Cp_n0p4(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.4, -0.2)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.4 and self.value <= -0.2)
        return self.value


class Cp_n0p2(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.2, 0.0)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.2 and self.value <= 0.0)
        return self.value

class Cp_0p0(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.0, 0.2)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.0 and self.value <= 0.2)
        return self.value

class Cp_0p2(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.2, 0.4)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.2 and self.value <= 0.4)
        return self.value

class Cp_0p4(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.4, 0.6)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.4 and self.value <= 0.6)
        return self.value

class Cv(Rule):
    def __init__(self):
        super().__init__()
        self.options = [Cv_n0p07, Cv_n0p06, Cv_n0p05, Cv_n0p04, Cv_n0p03, Cv_n0p02, Cv_n0p01, Cv_0p00, Cv_0p01, Cv_0p02, Cv_0p03, Cv_0p04, Cv_0p05, Cv_0p06]

class Cv_n0p07(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.07, -0.06)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.07 and self.value <= -0.06)
        return self.value

class Cv_n0p06(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.06, -0.05)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.06 and self.value <= -0.05)
        return self.value

class Cv_n0p05(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.05, -0.04)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.05 and self.value <= -0.04)
        return self.value

class Cv_n0p04(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.04, -0.03)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.04 and self.value <= -0.03)
        return self.value

class Cv_n0p03(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.03, -0.02)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.03 and self.value <= -0.02)
        return self.value

class Cv_n0p02(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.02, -0.01)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.02 and self.value <= -0.01)
        return self.value

class Cv_n0p01(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(-0.01, 0.0)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= -0.01 and self.value <= 0.0)
        return self.value

class Cv_0p06(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.06, 0.07)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.06 and self.value <= 0.07)
        return self.value

class Cv_0p05(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.05, 0.06)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.05 and self.value <= 0.06)
        return self.value

class Cv_0p04(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.04, 0.05)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.04 and self.value <= 0.05)
        return self.value

class Cv_0p03(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.03, 0.04)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.03 and self.value <= 0.04)
        return self.value

class Cv_0p02(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.02, 0.03)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.02 and self.value <= 0.03)
        return self.value

class Cv_0p01(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.01, 0.02)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.01 and self.value <= 0.02)
        return self.value
    

class Cv_0p00(Terminal):
    value: float

    def __init__(self):
        self.print_symbol = "(0.00, 0.01)"
        self.value = 0

    def evaluate(self):
        assert(self.value >= 0.0 and self.value <= 0.01)
        return self.value


class Action(Rule):
    def __init__(self):
        super().__init__()
        self.options = [ActionLeft, ActionRight]

class ActionLeft(Terminal):
    def __init__(self):
        self.print_symbol = "left"

    def evaluate(self):
        return 0

class ActionRight(Terminal):
    def __init__(self):
        self.print_symbol = "right"

    def evaluate(self):
        return 2

