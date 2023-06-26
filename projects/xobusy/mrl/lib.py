from z3 import Solver, Int

def add_ascii(p: Solver, x: list[Int]):
    for i in x:
        p.add(i >= 0)
        p.add(i <= 127)

def print_flag(s: Solver, x: list[Int]):
    print("".join([chr(int(str(s.model()[i]))) for i in x]))
