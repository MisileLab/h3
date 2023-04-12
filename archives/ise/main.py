from latex2sympy2 import latex2sympy
from sympy import Atom
from typer import Typer
from sympy_addons import Query

app = Typer()

@app.command()
def solve(latex: str):
    "Solve the latex equation"
    print("Convert the latex..")
    _converted = latex2sympy(latex)
    print("Solving the equation")
    query = Query(isinstance=Atom).run(_converted)
    for i in query:
        print(i)
    print(f"answer: {query.last()}")

if __name__ == "__main__":
    app()
