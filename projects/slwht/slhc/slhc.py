from typer import Typer

app = Typer()

@app.command(name="decompile", description="decompile the program to C")
def decompile(program: str, output: str):
    pass
