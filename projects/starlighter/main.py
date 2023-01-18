from rich import print as rprint
from simple_term_menu import TerminalMenu
from os import _exit
from sys import version as pversion

version = "0.1.0pre"

tm = TerminalMenu(["start", "options", "info", "exit"])
tm.show()

if tm.chosen_menu_entry == "start":
    raise NotImplementedError
elif tm.chosen_menu_entry == "options":
    raise NotImplementedError
elif tm.chosen_menu_entry == "info":
    print(f"Python {pversion}")
    rprint("[yellow]Star[/yellow][blue]Light[/blue][white]er[/white] by [red]Misile[/red][purple]Laboratory[/purple]")
    print(f"Version {version}")
elif tm.chosen_menu_entry == "exit":
    _exit(0)
