import platform
from sys import version as pyversion
from sys import exit
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import Reactive
from textual.widgets import Button, Header, Static
from misilelibpy import read_once

_version = "0.0.1a"

def linux_distribution():
  try:
    return platform.linux_distribution()
  except Exception:
    return ""

def start():
    pass

def info():
    pass

def options():
    pass

class Display(Static):
    """ae"""

class InfoScreen(Static):
    def on_button_pressed(self, event: Button.Pressed):
        SomeMekanism().run()
    
    def compose(self) -> ComposeResult:
        pversion = ','.join(pyversion.split('\n'))
        infos = (
            f"SomeMekanism {_version}",
            f"Run with python {pversion}",
            f"Run with {platform.system()} {linux_distribution()} {platform.machine()}"
            f"Run with {platform.uname()}"
        )
        for i in infos:
            yield Display(i)
        yield Button("back")

class Buttons(Static):
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "start":
            start()
        elif event.button.id == "info":
            info()
        elif event.button.id == "options":
            options()
        else:
            exit()

    def compose(self) -> ComposeResult:
        yield Button("start", variant="success")
        yield Button("info")
        yield Button("options")
        yield Button("exit", variant="error")

class SomeMekanism(App):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Buttons()

if __name__ == "__main__":
    SomeMekanism().run()