from tabulate import tabulate
from typer import Typer, Argument
from datetime import datetime
from requests import get
from simple_term_menu import TerminalMenu

from misilelibpy import write_once, read_once
from json import loads, dumps
from os.path import isfile

app = Typer()
today = datetime.now()
if isfile("ALMdata.json") is False or isfile("config.json") is False:
    print("please init the configration and data")
else:
    data = loads(read_once("ALMdata.json"))
    config = loads(read_once("config.json"))

def typer_argument(default):
    return Argument(default=default)

@app.command(name="list")
def list_the_values():
    """list the moneybook"""
    print(tabulate(data, headers="keys", tablefmt="rst"))

@app.command()
def init():
    """init and reset the configuration and data"""
    write_once("ALMdata.json", '[]')
    write_once("config.json", '{"currency": "USD", "api_key": ""}')

@app.command()
def add(amount: int, currency = typer_argument(config["currency"]), repeater = typer_argument(""), comment = typer_argument("")):
    """add history to the moneybook"""
    if config["currency"] != currency:
        _amount = get(f"https://api.apilayer.com/exchangerates_data/convert?to={config['currency']}&from={currency}&amount={amount}", headers={"apikey": config["api_key"]}).json()
        if _amount["success"]:
            amount = int(_amount["result"])
        else:
            print("failed response from exchangerate api")
            print(_amount)
    data.append({"date": today.strftime("%Y-%m-%d %H:%M:%S"), "value": amount, "repeater": repeater, "comment": comment})
    write_once("ALMdata.json", dumps(data))

@app.command()
def remove():
    """remove history to the moneybook"""
    _data = list(map(str, data))
    if not _data:
        print("no data to remove")
        return
    terminal_menu = TerminalMenu(_data, multi_select=True, show_multi_select_hint=True)
    terminal_menu.show()
    for i in terminal_menu.chosen_menu_indices:
        del data[i]
    write_once("ALMdata.json", dumps(data))

@app.command(name="sum")
def sumlist():
    """sum all result and print it"""
    print(sum(int(i["value"]) for i in data))

if __name__ == "__main__":
    app()
