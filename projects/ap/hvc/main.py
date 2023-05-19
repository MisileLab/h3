from typer import Typer, Argument
from typing_extensions import Annotated
import modules.module1 as md1
from re import compile as compile_regex
from pyfiglet import Figlet
from time import time
from tqdm import tqdm
from requests import post

def pre_help():
    print(trender.renderText('hvc') + 'by MisileLaboratory', end='')

def aarg(type: type, help: str):
    return Annotated[type, Argument(default=None, help=help)]

trender = Figlet(font='slant')
cli = Typer(name="Hacker's Vulnerability Cracker", help=pre_help())

@cli.command(name="bsql")
def blind_sql(
    url: aarg(str, "url of target"),
    name: aarg(str, help="name of account"),
    dstring: aarg(str, help="detect string (you can insert admin special message like hello, admin)")
):
    """blind sql injection"""
    param = {
        
    }
    cookie = {
        
    }
    stime = time()
    print(f"password is {md1.find_pw(name, url, param, dstring, cookie)}")
    print(f"elapsed time is {time() - stime}")

@cli.command(name="bmongo")
def blind_mongo(
    url: aarg(str, "url of target"),
    rname: aarg(str, help="name of account regex (for filter bypass)"),
    dstring: aarg(str, help="detect string (you can insert admin special message like hello, admin)")
):
    """mongodb blind injection"""
    stime = time()
    compile_regex(rname)
    md1.nosql_pw(url, rname, dstring)
    print(f"elapsed time is {time() - stime}")

if __name__ == '__main__':
    cli()
