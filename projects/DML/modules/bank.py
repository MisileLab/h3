from disnake.ext.commands import Cog, Bot, slash_command
from disnake import ApplicationCommandInteraction
from requests import get
from tomli import load

config = load("config.toml")

class Bank(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot
    
    @staticmethod
    def get_basic_header():
        return {
            "Authorization": f"Bearer {config['KTFC_TOKEN']}",
            "fintech_use_num": config["FINTECH_USE_NUM"],
            "bank_tran": config["BANK_TRAN"]
            }
    
    @slash_command(name="get_money")
    def get_money(self, inter: ApplicationCommandInteraction):
        pass

def setup(self: Bot):
    self.add_cog(Bank(self))
