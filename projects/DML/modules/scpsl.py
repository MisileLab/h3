from disnake.ext.commands import Cog, Bot, slash_command
from disnake import ApplicationCommandInteraction, Localized
from tomli import load
from requests import get

config = load(open("config.toml", "r"))
api_key = config["scpsl_api_key"]
acc_id = config["scpsl_acc_id"]

class SCPSL(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot

    @staticmethod()
    def backend_of_scpsl():
        a = get("https://api.scpslgame.com/serverinfo.php", params={
            "id": acc_id,
            "key": api_key,
            "players": True,
            "online": True
        })
        a.raise_for_status()
        return a.json()
    
    @slash_command(name="list", description=Localized("scpsl_list_description"))
    async def scpsl_list(self, ctx: ApplicationCommandInteraction):
        pass
