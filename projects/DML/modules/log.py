from disnake.ext.commands import Cog, Bot
from tomli import load
from datetime import datetime
from time import mktime

config = load(open("config.toml", "r"))

class LogCog(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot
        self.channel = self.bot.get_channel(config["log_channel"])
        if self.channel is None:
            raise ValueError("Log channel not found")

    @staticmethod()
    def timestamp() -> str:
        return f">>= <t:{int(mktime(datetime.now().timetuple()))}:F>"
