from disnake.ext.commands import Cog, Bot
from disnake import Member
from tomli import load

config = load(open("config.toml", "r"))

class LogCog(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot

    @Cog.listener()
    async def on_member_join(self, member: Member):
        member.add_roles(member.guild.get_role(1067441932076318790))
