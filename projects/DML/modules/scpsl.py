from disnake.ext.commands import Cog, Bot, slash_command
from disnake import ApplicationCommandInteraction, Embed
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
    
    @slash_command(name="list", description="SCP: SL 서버의 리스트를 보여줍니다")
    async def scpsl_list(self, ctx: ApplicationCommandInteraction):
        a = self.backend_of_scpsl()
        embed = Embed(title="SCP: SL list", description=f"현재 {len(i for i in a if i['online'])} 개의 서버가 온라인입니다.", color=0x00ff00)
        for i in a:
            embed.add_field(name=f"{i['name']}-{'online' if i['online'] else 'offline'}", value=f"현재 {i['players']}명이 접속중입니다.", inline=False)
        embed.set_author(name="Misile", url="https://github.com/misilelab", icon_url="https://avatars.githubusercontent.com/u/74066467")
        await ctx.send(embed=embed)
