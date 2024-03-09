from disnake.ext.commands import Cog, Bot, slash_command
from disnake.ext import tasks
from disnake import ApplicationCommandInteraction, Embed, Activity, ActivityType
from tomli import load
from requests import get

config = load(open("config.toml", "rb"))
api_key = config["scpsl_api_key"]
acc_id = config["scpsl_acc_id"]
test_ids = config["TEST_GUILDS"]

def backend_of_scpsl():
    a = get("https://api.scpslgame.com/serverinfo.php", params={
        "id": acc_id,
        "key": api_key,
        "players": True,
        "online": True
    })
    print(a.headers)
    a.raise_for_status()
    return a.json()

class SCPSL(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot
        self.scpsl_presence.start()

    def cog_unload(self):
        self.scpsl_presence.cancel()
    
    @slash_command(name="list", description="SCP: SL 서버의 리스트를 보여줍니다", guild_ids=test_ids)
    async def scpsl_list(self, ctx: ApplicationCommandInteraction):
        await ctx.response.defer()
        a = backend_of_scpsl()
        if not a["Success"]:
            await ctx.send("Failed", ephemeral=True)
            return
        a = a["Servers"]
        embed = Embed(title="SCP: SL list", description=f"현재 {len([i for i in a if i['Online']])} 개의 서버가 온라인입니다.", color=0x00ff00)
        for i in a:
            embed.add_field(name=f"{i['ID']}-{'online' if i['Online'] else 'offline'}", value=f"현재 {i['Players']}명이 접속중입니다.", inline=False)
        embed.set_author(name="Misile", url="https://github.com/misilelab", icon_url="https://avatars.githubusercontent.com/u/74066467")
        await ctx.send(embed=embed, ephemeral=True)

    @tasks.loop(minutes=1)
    async def scpsl_presence(self):
        a = backend_of_scpsl()["Servers"]
        try:
            await self.bot.change_presence(activity=Activity(type=ActivityType.playing, name=f"{a[0]['Players']}명"))
        except KeyError:
            print(a)
        del a

def setup(self: Bot):
    self.add_cog(SCPSL(self))
