from disnake import Intents, ApplicationCommandInteraction
from disnake.ext import commands
from requests import get
from tomli import load

bot = commands.Bot(intents=Intents.all(), command_prefix='/', help_command=None)
config = load(open("config.toml", 'rb'))
# sunrin_ids = [921295966143926352, 1082472876449476708]
sunrin_ids = [921295966143926352]

@bot.event
async def on_ready():
    print("ae")

@bot.slash_command(name="ae", guild_ids=sunrin_ids)
async def hello(ctx: ApplicationCommandInteraction):
    await ctx.send("ae")

bot.run(config["TOKEN"])
