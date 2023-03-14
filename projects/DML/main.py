from disnake import Intents, ApplicationCommandInteraction
from disnake.ext import commands
from os import listdir
from os.path import isfile
from tomli import load

bot = commands.Bot(intents=Intents.all(), command_prefix='/', help_command=None)
config = load(open("config.toml", 'rb'))
test_ids = config.get("TEST_GUILDS")

@bot.event
async def on_ready():
    for i in listdir("modules"):
        if isfile(f"modules/{i}") and i.endswith(".py"):
            print(f"found {i}")
    bot.load_extensions("modules")
    print("bot ready")

@bot.slash_command(name="ping", guild_ids=test_ids)
async def ping(ctx: ApplicationCommandInteraction):
    await ctx.send(f"pong(discord API latency): {round(ctx.bot.latency, 2) * 100}ms")

bot.run(config["TOKEN"])
