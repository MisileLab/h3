from disnake import Intents, ApplicationCommandInteraction
from disnake.ext import commands
from os import listdir
from os.path import isfile
from tomli import load
from disnake.ext.commands import CommandNotFound

bot = commands.Bot(intents=Intents.all(), command_prefix='/', help_command=None)
config = load(open("config.toml", 'rb'))
test_ids = config.get("TEST_GUILDS")
if test_ids is None:
    test_ids = []
block_list = config.get("BLOCK_LIST")
if block_list is None:
    block_list = []

@bot.event
async def on_ready():
    for i in listdir("modules"):
        if isfile(f"modules/{i}") and i.endswith(".py") and i not in block_list:
            print(f"loading {i}")
            bot.load_extension(f"modules/{i}")
    print("bot ready")

@bot.event
async def on_command_error(ctx, error):
    if not isinstance(error, CommandNotFound):
        await ctx.send("An error occured while executing the command. Please try again later.")
        raise error

@bot.slash_command(name="ping", guild_ids=test_ids)
async def ping(ctx: ApplicationCommandInteraction):
    await ctx.send(f"pong(discord API latency): {round(ctx.bot.latency, 2) * 1000}ms")

bot.run(config["TOKEN"])
