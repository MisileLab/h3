from disnake import Intents, ApplicationCommandInteraction, Guild
from disnake.ext import commands
from os import listdir
from os.path import isfile
from tomli import load
from disnake.ext.commands import CommandNotFound, is_owner, NotOwner
from requests import HTTPError

bot = commands.Bot(intents=Intents.all(), command_prefix='/', help_command=None, owner_ids=[735677489958879324, 338902243476635650])
config = load(open("config.toml", 'rb'))
test_ids = config.get("TEST_GUILDS")
if test_ids is None:
    test_ids = []
block_list = config.get("BLOCK_LIST")
if block_list is None:
    block_list = []

@bot.event
async def on_guild_join(guild: Guild):
    if guild.id not in config["TEST_GUILDS"]:
        await guild.leave()

@bot.event
async def on_ready():
    for i in listdir("modules"):
        if isfile(f"modules/{i}") and i.endswith(".py") and i not in block_list:
            print(f"loading {i}")
            bot.load_extension(f"modules/{i}")
    print("bot ready")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, HTTPError):
        await ctx.send(f"HTTP Error -> Status: {error.response.status_code}")
    elif isinstance(error, NotOwner):
        await ctx.send("You are not the owner of this bot.")
    elif not isinstance(error, CommandNotFound):
        await ctx.send("An error occured while executing the command. Please try again later.")
        raise error

@bot.slash_command(name="ping", guild_ids=test_ids)
async def ping(ctx: ApplicationCommandInteraction):
    await ctx.send(f"pong(discord API latency): {round(ctx.bot.latency, 2) * 1000}ms")

@bot.slash_command(name="load", guild_ids=test_ids)
@is_owner()
async def load(ctx: ApplicationCommandInteraction, module: str):
    bot.load_extension(f"modules/{module}.py")
    await ctx.send(f"loaded {module}")

@bot.slash_command(name="unload", guild_ids=test_ids)
@is_owner()
async def unload(ctx: ApplicationCommandInteraction, module: str):
    bot.unload_extension(f"modules/{module}.py")
    await ctx.send(f"unloaded {module}")

@bot.slash_command(name="reload", guild_ids=test_ids)
@is_owner()
async def reload(ctx: ApplicationCommandInteraction, module: str):
    bot.reload_extension(f"modules/{module}.py")
    await ctx.send(f"reloaded {module}")

bot.i18n.load('locales')
bot.run(config["TOKEN"])
