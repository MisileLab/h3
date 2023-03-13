from disnake.ext.commands import Cog, Bot, slash_command
from disnake import ApplicationCommandInteraction, SelectOption, MessageInteraction
from disnake.ui import StringSelect, View
from tomli import load
from requests import get
from datetime import datetime

sunrin_ids = [921295966143926352, 1082472876449476708]

class disnake_view(View):
    def __init__(self, views: list):
        super().__init__()

        for i in views:
            self.add_item(i)

class SNTMealMenu(StringSelect):
    def __init__(self, inter: ApplicationCommandInteraction, elements: list, meals: list[dict]):
        self.interaction = inter
        self.meals = meals
        options = []
        for i, i2 in enumerate(elements):
            if i == 24:
                break
            options.append(i2)
            
        super().__init__(
            placeholder="날짜를 선택하세요.",
            options=options,
        )
    
    async def callback(self, inter: MessageInteraction):
        if inter.author != self.interaction.author:
            await inter.send("당신은 명령어를 사용한 사람이 아닙니다.", ephemeral=True)
            return
        label = self._selected_values[0]
        for i in self.meals:
            if i["MLSV_YMD"][6:8] == label:
                await inter.send("\n".join(i["DISH"].split("<br/>")))
                break
        

class SNTModule(Cog):
    def __init__(self, bot: Bot):
        self.bot = bot
    
    @slash_command(name="meal", guild_ids=sunrin_ids)
    async def meal(self, inter: ApplicationCommandInteraction):
        meal = self.meal_backend()
        days = [i["MLSV_YMD"][6:8] for i in meal]
        await inter.send(view=disnake_view([SNTMealMenu(inter, days, meal)]))
    
    @slash_command(name="today", guild_ids=sunrin_ids)
    async def today(self, inter: ApplicationCommandInteraction):
        await inter.send("\n".join(self.meal_backend()[0]["DISH"].split("<br/>")))
    
    @slash_command(name="tomorrow", guild_ids=sunrin_ids)
    async def tomorrow(self, inter: ApplicationCommandInteraction):
        await inter.send("\n".join(self.meal_backend()[1]["DISH"].split("<br/>")))
    
    def meal_backend(self) -> list[dict]:
        config = load(open("config.toml", "rb"))
        date = datetime.now()
        _month = date.month
        _day = date.day
        if len(str(_month)) == 1:
            _month = f"0{_month}"
        if len(str(_day)) == 1:
            _day = f"0{_day}"
        resp = get(f"https://open.neis.go.kr/hub/mealServiceDietInfo?ATPT_OFCDC_SC_CODE=B10&KEY={config['NEISAPIKEY']}&SD_SCHUL_CODE=7010536&TYPE=json&MLSV_FROM_YMD={date.year}{_month}{_day}")
        resp.raise_for_status()
        return [{"MLSV_YMD": i["MLSV_FROM_YMD"], "DISH": i["DDISH_NM"]} for i in resp.json()["mealServiceDietInfo"][1]['row']]

def setup(self: Bot):
    self.add_cog(SNTModule(self))
