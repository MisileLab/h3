#1
for i in range(1, 16, 2):
    print('{:^20}'.format('*' * i))
for i in range(13, 0, -2):
    print('{:^20}'.format('*' * i))
#2
from time import sleep
for _ in range(10):
    print("Hello, world!")
    sleep(1)
#3
from asyncio import sleep as asleep
class WhatisThisGameThisisVeryNotGood:
    def __init__(self):
        self.speed = 1
        self.coin = 52
        self.hdistance = 100
        self.hprice = 24
        self.lock = False
    
    def buy_house(self):
        if self.coin < self.hprice:
            raise ValueError("no money")
        self.coin -= self.hprice
        self.hprice += 4
        self.hdistance -= 1
    
    async def buy_cucumber(self):
        if self.lock:
            raise ValueError("bruh lock")
        if self.coin < 2:
            raise ValueError("no money")
        self.lock = True
        self.coin -= 2
        self.speed += 1
        await asleep(self.hdistance / self.speed)
        self.lock = False
    
    def print_stats(self):
        print(f"speed: {self.speed}")
        print(f"coin: {self.coin}")
        print(f"house distance: {self.hdistance}")
        print(f"house price: {self.hprice}")
