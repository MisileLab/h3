from secrets import SystemRandom
from tqdm import tqdm

def add_safely(org: dict, key: str, amount: int):
    if key not in org:
        org[key] = 0
    org[key] += amount

def random_number(minn: int, maxn: int) -> int:
    return SystemRandom().randint(minn, maxn)

def random_percent(percent: float | int) -> bool:
    return SystemRandom().random() < percent / 100

def random_numbers(minn: int, maxn: int, count: int) -> list[int]:
    return [random_number(minn, maxn) for _ in tqdm(range(count))]

def random_percents(percent: float | int, count: int) -> list[bool]:
    return [random_percent(percent) for _ in tqdm(range(count))]

def scp_sl_loot():
    weapon_locker_large = int(input("Weapon Locker Type 21(Large): "))
    weapon_locker_small = int(input("Weapon Locker Type 21(Small): "))
    weapon_locker_small_hcz_ez = int(input("Weapon Locker Type 21(HCZ/EZ Checkpoint)"))
    drawer = int(input("Drawer: "))
    standard_locker = int(input("Standard Locker: "))
    mtf_e11_sr = int(input("MTF-E11-SR Rack: "))
    first_aid_cabinet = int(input("First Aid Cabinet: "))
    items = {}
    add_safely(items, "first aid kit", first_aid_cabinet)
    add_safely(items, "adrenaline", sum(random_percents(38.46, count=first_aid_cabinet)))
    add_safely(items, "MTF-E11-SR", mtf_e11_sr)
    add_safely(items, "5.56x45mm", mtf_e11_sr*125)
    add_safely(items, "High-Explosive Grenades", mtf_e11_sr*2)
    for _ in tqdm(range(standard_locker)):
        if random_percent(17.65):
            add_safely(items, "Painkillers", 3)
        if random_percent(17.65):
            add_safely(items, "Zone Manager Keycard", 1)
        if random_percent(17.65):
            add_safely(items, "Scientist Keycard", 1)
        if random_percent(11.76):
            add_safely(items, "Janitor Keycard", 1)
        if random_percent(11.76):
            add_safely(items, "Flashlight", 1)
        if random_percent(11.76):
            add_safely(items, "First Aid Kit", 1)
        if random_percent(5.88):
            add_safely(items, "Radio", 1)
        if random_percent(5.88):
            add_safely(items, "Coin", 4)
    for _ in tqdm(range(weapon_locker_large)):
        if random_percent(14.29):
            add_safely(items, "5.56x45mm", random_number(40, 80))
        if random_percent(14.29):
            add_safely(items, "Light Armor", 1)
        if random_percent(14.29):
            add_safely(items, "Combat Armor", 1)
        if random_percent(14.29):
            add_safely(items, "Heavy Armor", 1)
        if random_percent(14.29):
            add_safely(items, "Crossvec", 1)
        if random_percent(14.29):
            add_safely(items, "FSP-9", 1)
        if random_percent(14.29):
            add_safely(items, "COM-18", 1)
    for _ in tqdm(range(weapon_locker_small)):
        if random_percent(40):
            add_safely(items, "5.56x45mm", random_number(40, 80))
        if random_percent(20):
            add_safely(items, "9x19mm", random_number(15, 60))
        if random_percent(20):
            add_safely(items, "High-Explosive Grenade", random_number(1, 2))
        if random_percent(20):
            add_safely(items, "Flashbang Grenade", random_number(1, 3))
    add_safely(items, "9x19mm", sum(random_numbers(15, 90, weapon_locker_small_hcz_ez)))
    add_safely(items, "5.56x45mm", sum(random_numbers(40, 80, weapon_locker_small_hcz_ez)))
    for _ in tqdm(range(drawer)):
        if random_percent(37.5):
            add_safely(items, "First Aid Kit", 1)
        elif random_percent(25):
            add_safely(items, "Painkiller", random_number(1, 3))
        elif random_percent(25):
            add_safely(items, "Radio", 1)
        elif random_percent(12.5):
            add_safely(items, "Coin", random_number(1, 20))
    for i, i2 in items.items():
        print(f"{i}: {i2}")
