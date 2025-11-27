"""Configuration for item valuation system."""

from typing import Dict

# Item type enumeration
ITEM_TYPES = [
    "weapon",
    "armor",
    "material",
    "consumable",
    "mod",
    "currency",
]

# Rarity levels (lower index = higher rarity)
RARITY_LEVELS = [
    "legendary",
    "epic",
    "rare",
    "uncommon",
    "common",
]

# Item value mapping by type and rarity
ITEM_VALUE_MAP: Dict[str, Dict[str, float]] = {
    "weapon": {
        "legendary": 5000.0,
        "epic": 2500.0,
        "rare": 1000.0,
        "uncommon": 400.0,
        "common": 100.0,
    },
    "armor": {
        "legendary": 3000.0,
        "epic": 1500.0,
        "rare": 600.0,
        "uncommon": 250.0,
        "common": 80.0,
    },
    "material": {
        "legendary": 2000.0,
        "epic": 800.0,
        "rare": 300.0,
        "uncommon": 100.0,
        "common": 30.0,
    },
    "consumable": {
        "legendary": 1000.0,
        "epic": 400.0,
        "rare": 150.0,
        "uncommon": 50.0,
        "common": 20.0,
    },
    "mod": {
        "legendary": 3500.0,
        "epic": 1800.0,
        "rare": 700.0,
        "uncommon": 300.0,
        "common": 100.0,
    },
    "currency": {
        "legendary": 10000.0,
        "epic": 5000.0,
        "rare": 2000.0,
        "uncommon": 500.0,
        "common": 100.0,
    },
}

# Default value if item type/rarity not found
DEFAULT_ITEM_VALUE = 50.0

# Confidence threshold for detection
MIN_DETECTION_CONFIDENCE = 0.5

# Vision Model Configuration
VISION_MODEL_NAME = "gpt-5-nano"
VISION_API_TIMEOUT = 30.0  # seconds
VISION_API_MAX_RETRIES = 3

# Value multipliers based on game phase (can be adjusted dynamically)
PHASE_MULTIPLIERS = {
    "early_wipe": 1.5,  # Season start
    "mid_wipe": 1.0,    # Mid season
    "late_wipe": 0.7,   # Late season (inflation)
}

# Set bonus multipliers (for item combinations)
SET_BONUSES = {
    "weapon_armor_combo": 1.1,  # 10% bonus for weapon + armor
    "full_loadout": 1.2,        # 20% bonus for complete set
}


def get_item_value(item_type: str, rarity: str) -> float:
    """
    Get base value for an item.

    Args:
        item_type: Type of item (e.g., "weapon", "armor")
        rarity: Rarity level (e.g., "epic", "rare")

    Returns:
        Base value for the item
    """
    if item_type not in ITEM_VALUE_MAP:
        return DEFAULT_ITEM_VALUE

    if rarity not in ITEM_VALUE_MAP[item_type]:
        return DEFAULT_ITEM_VALUE

    return ITEM_VALUE_MAP[item_type][rarity]


def apply_phase_multiplier(value: float, phase: str = "mid_wipe") -> float:
    """
    Apply game phase multiplier to value.

    Args:
        value: Base value
        phase: Game phase ("early_wipe", "mid_wipe", "late_wipe")

    Returns:
        Adjusted value
    """
    multiplier = PHASE_MULTIPLIERS.get(phase, 1.0)
    return value * multiplier
