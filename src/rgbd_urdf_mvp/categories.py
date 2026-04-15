from __future__ import annotations

SUPPORTED_CATEGORIES: tuple[str, ...] = (
    "door",
    "drawer",
    "blender",
    "refrigerator",
    "microwave",
    "coffeemachine",
    "dishwasher",
    "electrickettle",
    "oven",
    "rangehood",
    "sink",
    "standmixer",
    "stove",
    "stovetop",
    "toaster",
    "toasteroven",
)

# Categories that usually represent rotary controls and can reasonably fall back
# to a full-turn revolute range if MJCF does not provide explicit limits.
FULL_ROTATION_CATEGORIES: tuple[str, ...] = (
    "blender",
    "coffeemachine",
    "standmixer",
    "stove",
    "stovetop",
    "rangehood",
    "toaster",
)

# Categories that usually behave like doors/lids with a partial rotation range.
HINGE_CATEGORIES: tuple[str, ...] = (
    "door",
    "refrigerator",
    "microwave",
    "oven",
    "dishwasher",
    "toasteroven",
    "electrickettle",
)

# Categories whose simple scaffold fallback is better represented by a prismatic DOF.
PRISMATIC_CATEGORIES: tuple[str, ...] = (
    "drawer",
)


def normalize_category(category: str) -> str:
    return category.strip().lower()
