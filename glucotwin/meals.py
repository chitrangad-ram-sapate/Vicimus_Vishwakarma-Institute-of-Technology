"""Indian meal library used by the simulator, the twin and the what-if engine.

Values are typical-portion approximations compiled from published Indian food
composition / glycaemic-index tables (IFCT 2017, Sydney GI database, ICMR-NIN).
They are illustrative for a proof of concept, not dietetic advice.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Meal:
    key: str
    name: str
    slot: str          # breakfast | lunch | dinner | snack
    carbs_g: float     # available carbohydrate per typical portion
    gi: float          # glycaemic index (glucose = 100)
    fat_g: float
    protein_g: float
    fiber_g: float
    region: str = "pan"  # north | south | pan

    @property
    def glycaemic_load(self) -> float:
        return self.carbs_g * self.gi / 100.0

    @property
    def bioavailability(self) -> float:
        """Fraction of carbohydrate appearing in plasma (higher GI -> higher, faster)."""
        return 0.55 + 0.4 * self.gi / 100.0

    @property
    def absorption_tau(self) -> float:
        """Gut absorption time constant in minutes; fat, protein and fibre slow absorption."""
        return 35.0 + 0.6 * self.fat_g + 0.3 * self.protein_g + 1.2 * self.fiber_g - 0.15 * (self.gi - 60)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(glycaemic_load=round(self.glycaemic_load, 1))
        return d


MEALS: dict[str, Meal] = {m.key: m for m in [
    # Breakfast
    Meal("idli_sambar", "Idli (4) with sambar", "breakfast", 62, 70, 4, 10, 5, "south"),
    Meal("masala_dosa", "Masala dosa with chutney", "breakfast", 70, 69, 16, 9, 4, "south"),
    Meal("poha", "Poha", "breakfast", 55, 66, 8, 6, 3, "pan"),
    Meal("upma", "Rava upma", "breakfast", 50, 66, 9, 7, 3, "south"),
    Meal("aloo_paratha", "Aloo paratha (2) with curd", "breakfast", 76, 62, 20, 12, 6, "north"),
    Meal("besan_chilla", "Besan chilla (2)", "breakfast", 30, 42, 9, 14, 6, "north"),
    Meal("oats_milk", "Oats porridge with milk", "breakfast", 42, 55, 6, 11, 6, "pan"),
    Meal("bread_omelette", "Bread (2) with omelette", "breakfast", 30, 72, 12, 15, 2, "pan"),
    # Lunch / dinner
    Meal("rice_dal_sabzi", "White rice, dal & sabzi", "lunch", 95, 73, 10, 14, 8, "south"),
    Meal("sambar_rice", "Sambar rice with poriyal", "lunch", 90, 70, 9, 12, 8, "south"),
    Meal("curd_rice", "Curd rice", "lunch", 70, 69, 8, 10, 1, "south"),
    Meal("roti_dal_sabzi", "Roti (3), dal & sabzi", "lunch", 66, 55, 11, 16, 10, "north"),
    Meal("rajma_chawal", "Rajma chawal", "lunch", 92, 60, 9, 17, 13, "north"),
    Meal("chicken_biryani", "Chicken biryani", "lunch", 88, 69, 22, 28, 3, "pan"),
    Meal("chole_bhature", "Chole bhature", "lunch", 85, 68, 30, 16, 9, "north"),
    Meal("jowar_roti_sabzi", "Jowar roti (2), dal & sabzi", "lunch", 58, 50, 9, 14, 11, "pan"),
    Meal("brown_rice_dal", "Brown rice, dal & salad", "lunch", 75, 55, 8, 14, 11, "pan"),
    Meal("ragi_mudde", "Ragi mudde with saaru", "lunch", 68, 60, 5, 9, 9, "south"),
    Meal("veg_pulao_raita", "Veg pulao with raita", "dinner", 80, 66, 12, 10, 5, "pan"),
    Meal("chapati_paneer", "Chapati (3) with paneer sabzi", "dinner", 60, 52, 22, 22, 8, "north"),
    # Snacks
    Meal("chai_biscuits", "Masala chai with biscuits", "snack", 26, 70, 6, 3, 0, "pan"),
    Meal("samosa", "Samosa (1)", "snack", 32, 63, 17, 4, 3, "north"),
    Meal("banana", "Banana (1)", "snack", 27, 51, 0, 1, 3, "pan"),
    Meal("gulab_jamun", "Gulab jamun (2)", "snack", 50, 76, 12, 4, 0, "pan"),
    Meal("roasted_chana", "Roasted chana", "snack", 18, 28, 3, 9, 7, "pan"),
    Meal("sprouts_chaat", "Sprouts chaat", "snack", 16, 32, 2, 8, 6, "pan"),
    Meal("murukku", "Murukku", "snack", 30, 70, 14, 3, 1, "south"),
]}


def by_slot(slot: str, region: str | None = None) -> list[Meal]:
    meals = [m for m in MEALS.values() if m.slot == slot or (slot == "dinner" and m.slot == "lunch")]
    if region is None:
        return meals
    return [m for m in meals if m.region in (region, "pan")]
