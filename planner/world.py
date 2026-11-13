"""Seeded synthetic world behind `get_weather` and `get_election`.

Every region, city and candidate is fictional, so the model cannot answer from
memory and `knowledge_search` cannot find them: the specialised tool is the only
way to get the fact. Each region has a capital that is a city in the weather
database, which makes cross-tool multi-hop questions possible.

Everything is a pure function of (seed, entity, date/year), so the world can be
rebuilt identically in training, evaluation and the demo.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import random
from dataclasses import dataclass, field

ELECTION_YEARS = (2016, 2020, 2024)
WEATHER_START = _dt.date(2016, 1, 1)
WEATHER_END = _dt.date(2025, 12, 31)
CONDITIONS = ("sunny", "cloudy", "rainy", "stormy", "snowy", "foggy", "windy")
WET_CONDITIONS = ("rainy", "stormy", "snowy")
PARTIES = (
    "Unity Party",
    "Green Alliance",
    "Civic Front",
    "Progress League",
    "Heritage Union",
    "Liberty Coalition",
)

_PREFIX = (
    "Nor", "Vel", "Ash", "Bren", "Cald", "Dor", "Eld", "Fen", "Gal", "Har",
    "Isk", "Jor", "Kel", "Lun", "Mar", "Nev", "Orl", "Pell", "Quen", "Rav",
    "Sel", "Tor", "Ulm", "Var", "Wyn", "Yar", "Zan", "Bel", "Cor", "Dun",
    "Esk", "Fal", "Gren", "Hol", "Ivr", "Kar", "Lor", "Mel", "Oss", "Tav",
)
_REGION_SUFFIX = ("ia", "land", "mark", "shire", "ova", "esia", "onia", "stan", "gard", "wick")
_CITY_SUFFIX = ("haven", "ford", "burg", "mouth", "field", "port", "stead", "ton", "vale", "crest", "bridge", "holm")
_FIRST = (
    "Alda", "Bram", "Cira", "Dain", "Elna", "Faro", "Greta", "Hale", "Ilsa", "Joss",
    "Kira", "Lev", "Mira", "Nils", "Oda", "Pavel", "Rhea", "Soren", "Tessa", "Ulric",
    "Vera", "Wim", "Xenia", "Yorick", "Zelda", "Anselm", "Brisa", "Corin", "Delia", "Emeric",
)
_LAST = (
    "Avery", "Brandt", "Calder", "Drummond", "Ellery", "Fenwick", "Galloway", "Hartwell",
    "Ingram", "Jarrow", "Kessler", "Lindqvist", "Marlowe", "Northcote", "Okafor", "Penrose",
    "Quill", "Rosendahl", "Stroud", "Thackeray", "Underhill", "Voss", "Wexford", "Yardley",
    "Zeller", "Ashdown", "Birkett", "Cromwell", "Dunmore", "Everly",
)


def _unit(*parts: object) -> float:
    """Deterministic float in [0, 1) from the given parts (stable across processes)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def canon(name: object) -> str:
    """Canonical form of an entity name for lookups and argument comparison."""
    return " ".join(str(name).strip().lower().split())


def parse_date(value: object) -> _dt.date:
    """Parse a strict ISO `YYYY-MM-DD` date, raising ValueError otherwise."""
    text = str(value).strip()
    try:
        return _dt.date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD, got {text!r}") from exc


@dataclass(frozen=True)
class Region:
    name: str
    capital: str
    cities: tuple[str, ...]
    split: str


@dataclass
class World:
    """The fictional regions and cities plus their weather and election records."""

    seed: int = 3407
    n_train_regions: int = 60
    n_eval_regions: int = 30
    cities_per_region: int = 3
    regions: dict[str, Region] = field(default_factory=dict, init=False)
    city_region: dict[str, str] = field(default_factory=dict, init=False)
    city_names: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        n_regions = self.n_train_regions + self.n_eval_regions
        region_pool = [p + s for p in _PREFIX for s in _REGION_SUFFIX]
        city_pool = [p + s for p in _PREFIX for s in _CITY_SUFFIX]
        rng.shuffle(region_pool)
        rng.shuffle(city_pool)
        need_cities = n_regions * self.cities_per_region
        if n_regions > len(region_pool) or need_cities > len(city_pool):
            raise ValueError("world is larger than the available name pools")
        for i in range(n_regions):
            cities = tuple(city_pool[i * self.cities_per_region : (i + 1) * self.cities_per_region])
            split = "train" if i < self.n_train_regions else "eval"
            region = Region(name=region_pool[i], capital=cities[0], cities=cities, split=split)
            self.regions[canon(region.name)] = region
            for city in cities:
                self.city_region[canon(city)] = canon(region.name)
                self.city_names[canon(city)] = city

    # ------------------------------------------------------------------ entities
    def region_list(self, split: str) -> list[Region]:
        return [r for r in self.regions.values() if r.split == split]

    def city_list(self, split: str) -> list[str]:
        return [c for r in self.region_list(split) for c in r.cities]

    # ------------------------------------------------------------------- weather
    def weather(self, city: object, date: object) -> dict:
        key = canon(city)
        if key not in self.city_names:
            raise ValueError(f"unknown city {str(city)!r}")
        day = parse_date(date)
        if not (WEATHER_START <= day <= WEATHER_END):
            raise ValueError(f"no weather record for {day.isoformat()}: supported range is {WEATHER_START} to {WEATHER_END}")
        climate = -4.0 + 30.0 * _unit(self.seed, "climate", key)
        season = 9.0 * math.sin(2 * math.pi * (day.timetuple().tm_yday - 105) / 365.25)
        noise = 10.0 * (_unit(self.seed, "temp", key, day.isoformat()) - 0.5)
        temp = round(climate + season + noise, 1)
        condition = CONDITIONS[int(_unit(self.seed, "cond", key, day.isoformat()) * len(CONDITIONS))]
        if condition == "snowy" and temp > 3:
            condition = "rainy"
        precip = round(1.0 + 39.0 * _unit(self.seed, "precip", key, day.isoformat()), 1) if condition in WET_CONDITIONS else 0.0
        return {
            "city": self.city_names[key],
            "date": day.isoformat(),
            "temp_c": temp,
            "condition": condition,
            "precip_mm": precip,
        }

    # ------------------------------------------------------------------ election
    def election(self, region: object, year: object) -> dict:
        key = canon(region)
        if key not in self.regions:
            raise ValueError(f"unknown region {str(region)!r}")
        try:
            year_int = int(str(year).strip())
        except ValueError as exc:
            raise ValueError(f"year must be an integer, got {year!r}") from exc
        if year_int not in ELECTION_YEARS:
            raise ValueError(f"no election in {year_int}: elections were held in {', '.join(map(str, ELECTION_YEARS))}")
        reg = self.regions[key]
        rng = random.Random(f"{self.seed}|election|{key}|{year_int}")
        winner, runner_up = rng.sample([f"{f} {l}" for f in _FIRST for l in _LAST], 2)
        winner_party, runner_party = rng.sample(PARTIES, 2)
        winner_share = round(rng.uniform(38.0, 62.0), 1)
        runner_share = round(rng.uniform(20.0, min(winner_share - 1.0, 100.0 - winner_share - 1.0)), 1)
        day = _dt.date(year_int, rng.choice((3, 4, 5, 9, 10, 11)), rng.randint(1, 28))
        return {
            "region": reg.name,
            "year": year_int,
            "winner": winner,
            "winner_party": winner_party,
            "winner_vote_share_pct": winner_share,
            "runner_up": runner_up,
            "runner_up_party": runner_party,
            "runner_up_vote_share_pct": runner_share,
            "turnout_pct": round(rng.uniform(45.0, 85.0), 1),
            "election_date": day.isoformat(),
            "capital": reg.capital,
        }
