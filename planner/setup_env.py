"""Build (or load from cache) the mock world, the knowledge base and the task splits.

Everything is deterministic given the config, and cached under `cache_dir`, so a
restarted notebook gets byte-identical tasks without re-downloading HotpotQA.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import tasks as T
from . import tools
from .config import Config, get_config
from .corpus import Corpus, load_hotpot_rows
from .persist import read_jsonl, write_jsonl
from .world import World


@dataclass
class PlannerEnv:
    world: World
    corpus: Corpus
    tasks: dict[str, list[dict]]


def _build_corpus(n_train: int, n_val: int, verbose: bool):
    if verbose:
        print(f"Building the knowledge base from HotpotQA ({n_train} train + {n_val} validation questions) ...")
    train_rows, val_rows = load_hotpot_rows(n_train, n_val)
    corpus = Corpus()
    for row in train_rows + val_rows:
        corpus.add_hotpot_row(row)
    return corpus, train_rows, val_rows


def setup_environment(
    cache_dir: str | None = None,
    corpus_path: str | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
    seed: int = 3407,
    verbose: bool = True,
) -> tuple[World, Corpus]:
    """World + corpus only (what the tools need). Used by the demo."""
    defaults = get_config("a100")
    n_train = defaults.hotpot_train_rows if n_train is None else n_train
    n_val = defaults.hotpot_val_rows if n_val is None else n_val
    corpus_path = corpus_path or (os.path.join(cache_dir, "corpus.jsonl") if cache_dir else None)
    if corpus_path and os.path.exists(corpus_path):
        corpus = Corpus.load(corpus_path)
    else:
        corpus, _, _ = _build_corpus(n_train, n_val, verbose)
        if corpus_path:
            corpus.save(corpus_path)
    world = World(seed=seed)
    tools.configure(world=world, corpus=corpus)
    if verbose:
        print(f"Knowledge base: {len(corpus)} passages. World: {len(world.regions)} regions, {len(world.city_names)} cities.")
    return world, corpus


def prepare(config: Config, cache_dir: str, verbose: bool = True) -> PlannerEnv:
    """World, corpus and the three task splits (sft / grpo / eval), cached as JSONL."""
    os.makedirs(cache_dir, exist_ok=True)
    corpus_path = os.path.join(cache_dir, "corpus.jsonl")
    task_paths = {split: os.path.join(cache_dir, f"tasks_{split}.jsonl") for split in ("sft", "grpo", "eval")}
    world = World(seed=config.seed)

    if os.path.exists(corpus_path) and all(os.path.exists(p) for p in task_paths.values()):
        corpus = Corpus.load(corpus_path)
        tasks = {split: read_jsonl(path) for split, path in task_paths.items()}
        if verbose:
            print(f"Loaded cached environment from {cache_dir}")
    else:
        corpus, train_rows, val_rows = _build_corpus(config.hotpot_train_rows, config.hotpot_val_rows, verbose)
        tools.configure(world=world, corpus=corpus)
        pools = T.split_kb_pools(T.kb_items(train_rows, corpus), T.kb_items(val_rows, corpus))
        sizes = {"sft": config.n_sft_per_family, "grpo": config.n_grpo_per_family, "eval": config.n_eval_per_family}
        tasks = {split: T.build_split(split, n, world, pools[split], config.seed) for split, n in sizes.items()}
        corpus.save(corpus_path)
        for split, path in task_paths.items():
            write_jsonl(path, tasks[split])

    tools.configure(world=world, corpus=corpus)
    if verbose:
        print(f"Knowledge base: {len(corpus)} passages")
        for split, items in tasks.items():
            print(f"  {split:5s} {len(items):5d} tasks  {T.family_counts(items)}")
    return PlannerEnv(world=world, corpus=corpus, tasks=tasks)
