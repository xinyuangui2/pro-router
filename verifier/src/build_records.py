"""Stage 1 -- rebuild the evaluation records from public HuggingFace datasets.

The benchmark is a five-source multimodal mix. `verifier/bench_mix_manifest.json`
pins the exact dataset-native record ids used in the paper so the split is
reproducible without redistributing any dataset content; this script resolves
those ids against the public sources and writes one JSONL per split plus the
decoded images.

    python src/build_records.py --out-dir runs/records

Requires HF_TOKEN for the gated sources. `--dry-run` reports per-source
manifest coverage without downloading anything.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

# source key -> (HF dataset repo, config, split, how to derive the manifest id)
SOURCES = {
    "chartqa":   ("lmms-lab/ChartQA",  None,          "test",       "index"),
    "docvqa":    ("lmms-lab/DocVQA",   "DocVQA",      "validation", "index"),
    "mathvista": ("AI4Math/MathVista", None,          "testmini",   "pid"),
    "mmbench":   ("lmms-lab/MMBench_EN", None,        "dev",        "index"),
    "mmmu":      ("MMMU/MMMU",         "all",         "validation", "native"),
}

QUESTION_KEYS = ["question", "query", "prompt", "hint"]
ANSWER_KEYS = ["answer", "answers", "gold", "label", "solution"]


def _first(row: dict, keys: list[str]):
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            return row[k]
    return None


def _load_split(repo: str, config: str | None, hf_split: str):
    """Load one split, tolerating hubs that no longer expose an "all" config.

    MMMU used to ship a combined `all` config; it now exposes one config per
    subject, so `all` is resolved by concatenating every subject config. Record
    ids are dataset-native there, so concatenation order does not matter.
    """
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

    if config is None:
        return load_dataset(repo, split=hf_split)
    try:
        return load_dataset(repo, config, split=hf_split)
    except ValueError:
        if config != "all":
            raise
        subjects = get_dataset_config_names(repo)
        print(f"[build]   '{repo}' has no 'all' config; concatenating "
              f"{len(subjects)} subject configs", flush=True)
        return concatenate_datasets(
            [load_dataset(repo, s, split=hf_split) for s in subjects])


def _manifest_id(src: str, row: dict, idx: int) -> str:
    """Reproduce the id scheme recorded in the manifest for each source."""
    mode = SOURCES[src][3]
    if mode == "index":
        return f"{src}_{idx}"
    if mode == "pid":
        # The manifest stores pid-derived ids source-prefixed, exactly as the
        # index-derived ones are ("mathvista_9", not "9").
        return f"{src}_{row.get('pid', idx)}"
    return str(row.get("id", idx))


def _images(row: dict) -> list:
    """Every bitmap attached to a record, in a stable order.

    `decoded_image` and `image` are alternatives for the same slot -- some
    sources put a filename in `image` and the bitmap in `decoded_image` -- so
    the first that yields a bitmap wins. Panel fields are additive: a
    multi-image question carries `image_1`, `image_2`, ... and dropping the
    later ones would silently change the question.
    """
    out = []
    for k in ("decoded_image", "image"):
        v = row.get(k)
        if v is None:
            continue
        cand = [im for im in (v if isinstance(v, list) else [v])
                if hasattr(im, "convert")]
        if cand:
            out.extend(cand)
            break
    for k in ("images", "image_1", "image_2", "image_3", "image_4",
              "image_5", "image_6", "image_7"):
        v = row.get(k)
        if v is None:
            continue
        out.extend(im for im in (v if isinstance(v, list) else [v])
                   if hasattr(im, "convert"))
    seen, uniq = set(), []
    for im in out:
        if id(im) not in seen:
            seen.add(id(im))
            uniq.append(im)
    return uniq


MCQ_LETTERS = "ABCDEFGHIJ"

# Each source states how terse the answer must be. This is load-bearing, not
# cosmetic: the head reads a decoding trajectory, so answer length sets the
# regime it sees. Free-form prompting on the same questions produces long
# answers that collide with the generation cap, and a truncated trajectory
# reads as confident. These instructions keep answers to a token or two.
PROMPT_STYLE = {
    "chartqa": ("Look at the chart and answer the question. Provide only the "
                "answer, no explanation.\n\nQuestion: {q}\n\nAnswer:"),
    "docvqa": ("Look at the document image and answer the question. Provide "
               "only the answer, no explanation.\n\nQuestion: {q}\n\nAnswer:"),
    "mmbench": ("Look at the image and answer the multiple-choice question "
                "with a single letter (A, B, C, or D). Output only the letter "
                "and nothing else.\n\nQuestion: {q}\n\n{opts}\n\nAnswer:"),
    "mmmu": ("Look at the provided image(s) and answer the multiple-choice "
             "question with a single letter (A through J). Output only the "
             "letter and nothing else.\n\nQuestion: {q}\n\n{opts}\n\nAnswer:"),
    # MathVista carries both multiple-choice and free-form questions, so it
    # needs one template per kind. Its earlier single template ended with
    # "provide the correct option letter ... at the end", which invites the
    # model to reason first and answer last: every generation that hit the
    # token cap in the reported runs came from this one source, while the other
    # four answered in a single token. These two match the shapes above.
    "mathvista_choice": (
        "Look at the image and answer the multiple-choice question with a "
        "single letter (A through J). Output only the letter and nothing "
        "else.\n\nQuestion: {q}\n\n{opts}\n\nAnswer:"),
    "mathvista_free": (
        "Look at the image and answer the question. Provide only the answer, "
        "no explanation.\n\nQuestion: {q}\n\nAnswer:"),
}


def _options(row: dict) -> list:
    """The answer choices, across the three ways the sources encode them.

    A multiple-choice prompt whose choices are missing is unanswerable as
    posed, so each encoding is handled explicitly rather than falling through
    to an empty list:

      * a real list/tuple in `options` / `choices`  (some sources)
      * a *stringified* list -- MMMU ships "['132,625', '134,485', ...]",
        which is a Python literal, not JSON, so json.loads rejects the single
        quotes and ast.literal_eval is needed
      * one column per letter -- MMBench has A/B/C/D columns and no options
        field at all, using the string "nan" for choices that do not exist
    """
    opts = row.get("options") or row.get("choices")
    if isinstance(opts, str):
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(opts)
                break
            except Exception:
                continue
        opts = parsed
    if isinstance(opts, (list, tuple)) and opts:
        return list(opts)

    lettered = []
    for letter in MCQ_LETTERS:
        v = row.get(letter)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none"):
            continue
        lettered.append(s)
    return lettered


def _style_key(src: str, opts: list) -> str:
    """Which PROMPT_STYLE entry applies. MathVista splits on question kind."""
    if src == "mathvista":
        return "mathvista_choice" if opts else "mathvista_free"
    return src


def _build_prompt(src: str, row: dict) -> str:
    q = (_first(row, QUESTION_KEYS) or "").strip()
    opts = _options(row)
    block = "\n".join(f"{MCQ_LETTERS[i]}. {o}" for i, o in enumerate(opts))
    style = PROMPT_STYLE.get(_style_key(src, opts))
    if style is None:
        return q if not block else f"{q}\n{block}"
    return style.format(q=q, opts=block)


def _gold(row: dict, opts: list | None = None) -> str:
    """The reference answer, in the same form the prompt asks for.

    Multiple-choice sources disagree on this. MMBench and MMMU store the option
    *letter*, which is what a letter-only prompt produces. MathVista stores the
    option *text* ("145°", not "A"), so a letter-only prompt could never be
    compared against it. When the gold text is one of the offered choices it is
    mapped to that choice's letter, making every multiple-choice source agree.

    Falls back to the raw value when the answer is not among the choices, so a
    formatting mismatch upstream degrades to the old behaviour instead of
    inventing a letter.
    """
    g = _first(row, ANSWER_KEYS)
    if isinstance(g, (list, tuple)):
        g = g[0] if g else ""
    g = str(g).strip()
    if opts and g and g.upper() not in set(MCQ_LETTERS):
        norm = [str(o).strip() for o in opts]
        if g in norm:
            return MCQ_LETTERS[norm.index(g)]
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",
                    default=str(Path(__file__).resolve().parents[1]
                                / "bench_mix_manifest.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report manifest coverage per source and exit.")
    ap.add_argument("--max-images", type=int, default=2,
                    help="Images kept per record; must not exceed the "
                         "engine's per-prompt image limit.")
    ap.add_argument("--allow-index-drift", action="store_true",
                    help="Emit positionally-keyed records even when the "
                         "upstream split has been resized (the positions will "
                         "point at different records than the manifest meant).")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    out_dir = Path(args.out_dir)
    if not args.dry_run:
        # imported here so --dry-run can validate the manifest with no extra deps
        from PIL import Image  # noqa: F401  (ensures the decoder is available)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

    wanted: dict[str, dict[str, str]] = {}
    for split in args.splits.split(","):
        for rec in man["records"][split]:
            wanted.setdefault(rec["source"], {})[rec["id"]] = split

    rows_by_split: dict[str, list] = {s: [] for s in args.splits.split(",")}
    for src, id_to_split in sorted(wanted.items()):
        repo, config, hf_split, _ = SOURCES[src]
        print(f"[build] {src}: {len(id_to_split)} ids from {repo}", flush=True)
        if args.dry_run:
            continue
        ds = _load_split(repo, config, hf_split)
        # Index-derived ids are positions, not content keys: if the upstream
        # split has been resized, a position still "resolves" but points at a
        # different record. Refuse rather than emit silently-wrong data.
        if SOURCES[src][3] == "index":
            hi = max(int(i.rsplit("_", 1)[1]) for i in id_to_split)
            if hi >= len(ds):
                print(f"[build]   ERROR: manifest indices reach {hi} but the "
                      f"upstream split now has {len(ds)} rows. Positional ids "
                      f"cannot be trusted against a resized split -- skipping "
                      f"{src}. (Pass --allow-index-drift to emit anyway.)",
                      flush=True)
                if not args.allow_index_drift:
                    continue
        hit = truncated = 0
        for idx, row in enumerate(ds):
            rid = _manifest_id(src, row, idx)
            split = id_to_split.get(rid)
            if split is None:
                continue
            imgs = _images(row)
            if len(imgs) > args.max_images:
                truncated += 1
                imgs = imgs[: args.max_images]
            paths = []
            for j, im in enumerate(imgs):
                p = out_dir / "images" / f"{src}_{rid}_{j}.jpg"
                if not p.exists():
                    im.convert("RGB").save(p, quality=92)
                # Absolute, because these records are read by engines running in
                # another process -- and on a Ray cluster, another node with a
                # different cwd. A relative path resolves against whatever cwd
                # the actor happens to have and the engine fails with
                # FileNotFoundError on the first image it tries to load.
                paths.append(str(p.resolve()))
            row_opts = _options(row)
            rows_by_split[split].append({
                "id": rid, "source": src,
                "prompt": _build_prompt(src, row), "images": paths,
                "gold": _gold(row, row_opts),
            })
            hit += 1
        print(f"[build]   resolved {hit}/{len(id_to_split)}", flush=True)
        if truncated:
            print(f"[build]   note: {truncated} record(s) had more than "
                  f"{args.max_images} images and were truncated", flush=True)
        if hit < len(id_to_split):
            print(f"[build]   WARNING: {len(id_to_split) - hit} ids unresolved "
                  f"-- the upstream dataset may have been revised.", flush=True)

    if args.dry_run:
        return 0
    for split, rows in rows_by_split.items():
        out = out_dir / f"bench_{split}.jsonl"
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[build] wrote {out} ({len(rows)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
