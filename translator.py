#!/usr/bin/env python3
"""
Generic resume-safe translator:
- Translates DEFAULT_COL -> TARGET_COL ONLY when EXAMPLE_COL is non-empty
  and TARGET_COL is empty.
- Preserves HTML or json formatting.
- Writes progress after each batch, so it can resume if interrupted.

Configurable via constants below or env vars.
"""

import os
import sys
import time
import pandas as pd
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- Config (can be overridden with env vars) ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")  # model to use
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "40"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
BASE_SLEEP = float(os.getenv("BASE_SLEEP", "2.0"))
SEP = "\n\n---\n\n"  # separator between items

DEFAULT_COL = os.getenv("DEFAULT_COL", "Default content")
EXAMPLE_COL = os.getenv("EXAMPLE_COL", "Translated content")
TARGET_COL = os.getenv("TARGET_COL", "New content")

DEFAULT_LANG = os.getenv("DEFAULT_LANG", "English")
TARGET_LANG = os.getenv("TARGET_LANG", "French")
# --------------------------------------------------------------


def die(msg: str, code: int = 1):
    print(f"[fatal] {msg}", file=sys.stderr)
    sys.exit(code)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def write_csv_atomic(df: pd.DataFrame, out_path: str):
    temp_path = out_path + ".tmp"
    df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, out_path)


def must_have_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"Missing required columns: {missing}")


def build_prompt(texts: List[str]) -> str:
    header = (
        f"You are a professional translator specialized in sports eyewear and outdoor gear. "
        f"Translate the following {DEFAULT_LANG} snippets into natural, compelling {TARGET_LANG}. "
        "Preserve ALL JSON structure exactly (do not add, remove, or reorder structure). "
        "Preserve HTML tags and attributes exactly when pertaining to formatting such as tables, lists, ... "
        "When encountering HTML entities used for emphasizing text, you may reuse them when appropriate in the translated version. "
        "Do not include explanations. Output ONLY the translations, joined by the exact separator below.\n\n"
        "Do not translate anything related to the HTML or JSON structure itself, such as tag names, attribute names, or JSON keys. "
        f"Separator to use between items (exactly): {SEP!r}\n\n"
        "Input items start below:\n"
    )
    return header + SEP.join(texts)


def translate_batch(client: OpenAI, texts: List[str]) -> List[str]:
    if not texts:
        return []
    prompt = build_prompt(texts)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=0.2,
            )
            out = resp.output_text.strip()
            parts = [p.strip() for p in out.split(SEP)]
            if len(parts) != len(texts):
                parts = [p for p in parts if p]  # drop empties if trailing sep
            if len(parts) != len(texts):
                raise ValueError(f"Item count mismatch: sent {len(texts)}, got {len(parts)}")
            return parts
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            sleep_s = BASE_SLEEP * (2 ** (attempt - 1))
            print(
                f"[warn] Batch failed ({attempt}/{MAX_RETRIES}): {e}. Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)


def ensure_target_last(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if cols[-1] != TARGET_COL:
        cols = [c for c in cols if c != TARGET_COL] + [TARGET_COL]
        df = df[cols]
    return df


def main():
    if len(sys.argv) != 3:
        die("Usage: python translate_generic.py /path/in.csv /path/out.csv")

    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        die(f"Input file not found: {in_path}")

    src = read_csv(in_path)
    must_have_columns(src, [DEFAULT_COL, EXAMPLE_COL])

    if os.path.exists(out_path):
        print(f"[info] Resuming from existing output: {out_path}")
        out = read_csv(out_path)
        if len(out) != len(src):
            print("[warn] Output length differs from input, aligning columns.")
            for col in src.columns:
                if col not in out.columns:
                    out[col] = src[col]
    else:
        out = src.copy()

    if TARGET_COL not in out.columns:
        out[TARGET_COL] = ""
    out[TARGET_COL] = out[TARGET_COL].astype(str).fillna("")

    example = out[EXAMPLE_COL].fillna("").astype(str).str.strip()
    target = out[TARGET_COL].fillna("").astype(str).str.strip()
    todo_mask = (example != "") & (target == "")
    idx_to_translate = out.index[todo_mask].tolist()

    print(f"[info] Total rows: {len(out)}")
    print(f"[info] Rows requiring translation: {len(idx_to_translate)}")

    if not idx_to_translate:
        out = ensure_target_last(out)
        write_csv_atomic(out, out_path)
        print(f"[done] Nothing to do. Wrote: {out_path}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        die("OPENAI_API_KEY not set in environment.")
    client = OpenAI()

    total = len(idx_to_translate)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_idx = idx_to_translate[start:end]
        texts = out.loc[batch_idx, DEFAULT_COL].fillna("").astype(str).tolist()

        nonempty_pairs = [(i, t) for i, t in enumerate(texts) if t.strip() != ""]
        order_indices = [i for (i, _) in nonempty_pairs]
        nonempty_texts = [t for (_, t) in nonempty_pairs]

        translations = []
        if nonempty_texts:
            translations = translate_batch(client, nonempty_texts)

        full_results = [""] * len(texts)
        for pos, trans in zip(order_indices, translations):
            full_results[pos] = trans

        for row_i, tgt_text in zip(batch_idx, full_results):
            out.at[row_i, TARGET_COL] = tgt_text

        out = ensure_target_last(out)
        write_csv_atomic(out, out_path)

        print(f"[info] Saved progress {start + 1}-{end} / {total}")

    print(f"[done] All required rows translated. Output: {out_path}")


if __name__ == "__main__":
    main()
