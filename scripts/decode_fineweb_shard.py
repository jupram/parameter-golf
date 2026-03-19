#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm


HEADER_BYTES = 256 * np.dtype("<i4").itemsize


def load_tokens(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=HEADER_BYTES)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}: expected {num_tokens} tokens, got {tokens.size}")
    return tokens.astype(np.int32, copy=False)


def decode_ids(sp: spm.SentencePieceProcessor, ids: list[int]) -> str:
    if hasattr(sp, "decode_ids"):
        return sp.decode_ids(ids)
    if hasattr(sp, "decode"):
        return sp.decode(ids)
    pieces = [sp.id_to_piece(i) for i in ids]
    return "".join(pieces).replace("▁", " ").lstrip()


def split_docs(tokens: np.ndarray, bos_id: int, eos_id: int) -> list[list[int]]:
    docs: list[list[int]] = []
    current: list[int] = []

    for token in tokens.tolist():
        if token == bos_id:
            if current:
                docs.append(current)
            current = [token]
            continue
        if not current:
            continue
        current.append(token)

    if current:
        docs.append(current)

    cleaned_docs: list[list[int]] = []
    for doc in docs:
        if doc and doc[0] == bos_id:
            doc = doc[1:]
        if doc and eos_id >= 0 and doc[-1] == eos_id:
            doc = doc[:-1]
        if doc:
            cleaned_docs.append(doc)
    return cleaned_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode the first few docs from a FineWeb .bin shard")
    parser.add_argument("shard", type=Path, help="Path to a fineweb_train_*.bin or fineweb_val_*.bin shard")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("./data/tokenizers/fineweb_1024_bpe.model"),
        help="SentencePiece tokenizer model used to produce the shard",
    )
    parser.add_argument("--max-docs", type=int, default=3, help="Number of documents to print")
    parser.add_argument("--max-chars", type=int, default=1200, help="Truncate each decoded doc to this many characters")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    if bos_id < 0:
        raise ValueError("Tokenizer does not define a BOS token")

    tokens = load_tokens(args.shard)
    docs = split_docs(tokens, bos_id=bos_id, eos_id=eos_id)

    print(f"shard: {args.shard}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"docs_found: {len(docs)}")
    print()

    for idx, doc in enumerate(docs[: args.max_docs], start=1):
        text = decode_ids(sp, doc)
        if len(text) > args.max_chars:
            text = text[: args.max_chars] + "..."
        print(f"--- doc {idx} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
