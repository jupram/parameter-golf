# Non-Record Submission: AuxBoundaryEdit Baseline

## Score: mean val_bpb = 1.2257 (3 seeds: 1.2252, 1.2258, 1.2262)

Trained on 8xH100 SXM for 600 seconds. All three runs fit under the 16MB artifact cap after the built-in int8+zlib roundtrip, with total submission sizes between 15.90MB and 15.92MB.

This is a non-record submission. The result is slightly worse than the published naive baseline, but the run is still interesting because it isolates a small auxiliary network that tries to inject tokenizer-boundary information into next-token prediction at very low serialized cost.

## What the aux network is trying to do

SentencePiece marks the start of a new word with a leading-space marker (`▁`). That means the next-token distribution changes a lot depending on whether the next token is likely to begin a new word or continue the current one.

The aux network tries to exploit that signal in two coupled ways:

1. It predicts a binary target: whether the ground-truth next token has a leading space (`has_leading_space_lut[target_id]`).
2. It uses the same low-dimensional features to generate a small residual edit in embedding space, projects that edit through the tied embedding matrix, and adds the resulting correction directly to the LM logits.

In code, this is the `AuxNet` module:

- `512 -> 32` bottleneck (`AUX_DIM=32`)
- tiny 32-dim MLP editor
- `32 -> 512` up-projection back into model space
- a separate 1-logit boundary classifier trained with BCE
- `edit_scale` initialized to zero so the whole module starts as a no-op

Conceptually, it is trying to learn a cheap "word-boundary prior." If the hidden state suggests the next piece is probably a word start, the residual logit edit can softly favor tokens whose tied embeddings line up with that direction; if it looks like a continuation piece, it can bias the other way. The auxiliary BCE loss encourages the bottleneck to carry exactly that boundary information.

This is not a second full language-model head. It is a very small low-rank correction path, intended to buy some tokenizer-aware structure without paying for a large extra projection.

## Configuration

This record snapshots the repository's `train_gpt.py` as of this run, launched via `scripts/baseline_8xh100.sh` with different `SEED` values. Relevant hyperparameters from the logs:

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `TRAIN_BATCH_TOKENS=524288`
- `TRAIN_SEQ_LEN=1024`
- `AUX_LOSS_WEIGHT=0.1`
- `AUX_DIM=32`
- `TIED_EMBED_LR=0.05`
- `MATRIX_LR=0.03`
- `SCALAR_LR=0.03`
- `HEAD_LR=0.0`
- `MAX_WALLCLOCK_SECONDS=600`

Representative command:

```bash
SEED=42 bash scripts/baseline_8xh100.sh
```

## Reproducibility

Three independent training runs with different seeds:

| Seed | val_loss | val_bpb | step_stop | bytes_total |
|------|----------|---------|-----------|-------------|
| 42 | 2.06974043 | 1.22581687 | 13217 | 15,900,186 |
| 1227 | 2.07040067 | 1.22620790 | 13232 | 15,924,267 |
| 2000 | 2.06864145 | 1.22516599 | 13243 | 15,924,165 |
| **Mean** | **2.06959418** | **1.22573025** | **13230.67** | **15,916,206** |
| **Std** | **0.00089** | **0.00053** | - | - |

Additional run metrics:

- Mean wallclock: `600.041s`
- Peak memory: `10666 MiB allocated`
- Model params: `17,227,624`
- Code size: `55,890 bytes`
- Worst-case serialized model int8+zlib size: `15,868,377 bytes`

## Included files

- `train_gpt.py` - exact code snapshot used for the runs
- `train_seed42.log`
- `train_seed1227.log`
- `train_seed2000.log`
- `submission.json`
