This submission captures a small-auxiliary-network variant of the baseline that replaces the previous full-width auxiliary transformer with a bottleneck editor.

It is a useful snapshot because it improves the compressed model size materially while keeping baseline-like training behavior, but it is still a near-miss on the true decimal artifact cap. The logged `int8+zlib` model file is under 16MB by itself, but `code + model` lands slightly over the `16,000,000` byte submission limit, so this should be treated as a non-record / in-progress submission snapshot rather than a compliant leaderboard entry.

Configuration highlights:
- Baseline body kept intact: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Auxiliary editor changed from two full `Block`s to a bottleneck MLP editor with `AUX_DIM=64`
- Batch/sequence length unchanged: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- 10-minute wallclock cap on `8xH100`

Command used:
```bash
RUN_ID=baseline_8xh100_20260322_141006 \
SEED=12 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics from `train.log`:
- Timed training stopped at `13188/20000` steps due to the wallclock cap
- Pre-quant eval at stop: `val_loss:2.0663`, `val_bpb:1.2238`
- Post-quant roundtrip eval: `val_loss:2.0672`, `val_bpb:1.2243`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_loss:2.06717489 val_bpb:1.22429741`
- Train time: `600052ms` with `step_avg:45.50ms`
- Peak memory: `10694 MiB allocated`, `10896 MiB reserved`
- Serialized model int8+zlib: `15994724 bytes`
- Code size: `55890 bytes`
- Total submission size int8+zlib: `16050614 bytes`
- Amount over cap: `50614 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `6914310144`

Included files:
- `train_gpt.py` - exact code snapshot used for the run
- `train.log` - full 8xH100 training log
- `submission.json` - metadata for the run
