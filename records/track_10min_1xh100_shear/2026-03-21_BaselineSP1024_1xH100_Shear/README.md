This record captures a 10-minute single-GPU baseline run for the custom `track_10min_1xh100_shear` track.

This run uses the current root `train_gpt.py` on `1xH100` with the published `fineweb10B_sp1024` dataset/tokenizer and only the minimal README-style environment overrides (`RUN_ID`, `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE`). All other settings come from the defaults in `train_gpt.py`.

Configuration:
- Track: `track_10min_1xh100_shear`
- Hardware: `1x NVIDIA H100 80GB HBM3`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Effective single-GPU accumulation: `WORLD_SIZE=1`, `GRAD_ACCUM_STEPS=8`
- Defaults inherited from `train_gpt.py`: `MAX_WALLCLOCK_SECONDS=600 WARMUP_STEPS=20 VAL_LOSS_EVERY=1000 ENABLE_COMPILE=1`

Command (track-relevant params):
```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 /workspace/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `1146/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.2625`, `val_bpb:1.3400`
- Post-quant roundtrip eval: `val_loss:2.2652`, `val_bpb:1.3416`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.34156706`
- Train time: `600417ms` (`step_avg:523.92ms`)
- Peak memory: `10306 MiB allocated`, `10684 MiB reserved`
- Serialized model int8+zlib: `13022501 bytes`
- Code size: `52783 bytes`
- Total submission size int8+zlib: `13075284 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `600834048`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact training log)
- `submission.json` (track metadata)
