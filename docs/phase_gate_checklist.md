# Phase Gate Checklist

## Phase 0 gate
- [ ] environment reproducible
- [ ] CPU tests pass
- [ ] GPU smoke test passes
- [ ] CI configured

## Phase 1 gate
- [ ] tokenizer trained
- [ ] tokenizer report complete
- [ ] tokenizer artifacts frozen

## Phase 2 gate
- [ ] data sources audited
- [ ] filters/dedup/contamination checks complete
- [ ] tokenized shards checksummed
- [ ] manifest complete

## Phase 3 gate
- [ ] model tests pass
- [ ] parameter count verified
- [ ] debug model overfits toy data
- [ ] checkpoint restore works

## Phase 4 gate
- [ ] pretraining completed or milestone accepted
- [ ] loss curves reviewed
- [ ] base benchmarks complete

## Phase 5 gate
- [ ] SFT complete
- [ ] DPO complete if releasing aligned model
- [ ] instruction and safety eval complete

## Phase 6 gate
- [ ] benchmark report complete
- [ ] quantization complete
- [ ] throughput/latency measured

## Phase 7 gate
- [ ] HF release validates
- [ ] GGUF/Ollama validate
- [ ] Docker/API validate
- [ ] SDK validates
- [ ] final docs complete
