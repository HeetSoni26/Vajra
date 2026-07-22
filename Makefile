.PHONY: test lint format tokenizer dataset pretrain-1b pretrain-2b sft dpo eval serve smoke

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

tokenizer:
	python tokenizer/train.py --config configs/tokenizer.yaml

dataset:
	python dataset/run_pipeline.py --config configs/data/preprocessing.yaml

pretrain-1b:
	bash training/launch/launch_pretrain_1b.sh

pretrain-2b:
	bash training/launch/launch_pretrain_2b.sh

sft:
	python training/sft.py --config configs/training/sft.yaml

dpo:
	python training/dpo.py --config configs/training/dpo.yaml

eval:
	python evaluation/run_lm_eval.py --config configs/eval/benchmarks.yaml

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

smoke:
	python scripts/count_parameters.py --config configs/model/model_1b.yaml
