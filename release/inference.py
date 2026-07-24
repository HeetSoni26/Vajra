from pathlib import Path


class InferenceExamplesGenerator:
    """Generates standard inference scripts and examples for the released model."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "examples"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self):
        self._generate_python_basic()
        self._generate_cli()
        self._generate_streaming()
        self._generate_batch()

    def _generate_python_basic(self):
        code = """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "../" # Path to the release directory
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
        (self.output_dir / "basic_inference.py").write_text(code, encoding="utf-8")

    def _generate_cli(self):
        code = """import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="../")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()
"""
        (self.output_dir / "cli_inference.py").write_text(code, encoding="utf-8")

    def _generate_streaming(self):
        code = """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "../"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
streamer = TextStreamer(tokenizer)

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

_ = model.generate(**inputs, streamer=streamer, max_new_tokens=100)
"""
        (self.output_dir / "streaming_inference.py").write_text(code, encoding="utf-8")

    def _generate_batch(self):
        code = """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "../"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompts = ["Hello world", "What is the capital of France?"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)
for i, out in enumerate(outputs):
    print(f"Result {i}:", tokenizer.decode(out, skip_special_tokens=True))
"""
        (self.output_dir / "batch_inference.py").write_text(code, encoding="utf-8")
