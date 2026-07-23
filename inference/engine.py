"""Production inference engine for FoundationLM with KV cache, sampling, and streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import torch
import yaml

from inference.kv_cache import KVCache
from model import FoundationLM, ModelConfig
from training.checkpoint import load_checkpoint
from utils.environment import get_device
from utils.logging import setup_logger

logger = setup_logger("inference_engine")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True
    seed: int | None = None
    stop_tokens: list[int] = field(default_factory=list)
    use_kv_cache: bool = True


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Penalise tokens that have already been generated."""
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, generated_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, generated_ids, score)
    return logits


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k logits, setting the rest to -inf."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    top_k_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_vals[:, -1:]
    logits[logits < threshold] = float("-inf")
    return logits


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[mask] = float("-inf")
    # Scatter back
    logits.scatter_(1, sorted_indices, sorted_logits)
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    gen_cfg: GenerationConfig,
    generated_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample or greedily select the next token from logits."""
    # Repetition penalty
    if generated_ids is not None and gen_cfg.repetition_penalty != 1.0:
        logits = _apply_repetition_penalty(logits, generated_ids, gen_cfg.repetition_penalty)

    if not gen_cfg.do_sample or gen_cfg.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / gen_cfg.temperature
    logits = _apply_top_k(logits, gen_cfg.top_k)
    logits = _apply_top_p(logits, gen_cfg.top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class InferenceEngine:
    """Production-grade inference engine with KV cache, AMP, and streaming."""

    def __init__(
        self,
        model: FoundationLM,
        tokenizer: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config_path: str | Path,
        tokenizer_path: str | Path,
        device: torch.device | str | None = None,
        precision: str = "fp32",
    ) -> "InferenceEngine":
        """Load model from a training checkpoint (.pt file)."""
        from tokenizers import Tokenizer

        model_cfg = ModelConfig.from_yaml(model_config_path)
        model = FoundationLM(model_cfg)

        # Load checkpoint weights (skip if path is empty or doesn't point to a file)
        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        if ckpt_path is not None and ckpt_path.is_file():
            load_checkpoint(ckpt_path, model)
            logger.info(f"Loaded checkpoint from {ckpt_path}")

        # Tokenizer path resolution
        tok_path = Path(tokenizer_path)
        candidates = [
            tok_path,
            tok_path / "tokenizer.json",
            tok_path / "v1.0" / "tokenizer.json",
            Path("tokenizer/v1.0/tokenizer.json"),
            Path("tokenizer/tokenizer.json"),
        ]
        tok_file = None
        for c in candidates:
            if c.is_file():
                tok_file = c
                break

        if tok_file is None:
            raise FileNotFoundError(
                f"Could not find tokenizer.json in candidate paths: {[str(c) for c in candidates]}"
            )

        tokenizer = Tokenizer.from_file(str(tok_file))

        # Device & precision
        resolved_device = torch.device(device) if device else get_device()
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        resolved_dtype = dtype_map.get(precision.lower(), torch.float32)

        if resolved_dtype != torch.float32:
            model = model.to(resolved_dtype)

        return cls(model, tokenizer, resolved_device, resolved_dtype)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        checkpoint: str | None = None,
    ) -> "InferenceEngine":
        """Load from a training YAML config file (convenience wrapper)."""
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        ckpt = checkpoint or cfg.get("checkpoint_path", "")
        return cls.from_checkpoint(
            checkpoint_path=ckpt,
            model_config_path=cfg["model_config"],
            tokenizer_path=cfg.get("tokenizer_path", "tokenizer"),
            precision=cfg.get("precision", "fp32"),
        )

    def _encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text).ids

    def _decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        gen_cfg: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate text from one or more prompts.

        Args:
            prompt: Single prompt string or list of prompts for batch generation.
            gen_cfg: Generation configuration. Uses defaults if ``None``.

        Returns:
            List of generated text strings (one per prompt).
        """
        gen_cfg = gen_cfg or GenerationConfig()
        if gen_cfg.seed is not None:
            torch.manual_seed(gen_cfg.seed)

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        # Encode all prompts
        encoded = [self._encode(p) for p in prompts]
        max_prompt_len = max(len(e) for e in encoded)

        # Left-pad to align prompts for batch processing
        pad_id = 0
        padded = [([pad_id] * (max_prompt_len - len(e))) + e for e in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)

        # Build stop token set
        stop_ids = set(gen_cfg.stop_tokens)
        if self.eos_token_id is not None:
            stop_ids.add(self.eos_token_id)

        # KV Cache
        kv_cache = None
        if gen_cfg.use_kv_cache:
            kv_cache = KVCache(
                num_layers=self.model.config.num_layers,
                max_batch_size=len(prompts),
                max_seq_len=max_prompt_len + gen_cfg.max_new_tokens,
            )

        generated = input_ids
        start_pos = 0

        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            if kv_cache is not None:
                # Prefill: process entire prompt
                out = self.model(input_ids, kv_cache=kv_cache, start_pos=0)
                logits = out["logits"][:, -1, :] if isinstance(out, dict) else (out[0][:, -1, :] if isinstance(out, (tuple, list)) else out[:, -1, :])
                next_token = _sample_next_token(logits, gen_cfg, generated)
                generated = torch.cat([generated, next_token], dim=-1)
                start_pos = input_ids.shape[1]

                # Decode: one token at a time
                for _ in range(gen_cfg.max_new_tokens - 1):
                    out = self.model(next_token, kv_cache=kv_cache, start_pos=start_pos)
                    logits = out["logits"][:, -1, :] if isinstance(out, dict) else (out[0][:, -1, :] if isinstance(out, (tuple, list)) else out[:, -1, :])
                    next_token = _sample_next_token(logits, gen_cfg, generated)
                    generated = torch.cat([generated, next_token], dim=-1)
                    start_pos += 1

                    # Check stopping conditions
                    if all(int(next_token[b, 0]) in stop_ids for b in range(len(prompts))):
                        break
            else:
                # No KV cache — recompute full context each step
                for _ in range(gen_cfg.max_new_tokens):
                    out = self.model(generated)
                    logits = out["logits"][:, -1, :] if isinstance(out, dict) else (out[0][:, -1, :] if isinstance(out, (tuple, list)) else out[:, -1, :])
                    next_token = _sample_next_token(logits, gen_cfg, generated)
                    generated = torch.cat([generated, next_token], dim=-1)

                    if all(int(next_token[b, 0]) in stop_ids for b in range(len(prompts))):
                        break

        # Clean up KV cache
        if kv_cache is not None:
            kv_cache.reset()

        # Decode results — strip padding
        results = []
        for i, enc in enumerate(encoded):
            gen_ids = generated[i, max_prompt_len:].tolist()
            # Trim at first stop token
            trimmed = []
            for tok_id in gen_ids:
                if tok_id in stop_ids:
                    break
                trimmed.append(tok_id)
            results.append(self._decode(trimmed))

        return results

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        gen_cfg: GenerationConfig | None = None,
    ) -> Generator[str, None, None]:
        """Stream generated tokens one at a time.

        Yields:
            Individual decoded token strings.
        """
        gen_cfg = gen_cfg or GenerationConfig()
        if gen_cfg.seed is not None:
            torch.manual_seed(gen_cfg.seed)

        encoded = self._encode(prompt)
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)

        stop_ids = set(gen_cfg.stop_tokens)
        if self.eos_token_id is not None:
            stop_ids.add(self.eos_token_id)

        kv_cache = None
        if gen_cfg.use_kv_cache:
            kv_cache = KVCache(
                num_layers=self.model.config.num_layers,
                max_batch_size=1,
                max_seq_len=len(encoded) + gen_cfg.max_new_tokens,
            )

        generated = input_ids
        start_pos = 0
        prev_ids: list[int] = list(encoded)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            if kv_cache is not None:
                # Prefill
                out = self.model(input_ids, kv_cache=kv_cache, start_pos=0)
                logits = out["logits"][:, -1, :] if isinstance(out, dict) else (out[0][:, -1, :] if isinstance(out, (tuple, list)) else out[:, -1, :])
                next_token = _sample_next_token(logits, gen_cfg, generated)
                tok_id = int(next_token[0, 0])
                if tok_id in stop_ids:
                    if kv_cache:
                        kv_cache.reset()
                    return
                prev_ids.append(tok_id)
                yield self._decode([tok_id])
                generated = torch.cat([generated, next_token], dim=-1)
                start_pos = input_ids.shape[1]

                for _ in range(gen_cfg.max_new_tokens - 1):
                    out = self.model(next_token, kv_cache=kv_cache, start_pos=start_pos)
                    logits = out["logits"][:, -1, :] if isinstance(out, dict) else (out[0][:, -1, :] if isinstance(out, (tuple, list)) else out[:, -1, :])
                    next_token = _sample_next_token(logits, gen_cfg, generated)
                    tok_id = int(next_token[0, 0])
                    if tok_id in stop_ids:
                        break
                    prev_ids.append(tok_id)
                    yield self._decode([tok_id])
                    generated = torch.cat([generated, next_token], dim=-1)
                    start_pos += 1
            else:
                for _ in range(gen_cfg.max_new_tokens):
                    out = self.model(generated)
                    logits = out["logits"][:, -1, :] if isinstance(out, dict) else (out[0][:, -1, :] if isinstance(out, (tuple, list)) else out[:, -1, :])
                    next_token = _sample_next_token(logits, gen_cfg, generated)
                    tok_id = int(next_token[0, 0])
                    if tok_id in stop_ids:
                        break
                    prev_ids.append(tok_id)
                    yield self._decode([tok_id])
                    generated = torch.cat([generated, next_token], dim=-1)

        if kv_cache is not None:
            kv_cache.reset()

    def tokenize(self, text: str) -> dict[str, Any]:
        """Tokenize text and return token IDs plus metadata."""
        enc = self.tokenizer.encode(text)
        return {
            "ids": enc.ids,
            "tokens": enc.tokens,
            "num_tokens": len(enc.ids),
        }

    def detokenize(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._decode(ids)

    def model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": getattr(self.model.config, "model_name", "vajra-lm"),
            "total_parameters": total_params,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_layers,
            "vocab_size": self.model.config.vocab_size,
            "max_position_embeddings": self.model.config.max_position_embeddings,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
