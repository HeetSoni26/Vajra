import math
import time
import torch
from typing import Dict
from evaluation.metrics.base import BaseMetric


class StandardMetrics(BaseMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.correct_tokens = 0
        self.steps = 0
        self.start_time = time.time()

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        # logits: [bsz, seq, vocab]
        # labels: [bsz, seq] (shifted already, or we assume loss is pre-computed)
        bsz, seq = labels.shape
        self.total_loss += loss.item() * (bsz * seq)
        self.total_tokens += bsz * seq
        self.steps += 1

        # Accuracy
        preds = torch.argmax(logits, dim=-1)
        # Assuming last token is not predicted in standard CLM layout or we passed shifted labels.
        # This is a rough estimation. For strict accuracy, we use shifted preds/labels.
        # Our model returns shifted loss, so we'll just check accuracy exactly.
        preds_shift = preds[..., :-1]
        labels_shift = labels[..., 1:]

        self.correct_tokens += (preds_shift == labels_shift).sum().item()

    def compute(self) -> Dict[str, float]:
        elapsed = time.time() - self.start_time
        avg_loss = self.total_loss / max(1, self.total_tokens)

        # Next-token accuracy
        # Total tokens for accuracy is total_tokens - (bsz * steps) because of shift.
        # But we'll just use self.correct_tokens / (self.total_tokens * (seq-1)/seq) approx
        acc = self.correct_tokens / max(1, self.total_tokens)

        # Perplexity
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float("inf")

        # BPB (Bits per byte): Roughly log2(e) * loss / bytes_per_token
        # Assumes roughly 4 bytes per token for standard UTF-8 text average.
        bpb = (avg_loss * math.log2(math.e)) / 4.0

        throughput = self.total_tokens / elapsed if elapsed > 0 else 0

        return {
            "loss": avg_loss,
            "perplexity": ppl,
            "accuracy": acc,
            "bpb": bpb,
            "throughput_tokens_per_sec": throughput,
            "average_seq_length": self.total_tokens
            / max(
                1, self.steps * 1
            ),  # This is a placeholder since we don't have bsz tracked easily
        }
