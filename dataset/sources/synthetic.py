"""
Synthetic data generator for Vajra Framework development and testing.

Generates synthetic text data that mimics the structure and characteristics
of real pretraining data, enabling pipeline validation and training dry-runs
without requiring actual dataset downloads.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

from utils.file_utils import ensure_dir
from utils.logging import setup_logger

logger = setup_logger("synthetic_data_generator")

# ──────────────────────────────────────────────────────────────────────────────
# Domain templates — representative text patterns for each domain
# ──────────────────────────────────────────────────────────────────────────────
_DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "web": [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
        "In today's world, technology plays an increasingly important role in our daily lives. From smartphones to artificial intelligence, the digital revolution continues to transform how we work, communicate, and learn.",
        "Climate change is one of the most pressing challenges facing humanity. Scientists around the world are working to understand its causes, predict its effects, and develop solutions to mitigate its impact.",
        "Education is the foundation of a prosperous society. Access to quality education empowers individuals and communities, fostering innovation, economic growth, and social progress.",
        "The internet has revolutionized the way we access information. Search engines, social media platforms, and online encyclopedias have made knowledge more accessible than ever before.",
    ],
    "code": [
        "def fibonacci(n: int) -> int:\n    \"\"\"Calculate the nth Fibonacci number using dynamic programming.\"\"\"\n    if n <= 1:\n        return n\n    dp = [0, 1]\n    for i in range(2, n + 1):\n        dp.append(dp[i-1] + dp[i-2])\n    return dp[n]\n",
        "class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n\n    def insert(self, value):\n        if value < self.value:\n            if self.left is None:\n                self.left = BinarySearchTree(value)\n            else:\n                self.left.insert(value)\n        else:\n            if self.right is None:\n                self.right = BinarySearchTree(value)\n            else:\n                self.right.insert(value)\n",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):\n        super().__init__()\n        self.attention = nn.MultiheadAttention(d_model, n_heads)\n        self.ffn = nn.Sequential(\n            nn.Linear(d_model, d_ff),\n            nn.GELU(),\n            nn.Linear(d_ff, d_model),\n        )\n        self.norm1 = nn.LayerNorm(d_model)\n        self.norm2 = nn.LayerNorm(d_model)\n        self.dropout = nn.Dropout(dropout)\n",
    ],
    "math": [
        "Theorem: For any positive integer n, the sum of the first n natural numbers is n(n+1)/2.\nProof: We proceed by mathematical induction.\nBase case: For n=1, the sum is 1 = 1(2)/2 = 1. ✓\nInductive step: Assume the formula holds for some k >= 1. Then the sum of the first k+1 numbers is k(k+1)/2 + (k+1) = (k+1)(k+2)/2. ✓\nBy the principle of mathematical induction, the formula holds for all positive integers.",
        "The quadratic formula states that for ax² + bx + c = 0, the solutions are x = (-b ± √(b²-4ac)) / (2a). The discriminant Δ = b²-4ac determines the nature of the roots: if Δ > 0, there are two distinct real roots; if Δ = 0, there is exactly one real root; if Δ < 0, there are two complex conjugate roots.",
        "Let f(x) = e^x. The Taylor series expansion of f about x = 0 is:\ne^x = Σ_{n=0}^{∞} x^n / n! = 1 + x + x²/2! + x³/3! + ...\nThis series converges for all real x, and the radius of convergence is infinite.",
    ],
    "science": [
        "The central dogma of molecular biology describes the flow of genetic information from DNA to RNA to protein. DNA replication produces identical copies of DNA, transcription converts DNA to messenger RNA, and translation converts mRNA into polypeptide chains that fold into functional proteins.",
        "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scale. The wave function ψ(x,t) contains all information about a quantum system, and the probability of finding a particle at position x is given by |ψ(x,t)|².",
        "The standard model of particle physics classifies all known elementary particles into quarks, leptons, and gauge bosons. The Higgs boson, discovered at CERN in 2012, completes the model by explaining how particles acquire mass through the Higgs mechanism.",
    ],
    "books": [
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness. The protagonist wandered through the streets of the ancient city, contemplating the nature of existence and the passage of time.",
        "The old man sat on the porch, watching the sun dip below the horizon. He had lived through wars and revolutions, witnessed the rise and fall of empires, and yet found the greatest wisdom in the simple act of watching the world turn.",
        "She opened the weathered journal, its pages yellowed with age. The handwriting was faded but legible, telling the story of a voyage across uncharted seas, of encounters with peoples whose customs were as varied as the stars.",
    ],
    "wikipedia": [
        "Machine learning is a subset of artificial intelligence (AI) that focuses on building systems that learn from data. Unlike traditional programming where explicit rules are coded, machine learning algorithms identify patterns and make decisions with minimal human intervention. The field has evolved significantly since its inception in the 1950s.",
        "The periodic table of elements organizes chemical elements by their atomic number, electron configuration, and recurring chemical properties. Elements in the same column (group) share similar chemical behaviors. Dmitri Mendeleev published the first widely recognized periodic table in 1869.",
        "The solar system consists of the Sun and everything bound to it by gravity, including eight planets, dwarf planets, asteroids, comets, and meteoroids. The four inner planets (Mercury, Venus, Earth, Mars) are terrestrial, while the four outer planets (Jupiter, Saturn, Uranus, Neptune) are gas and ice giants.",
    ],
    "technical": [
        "Q: How do I implement a REST API with pagination?\nA: Use cursor-based pagination for large datasets. Return a `next_cursor` token in each response that clients pass in subsequent requests. This avoids the performance issues of offset-based pagination when dealing with millions of records.\n\nExample:\nGET /api/users?cursor=abc123&limit=50\nResponse: { \"data\": [...], \"next_cursor\": \"def456\" }",
        "Q: What is the difference between TCP and UDP?\nA: TCP (Transmission Control Protocol) provides reliable, ordered delivery of data with error checking and flow control. UDP (User Datagram Protocol) provides faster, connectionless communication without guaranteed delivery. TCP is used for web browsing and email, while UDP is preferred for real-time applications like gaming and video streaming.",
    ],
}


def _generate_paragraph(domain: str, rng: random.Random) -> str:
    """Generate a single paragraph for the given domain."""
    templates = _DOMAIN_TEMPLATES.get(domain, _DOMAIN_TEMPLATES["web"])
    base = rng.choice(templates)

    # Add random variation to avoid exact deduplication
    suffix_words = rng.randint(3, 15)
    suffix = " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10)))
        for _ in range(suffix_words)
    )
    return f"{base} {suffix}"


def generate_synthetic_documents(
    num_documents: int = 100,
    domain_weights: dict[str, float] | None = None,
    min_paragraphs: int = 1,
    max_paragraphs: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic documents with realistic domain distribution.

    Args:
        num_documents: Number of documents to generate.
        domain_weights: Domain → weight mapping. Defaults to standard mix.
        min_paragraphs: Minimum paragraphs per document.
        max_paragraphs: Maximum paragraphs per document.
        seed: Random seed for reproducibility.

    Returns:
        List of document dicts with ``doc_id``, ``text``, ``domain``, ``source_file`` keys.
    """
    rng = random.Random(seed)

    if domain_weights is None:
        domain_weights = {
            "web": 0.35,
            "code": 0.25,
            "math": 0.10,
            "science": 0.10,
            "books": 0.08,
            "wikipedia": 0.05,
            "technical": 0.05,
        }

    domains = list(domain_weights.keys())
    weights = [domain_weights[d] for d in domains]

    documents: list[dict[str, Any]] = []
    for i in range(num_documents):
        domain = rng.choices(domains, weights=weights, k=1)[0]
        num_paragraphs = rng.randint(min_paragraphs, max_paragraphs)
        paragraphs = [_generate_paragraph(domain, rng) for _ in range(num_paragraphs)]
        text = "\n\n".join(paragraphs)

        documents.append({
            "doc_id": f"synthetic_{i:06d}",
            "text": text,
            "domain": domain,
            "source_file": f"synthetic/{domain}/doc_{i:06d}.txt",
        })

    logger.info(
        f"Generated {len(documents)} synthetic documents across "
        f"{len(set(d['domain'] for d in documents))} domains"
    )
    return documents


def write_synthetic_corpus(
    output_dir: str | Path = "data/raw",
    num_documents: int = 200,
    domain_weights: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic corpus and write JSONL files to disk.

    Writes one JSONL file per domain under ``output_dir/synthetic/``.

    Returns:
        Summary statistics of the generated corpus.
    """
    output_dir = ensure_dir(Path(output_dir) / "synthetic")
    documents = generate_synthetic_documents(
        num_documents=num_documents,
        domain_weights=domain_weights,
        seed=seed,
    )

    # Group by domain and write
    domain_docs: dict[str, list[dict[str, Any]]] = {}
    for doc in documents:
        domain_docs.setdefault(doc["domain"], []).append(doc)

    total_chars = 0
    for domain, docs in domain_docs.items():
        file_path = output_dir / f"{domain}.jsonl"
        with file_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({"text": doc["text"], "doc_id": doc["doc_id"]}, ensure_ascii=False) + "\n")
                total_chars += len(doc["text"])

    stats = {
        "output_dir": str(output_dir),
        "total_documents": len(documents),
        "total_characters": total_chars,
        "domains": {d: len(docs) for d, docs in domain_docs.items()},
        "seed": seed,
    }

    logger.info(f"Synthetic corpus written to {output_dir}: {len(documents)} documents, {total_chars:,} chars")
    return stats
