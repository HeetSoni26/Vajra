"""
Tests for Phase 5 — Dataset Engineering & Training Preparation.
Covers: DataSourceRegistry, synthetic data generation, DatasetStatistics,
training configs, and training readiness verification.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# DataSourceRegistry tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDataSourceRegistry:
    """Tests for DataSourceRegistry and DataSource."""

    def test_register_and_get(self):
        from dataset.sources.registry import DataSource, DataSourceRegistry

        registry = DataSourceRegistry()
        src = DataSource(
            name="test-src",
            url="https://example.com/data",
            domain="web",
            license="CC-BY-4.0",
        )
        registry.register(src)
        retrieved = registry.get("test-src")
        assert retrieved is not None
        assert retrieved.name == "test-src"
        assert retrieved.domain == "web"

    def test_list_all(self):
        from dataset.sources.registry import DataSource, DataSourceRegistry

        registry = DataSourceRegistry()
        for i in range(3):
            registry.register(
                DataSource(
                    name=f"src-{i}",
                    url=f"https://example.com/{i}",
                    domain="web",
                    license="CC0-1.0",
                )
            )
        assert len(registry.list_all()) == 3

    def test_list_enabled(self):
        from dataset.sources.registry import DataSource, DataSourceRegistry

        registry = DataSourceRegistry()
        registry.register(
            DataSource(
                name="enabled", url="https://a.com", domain="web", license="CC0", enabled=True
            )
        )
        registry.register(
            DataSource(
                name="disabled", url="https://b.com", domain="web", license="CC0", enabled=False
            )
        )
        enabled = registry.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"

    def test_filter_by_domain(self):
        from dataset.sources.registry import DataSource, DataSourceRegistry

        registry = DataSourceRegistry()
        registry.register(
            DataSource(name="web-src", url="https://a.com", domain="web", license="CC0")
        )
        registry.register(
            DataSource(name="code-src", url="https://b.com", domain="code", license="MIT")
        )
        web = registry.filter_by_domain("web")
        assert len(web) == 1
        assert web[0].name == "web-src"

    def test_filter_by_tag(self):
        from dataset.sources.registry import DataSource, DataSourceRegistry

        registry = DataSourceRegistry()
        registry.register(
            DataSource(
                name="tagged",
                url="https://a.com",
                domain="math",
                license="CC0",
                tags=["pretraining", "math"],
            )
        )
        registry.register(
            DataSource(name="untagged", url="https://b.com", domain="web", license="CC0")
        )
        result = registry.filter_by_tag("math")
        assert len(result) == 1

    def test_summary(self):
        from dataset.sources.registry import create_default_registry

        registry = create_default_registry()
        summary = registry.summary()
        assert "total_sources" in summary
        assert summary["total_sources"] > 0
        assert "domains" in summary
        assert "web" in summary["domains"]

    def test_deregister(self):
        from dataset.sources.registry import DataSource, DataSourceRegistry

        registry = DataSourceRegistry()
        registry.register(DataSource(name="tmp", url="https://x.com", domain="web", license="CC0"))
        registry.deregister("tmp")
        assert registry.get("tmp") is None

    def test_save_and_load_yaml(self):
        from dataset.sources.registry import DataSource, DataSourceRegistry

        registry = DataSourceRegistry()
        registry.register(
            DataSource(
                name="yaml-test",
                url="https://example.com",
                domain="science",
                license="CC-BY-4.0",
                tags=["test"],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.yaml"
            registry.save_yaml(path)
            assert path.exists()

            registry2 = DataSourceRegistry()
            registry2.load_yaml(path)
            src = registry2.get("yaml-test")
            assert src is not None
            assert src.domain == "science"
            assert "test" in src.tags

    def test_download_manifest(self):
        from dataset.sources.registry import create_default_registry

        registry = create_default_registry()
        manifest = registry.generate_download_manifest()
        assert len(manifest) > 0
        assert all("name" in m for m in manifest)
        assert all("url" in m for m in manifest)

    def test_create_default_registry(self):
        from dataset.sources.registry import create_default_registry

        registry = create_default_registry()
        assert registry.get("fineweb-edu") is not None
        assert registry.get("wikipedia-en") is not None
        assert registry.get("the-stack-v2-dedup") is not None
        assert registry.get("open-web-math") is not None

    def test_datasource_to_dict_roundtrip(self):
        from dataset.sources.registry import DataSource

        src = DataSource(
            name="roundtrip",
            url="https://rt.com",
            domain="code",
            license="MIT",
            format="parquet",
            estimated_size_gb=10.5,
            tags=["a", "b"],
            quality_tier="high",
        )
        d = src.to_dict()
        src2 = DataSource.from_dict(d)
        assert src2.name == src.name
        assert src2.estimated_size_gb == src.estimated_size_gb
        assert src2.tags == src.tags


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSyntheticDataGenerator:
    """Tests for synthetic corpus generation."""

    def test_generate_documents_count(self):
        from dataset.sources.synthetic import generate_synthetic_documents

        docs = generate_synthetic_documents(num_documents=50, seed=42)
        assert len(docs) == 50

    def test_generate_documents_structure(self):
        from dataset.sources.synthetic import generate_synthetic_documents

        docs = generate_synthetic_documents(num_documents=10, seed=1)
        for doc in docs:
            assert "doc_id" in doc
            assert "text" in doc
            assert "domain" in doc
            assert len(doc["text"]) > 0

    def test_domain_distribution(self):
        from dataset.sources.synthetic import generate_synthetic_documents

        docs = generate_synthetic_documents(num_documents=200, seed=42)
        domains = set(d["domain"] for d in docs)
        # Should have multiple domains
        assert len(domains) >= 3

    def test_custom_domain_weights(self):
        from dataset.sources.synthetic import generate_synthetic_documents

        docs = generate_synthetic_documents(
            num_documents=100,
            domain_weights={"code": 1.0},
            seed=42,
        )
        assert all(d["domain"] == "code" for d in docs)

    def test_reproducibility(self):
        from dataset.sources.synthetic import generate_synthetic_documents

        docs1 = generate_synthetic_documents(num_documents=20, seed=42)
        docs2 = generate_synthetic_documents(num_documents=20, seed=42)
        assert docs1[0]["text"] == docs2[0]["text"]

    def test_write_synthetic_corpus(self):
        from dataset.sources.synthetic import write_synthetic_corpus

        with tempfile.TemporaryDirectory() as tmp:
            stats = write_synthetic_corpus(output_dir=tmp, num_documents=30, seed=42)
            assert stats["total_documents"] == 30
            assert stats["total_characters"] > 0
            synthetic_dir = Path(tmp) / "synthetic"
            jsonl_files = list(synthetic_dir.glob("*.jsonl"))
            assert len(jsonl_files) > 0
            # Verify file is valid JSONL
            first_file = jsonl_files[0]
            lines = first_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) > 0
            for line in lines:
                obj = json.loads(line)
                assert "text" in obj


# ──────────────────────────────────────────────────────────────────────────────
# DatasetStatistics tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDatasetStatistics:
    """Tests for dataset statistics and integrity validation."""

    @pytest.fixture
    def synthetic_tokenized_dir(self, tmp_path):
        """Create a minimal tokenized dataset for testing."""
        tokens = np.arange(2000, dtype=np.uint32) % 200  # 200 vocab
        train_tokens = tokens[:1600]
        val_tokens = tokens[1600:1800]
        test_tokens = tokens[1800:]

        train_tokens.tofile(tmp_path / "train.bin")
        val_tokens.tofile(tmp_path / "val.bin")
        test_tokens.tofile(tmp_path / "test.bin")

        meta = {
            "dtype": "uint32",
            "sequence_length": 128,
            "splits": {
                "train": {"file": "train.bin", "count": len(train_tokens)},
                "val": {"file": "val.bin", "count": len(val_tokens)},
                "test": {"file": "test.bin", "count": len(test_tokens)},
            },
        }
        import json

        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        return tmp_path

    def test_compute_statistics_runs(self, synthetic_tokenized_dir):
        from dataset.statistics import DatasetStatistics

        engine = DatasetStatistics(synthetic_tokenized_dir, vocab_size=200)
        stats = engine.compute_statistics()
        assert "splits" in stats
        assert "train" in stats["splits"]
        assert stats["splits"]["train"]["token_count"] == 1600

    def test_aggregate_stats(self, synthetic_tokenized_dir):
        from dataset.statistics import DatasetStatistics

        engine = DatasetStatistics(synthetic_tokenized_dir, vocab_size=200)
        stats = engine.compute_statistics()
        agg = stats["aggregate"]
        assert "total_tokens" in agg
        assert agg["total_tokens"] == 2000
        assert "vocab_coverage_pct" in agg
        assert agg["vocab_coverage_pct"] > 0

    def test_integrity_validation_passes(self, synthetic_tokenized_dir):
        from dataset.statistics import DatasetStatistics

        engine = DatasetStatistics(synthetic_tokenized_dir, vocab_size=200)
        result = engine.validate_integrity()
        assert result["all_passed"] is True

    def test_integrity_fails_missing_train(self, tmp_path):
        from dataset.statistics import DatasetStatistics

        engine = DatasetStatistics(tmp_path, vocab_size=200)
        result = engine.validate_integrity()
        assert result["all_passed"] is False

    def test_generate_report(self, synthetic_tokenized_dir):
        from dataset.statistics import DatasetStatistics

        engine = DatasetStatistics(synthetic_tokenized_dir, vocab_size=200)
        report = engine.generate_report()
        assert "statistics" in report
        assert "integrity_validation" in report
        report_path = synthetic_tokenized_dir / "dataset_report.json"
        assert report_path.exists()


# ──────────────────────────────────────────────────────────────────────────────
# Training configs tests
# ──────────────────────────────────────────────────────────────────────────────


class TestTrainingConfigs:
    """Tests for training configuration files."""

    def test_pretrain_tiny_config_exists(self):
        path = Path("configs/training/pretrain_tiny.yaml")
        assert path.exists(), f"Missing: {path}"

    def test_pretrain_125m_config_exists(self):
        path = Path("configs/training/pretrain_125m.yaml")
        assert path.exists(), f"Missing: {path}"

    def test_pretrain_370m_config_exists(self):
        path = Path("configs/training/pretrain_370m.yaml")
        assert path.exists(), f"Missing: {path}"

    def test_model_tiny_config_loadable(self):
        import yaml

        path = Path("configs/model/model_tiny.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert "hidden_size" in cfg
        assert "num_layers" in cfg

    def test_model_125m_config_loadable(self):
        import yaml

        path = Path("configs/model/model_125m.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["hidden_size"] == 768
        assert cfg["num_layers"] == 12

    def test_model_370m_config_loadable(self):
        import yaml

        path = Path("configs/model/model_370m.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["hidden_size"] == 1024
        assert cfg["num_layers"] == 24

    def test_model_configs_instantiable(self):
        from model.config import VajraConfig

        for name in ["model_tiny", "model_125m", "model_370m"]:
            cfg_path = Path(f"configs/model/{name}.yaml")
            cfg = VajraConfig.from_yaml(cfg_path)
            assert cfg.num_layers > 0
            assert cfg.hidden_size > 0

    def test_dataset_mix_config_loadable(self):
        import yaml

        path = Path("configs/data/dataset_mix.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert "domain_weights" in cfg
        weights = cfg["domain_weights"]
        # Weights should approximately sum to 1
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_sources_config_loadable(self):
        import yaml

        path = Path("configs/data/sources.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert "sources" in cfg
        assert len(cfg["sources"]) > 0
        for src in cfg["sources"]:
            assert "name" in src
            assert "domain" in src
            assert "license" in src


# ──────────────────────────────────────────────────────────────────────────────
# Prepare dataset script tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPrepareDataset:
    """Tests for the prepare_dataset script."""

    def test_prepare_synthetic_pipeline(self, tmp_path):
        """Full synthetic pipeline: generate → clean → tokenize → binary build."""
        from scripts.prepare_dataset import prepare_synthetic

        result = prepare_synthetic(
            output_dir=tmp_path,
            num_docs=50,
            sequence_length=32,
            vocab_size=128,
            seed=42,
        )
        assert "split_stats" in result
        assert result["cleaned_docs"] > 0
        assert result["split_stats"]["total_tokens"] > 0

        tokenized_dir = tmp_path / "tokenized"
        assert (tokenized_dir / "train.bin").exists()
        assert (tokenized_dir / "metadata.json").exists()

    def test_synthetic_train_bin_loadable(self, tmp_path):
        """Verify train.bin produced by synthetic pipeline is loadable."""
        from scripts.prepare_dataset import prepare_synthetic

        prepare_synthetic(
            output_dir=tmp_path,
            num_docs=30,
            sequence_length=32,
            vocab_size=128,
            seed=7,
        )
        train_bin = tmp_path / "tokenized" / "train.bin"
        arr = np.memmap(train_bin, dtype=np.uint32, mode="r")
        assert len(arr) > 0
        assert arr.max() < 128  # Within vocab bounds
