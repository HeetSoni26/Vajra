import torch.nn as nn
from release.export import ModelExporter
from release.package import ReleasePackager
from release.inference import InferenceExamplesGenerator


def test_model_exporter(tmp_path):
    exporter = ModelExporter(str(tmp_path))
    model = nn.Linear(10, 10)

    exporter.export_pytorch(model)
    assert (tmp_path / "pytorch_model.bin").exists()

    exporter.export_config({"vocab_size": 100})
    assert (tmp_path / "config.json").exists()


def test_release_packager(tmp_path):
    packager = ReleasePackager(str(tmp_path))
    model = nn.Linear(10, 10)
    config = {"vocab_size": 1024, "hidden_size": 64}

    packager.create_package(model, config)

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "LICENSE").exists()
    assert packager.verify_package()


def test_inference_examples(tmp_path):
    gen = InferenceExamplesGenerator(str(tmp_path))
    gen.generate_all()

    assert (tmp_path / "examples" / "basic_inference.py").exists()
    assert (tmp_path / "examples" / "cli_inference.py").exists()
    assert (tmp_path / "examples" / "streaming_inference.py").exists()
    assert (tmp_path / "examples" / "batch_inference.py").exists()
