import tempfile
from dataset.mixture.models import DatasetMixture, DatasetMixtureEntry
from dataset.mixture.validators import MixtureValidator
from dataset.mixture.analysis import MixtureAnalyzer
from dataset.mixture.manager import MixtureManager

def test_mixture_model_creation():
    entry1 = DatasetMixtureEntry(name="c4", weight=50.0)
    entry2 = DatasetMixtureEntry(name="wikipedia", weight=50.0)
    
    mixture = DatasetMixture(name="test_mix", entries=[entry1, entry2])
    
    assert mixture.name == "test_mix"
    assert len(mixture.entries) == 2
    assert mixture.entries[0].weight == 50.0

def test_mixture_validation_weights():
    entry1 = DatasetMixtureEntry(name="c4", weight=50.0)
    entry2 = DatasetMixtureEntry(name="wikipedia", weight=40.0) # Sums to 90
    
    mixture = DatasetMixture(name="test_mix", entries=[entry1, entry2])
    
    report = MixtureValidator.validate(mixture)
    assert len(report["errors"]) > 0
    assert "100" in report["errors"][0]

def test_mixture_validation_duplicates():
    entry1 = DatasetMixtureEntry(name="c4", weight=50.0)
    entry2 = DatasetMixtureEntry(name="c4", weight=50.0) 
    
    mixture = DatasetMixture(name="test_mix", entries=[entry1, entry2])
    
    report = MixtureValidator.validate(mixture)
    assert any("Duplicate" in err for err in report["errors"])

def test_mixture_analysis():
    entry1 = DatasetMixtureEntry(name="c4", weight=60.0, language="en", estimated_tokens=1000)
    entry2 = DatasetMixtureEntry(name="wikipedia", weight=40.0, language="fr", estimated_tokens=500)
    
    mixture = DatasetMixture(name="test_mix", entries=[entry1, entry2])
    report = MixtureAnalyzer.generate_report(mixture)
    
    assert report["estimated_corpus_tokens"] == 1500
    assert report["language_distribution"]["en"] == 60.0
    assert report["language_distribution"]["fr"] == 40.0
    assert report["dataset_contribution"]["c4:1.0.0"] == 60.0

def test_mixture_manager():
    with tempfile.TemporaryDirectory() as d:
        manager = MixtureManager(d)
        
        entry = DatasetMixtureEntry(name="test_data", weight=100.0)
        mixture = DatasetMixture(name="my_mix", entries=[entry])
        
        manager.save(mixture)
        
        manager2 = MixtureManager(d)
        loaded = manager2.get("my_mix")
        assert loaded is not None
        assert len(loaded.entries) == 1
        assert loaded.entries[0].name == "test_data"
