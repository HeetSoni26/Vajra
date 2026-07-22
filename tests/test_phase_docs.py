from pathlib import Path


def test_completed_sections_exist():
    assert Path("docs/section_1_project_phases.md").exists()
    assert Path("docs/section_2_architecture_design.md").exists()


def test_phase_7_assets_exist():
    assert Path("Dockerfile.serve").exists()
    assert Path("deployment/Modelfile").exists()
    assert Path("sdk/foundationlm/client.py").exists()
