from pathlib import Path


def test_sections_1_to_15_exist():
    for idx in range(1, 16):
        matches = list(Path("docs").glob(f"section_{idx}_*.md"))
        assert matches, f"missing section {idx}"
        assert matches[0].read_text().strip().startswith(f"# Section {idx}")


def test_appendices_exist():
    assert Path("docs/appendix_a_key_papers.md").exists()
    assert Path("docs/appendix_b_quick_start_checklist.md").exists()
