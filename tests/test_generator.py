import csv
from test_case_generator import TestCaseGenerator, TestCase


def test_export_csv(tmp_path):
    generator = TestCaseGenerator(model_name="google/flan-t5-base", load_model=False)
    cases = [
        TestCase(id=1, description="Login", steps="Open app", expected="Success"),
        TestCase(id=2, description="Logout", steps="Press logout", expected="Logged out"),
    ]
    out_file = tmp_path / "cases.csv"
    generator.export_csv(cases, out_file)
    assert out_file.exists()
    with open(out_file, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == ["ID", "Description", "Steps", "Expected"]
    assert len(reader) == 3
