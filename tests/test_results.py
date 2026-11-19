import pandas as pd

from missingness.results import write_table


def test_write_table_with_models_keeps_the_other_models_rows(tmp_path):
    out = tmp_path / "measures.csv"
    write_table(pd.DataFrame({"model": ["gru", "gru"], "value": [1, 2]}), out)
    write_table(pd.DataFrame({"model": ["grud"], "value": [3]}), out, ["grud"])
    write_table(pd.DataFrame({"model": ["grud"], "value": [4]}), out, ["grud"])
    table = pd.read_csv(out)
    assert sorted(table["value"]) == [1, 2, 4]  # gru kept, old grud row replaced

    write_table(pd.DataFrame({"model": ["odernn"], "value": [5]}), out)  # no models: rewrite
    assert pd.read_csv(out)["value"].tolist() == [5]
