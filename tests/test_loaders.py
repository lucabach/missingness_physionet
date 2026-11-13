"""Raw-file parsers on tiny hand-made files in the official formats."""

import numpy as np

from missingness.data import physionet2012, physionet2019
from missingness.data.records import records_from_frame


def test_physionet2019_psv(tmp_path):
    for folder, pid in [("training_setA", "p000001"), ("training_setB", "p100001")]:
        d = tmp_path / "training" / folder
        d.mkdir(parents=True)
        (d / f"{pid}.psv").write_text(
            "HR|Lactate|Age|Gender|ICULOS|SepsisLabel\n80|NaN|65|1|1|0\nNaN|2.1|65|1|2|1\n"
        )
    df = physionet2019.load_raw(tmp_path, ["HR", "Lactate"], ["Age", "Gender"])
    recs = records_from_frame(df, ["HR", "Lactate"], ["Age", "Gender"], stay_label=False)
    assert [r.source for r in recs] == ["hospital1", "hospital2"]
    assert recs[0].labels.tolist() == [0, 1]
    assert np.isnan(recs[0].values[0, 1]) and recs[0].values[1, 1] == 2.1


def test_physionet2012_txt(tmp_path):
    (tmp_path / "set-a").mkdir()
    (tmp_path / "set-a" / "132539.txt").write_text(
        "Time,Parameter,Value\n"
        "00:00,RecordID,132539\n00:00,Age,54\n00:00,Gender,0\n00:00,Height,-1\n"
        "00:00,ICUType,4\n00:00,Weight,-1\n"
        "00:07,HR,73\n00:37,HR,77\n05:10,Weight,80\n47:59,HR,90\n"
    )
    (tmp_path / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n132539,6,1,5,-1,0\n"
    )
    static = ["Age", "Gender", "Height", "ICUType"]
    df = physionet2012.load_raw(tmp_path, ["HR", "Weight"], static)
    (rec,) = records_from_frame(df, ["HR", "Weight"], static, stay_label=True)
    assert rec.values.shape == (48, 2)
    assert rec.values[0, 0] == 75  # mean of the two HR values in hour 0
    assert rec.values[5, 1] == 80 and rec.values[47, 0] == 90
    assert np.isnan(rec.static[2])  # Height -1 -> unknown
    assert rec.labels.tolist() == [0]


def test_physionet2012_drops_stays_without_measurements(tmp_path):
    (tmp_path / "set-a").mkdir()
    (tmp_path / "set-a" / "140501.txt").write_text(
        "Time,Parameter,Value\n00:00,RecordID,140501\n00:00,Age,70\n00:00,Gender,1\n"
        "00:00,Height,-1\n00:00,ICUType,2\n00:00,Weight,-1\n"
    )
    (tmp_path / "set-a" / "132539.txt").write_text(
        "Time,Parameter,Value\n00:00,RecordID,132539\n00:00,Age,54\n00:00,Gender,0\n"
        "00:00,Height,-1\n00:00,ICUType,4\n00:00,Weight,-1\n00:07,HR,73\n"
    )
    (tmp_path / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "140501,6,1,5,-1,0\n132539,6,1,5,-1,0\n"
    )
    df = physionet2012.load_raw(tmp_path, ["HR"], ["Age", "Gender", "Height", "ICUType"])
    assert df["patient_id"].unique().tolist() == ["132539"]
