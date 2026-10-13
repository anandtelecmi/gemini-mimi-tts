import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

_spec = importlib.util.spec_from_file_location(
    "prep", Path(__file__).resolve().parents[1] / "scripts" / "prepare_jsonl_dataset.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def test_parse_name():
    assert prep.parse_name("/x/studio__hindi__spk7__a1.wav") == ("studio", "spk7")
    assert prep.parse_name("/x/studio__hindi__a__b__c.wav") == ("studio", "a__b")
    assert prep.parse_name("/x/syn__hindi__a.wav") == ("syn", "hindi")
    assert prep.parse_name("/x/plain.wav") == ("unknown", "unknown")


def test_normalize_text():
    assert prep.normalize_text("  क​ा  \n x ") == "का x"


def test_pearson_and_pct():
    assert abs(prep.pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1) < 1e-9
    assert prep.pearson([1, 1, 1], [1, 2, 3]) != prep.pearson([1, 1, 1], [1, 2, 3])  # nan
    assert prep.pct([1, 2, 3, 4], 0.5) == 3


def test_keep():
    r = {"text": "a" * 50, "dur": 5.0}
    assert prep.keep(r, 1.5, 25, 4, 35) is None
    assert prep.keep({"text": "", "dur": 5}, 1.5, 25, 4, 35) == "empty_text"
    assert prep.keep({"text": "abc", "dur": 30}, 1.5, 25, 4, 35) == "duration"
    assert prep.keep({"text": "a", "dur": 5}, 1.5, 25, 4, 35) == "cps"


def test_split_and_tags():
    held = prep.split_speakers({"a": ["x", "y", "z"], "b": ["only"]}, 1, 0)
    assert len([h for h in held if h[0] == "a"]) == 1 and not [h for h in held if h[0] == "b"]
    tags = prep.speaker_tag_map(Counter({"x": 100, "y": 60, "z": 3}), 50)
    assert tags == {"x": "s000", "y": "s001"}


def test_load_records(tmp_path):
    p = tmp_path / "studio__hindi__spk1__a.wav"
    sf.write(str(p), np.zeros(24000, dtype="float32"), 24000)
    j = tmp_path / "m.jsonl"
    j.write_text(json.dumps({"audio": str(p), "text": "नमस्ते"}) + "\n")
    recs = prep.load_records([j])
    assert len(recs) == 1 and abs(recs[0]["dur"] - 1.0) < 1e-3
