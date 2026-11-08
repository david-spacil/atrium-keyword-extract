"""tests/test_dm_doc_type.py — the document type from a document's text (#3; atrium-project#83).

No model or network: ``decision_client.FakeSystemOne`` answers on 127.0.0.1. The documents are the
three synthetic ``data_samples/DOC_LINE_CATEG`` CSVs and small ones written here.
"""

import csv
import json
from pathlib import Path

import pytest

import decision_client as dc
import dm_run

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "data_samples" / "DOC_LINE_CATEG"


def _client(fake):
    return dc.DecisionClient(fake.url, "fake-model", timeout_s=20, retries=0)


def _doc(folder: Path, name: str, lines) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file", "page_num", "line_num", "text", "categ"]
        )
        writer.writeheader()
        for i, (text, categ) in enumerate(lines, start=1):
            writer.writerow(
                {"file": name, "page_num": 1, "line_num": i, "text": text, "categ": categ}
            )
    return path


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_vendored_list_loads_with_its_digest_and_the_text_state():
    doc_types = dm_run.load_doc_types()
    assert list(doc_types["questions"]) == ["group", "document_type"]
    assert len(doc_types["types"]) == 32 and doc_types["abstain"] == "unspecified"
    assert "document_text" in doc_types["states"]


def test_load_refuses_a_list_whose_questions_drifted_from_the_digest(tmp_path):
    content = json.loads(dm_run.DOC_TYPE_LIST.read_text(encoding="utf-8"))
    content["questions"]["group"]["instructions"] = "edited"
    bad = tmp_path / "list.json"
    bad.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(SystemExit):
        dm_run.load_doc_types(bad)


def test_document_text_takes_clear_and_noisy_lines_whole_within_the_budget():
    rows = [
        {"text": "první řádek", "categ": "Clear"},
        {"text": "x#@!", "categ": "Trash"},
        {"text": "druhý řádek", "categ": "Noisy"},
        {"text": "", "categ": "Empty"},
        {"text": "třetí řádek", "categ": "Clear"},
    ]
    assert dm_run.document_text(rows, 1000) == ("první řádek\ndruhý řádek\ntřetí řádek", 3, 3)
    text, used, available = dm_run.document_text(rows, 24)
    assert text == "první řádek\ndruhý řádek" and (used, available) == (2, 3)


def test_gold_accepts_an_id_a_hes_number_or_a_czech_label(tmp_path):
    doc_types = dm_run.load_doc_types()
    report = next(t for t in doc_types["types"] if t["id"] == "fieldwork_report")
    gold = tmp_path / "gold.csv"
    gold.write_text(
        f"file,document_type\nA,fieldwork_report\nB,{report['hes']}\nC,{report['cs']}\nD,\n",
        encoding="utf-8",
    )
    assert dm_run._gold_types(gold, doc_types) == dict.fromkeys("ABC", "fieldwork_report")
    gold.write_text("file,document_type\nA,not a type\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        dm_run._gold_types(gold, doc_types)


def test_one_request_per_document_with_the_shared_questions_and_the_text_state(tmp_path):
    doc_types = dm_run.load_doc_types()
    out = tmp_path / "DM_DOC_TYPE.csv"
    with dc.FakeSystemOne() as fake:
        rows = dm_run.run_doc_types(_client(fake), sorted(SAMPLES.glob("*.csv")), out, doc_types)
        requests = fake.requests
    assert len(rows) == len(requests) == 3
    for request in requests:
        assert request["questions"] == doc_types["questions"]
        assert request["state"]["item"] == doc_types["states"]["document_text"]["item"]
        assert "images" not in request
    table = _read(out)
    assert list(table[0]) == dm_run.doc_type_columns(doc_types)
    for row in table:
        assert row["status"] == "ok" and row["schema_digest"] == doc_types["schema_digest"]
        assert row["document_type"] in {t["id"] for t in doc_types["types"]}
        assert row["group"] in doc_types["groups"] and row["type_group"] in doc_types["groups"]
        assert len(row["top3"].split("|")) == 3


def test_a_document_without_text_is_rejected_without_a_request_and_resume_skips_done(tmp_path):
    doc_types = dm_run.load_doc_types()
    _doc(tmp_path / "in", "EMPTYDOC", [("x#@!", "Trash"), ("", "Empty")])
    _doc(tmp_path / "in", "TEXTDOC", [("Nálezová zpráva z výzkumu", "Clear")])
    inputs = sorted((tmp_path / "in").glob("*.csv"))
    out = tmp_path / "t.csv"
    with dc.FakeSystemOne() as fake:
        dm_run.run_doc_types(_client(fake), inputs[:1], out, doc_types)
        assert not fake.requests
        with pytest.raises(SystemExit):
            dm_run.run_doc_types(_client(fake), inputs, out, doc_types)
        again = dm_run.run_doc_types(_client(fake), inputs, out, doc_types, resume=True)
    assert [r["file"] for r in again] == ["TEXTDOC"]
    first, second = _read(out)
    assert first["status"] == "rejected" and first["document_type"] == ""
    assert second["status"] == "ok"


def test_the_summary_scores_types_and_groups_against_gold():
    rows = [
        {
            "status": "ok",
            "latency_ms": "5",
            "group": "text",
            "document_type": "fieldwork_report",
            "type_group": "text",
            "gold_type": "fieldwork_report",
            "gold_group": "text",
        },
        {
            "status": "ok",
            "latency_ms": "7",
            "group": "text",
            "document_type": "unspecified",
            "type_group": "unspecified",
            "gold_type": "report_annex",
            "gold_group": "text",
        },
        {
            "status": "error",
            "latency_ms": "",
            "group": "",
            "document_type": "",
            "type_group": "",
            "gold_type": "",
            "gold_group": "",
        },
    ]
    summary = dm_run.summarise_doc_types(rows, "unspecified")
    assert summary["statuses"] == {"ok": 2, "error": 1}
    assert summary["abstained"] == 0.5 and summary["group_disagrees_with_type"] == 1
    assert summary["gold"] == {
        "documents": 2,
        "type_accuracy": 0.5,
        "group_accuracy": 1.0,
        "type_group_accuracy": 0.5,
    }


def test_main_end_to_end_names_the_server_by_kind_and_writes_paradata(tmp_path):
    out = tmp_path / "run" / "DM_DOC_TYPE.csv"
    with dc.FakeSystemOne() as fake:
        code = dm_run.main(
            [
                "--doc-type",
                "--input",
                str(SAMPLES),
                "--out",
                str(out),
                "--url",
                fake.url,
                "--model",
                "m",
            ]
        )
    assert code == 0
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["documents"] == 3 and summary["served_on"] == "local"
    assert fake.url not in json.dumps(summary)
    records = list((out.parent / "paradata").glob("*.json"))
    assert records and "decision_model" in records[0].read_text(encoding="utf-8")


def test_main_refuses_an_unconfigured_server_and_a_missing_input(monkeypatch):
    monkeypatch.delenv("DECISION_MODEL_URL", raising=False)
    with pytest.raises(SystemExit):
        dm_run.main(["--doc-type", "--input", str(SAMPLES), "--url", "", "--model", "m"])
    with pytest.raises(SystemExit):
        dm_run.main(["--doc-type", "--url", "http://127.0.0.1:9", "--model", "m"])
    assert dm_run.main(["--lines"]) == 2
