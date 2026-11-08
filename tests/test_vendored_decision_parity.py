"""tests/test_vendored_decision_parity.py -- the decision-model client and the shared document-type list.

ufal/atrium-project#83 runs decision models (Clef, Clef-Flash, Strands Decider 2B, served at
``POST /v1/systemone``) in four tools. This repository uses two of their shared files, offline
only, in ``dm_run.py --doc-type`` (#3); the service never imports them:

* ``decision_client.py``, the client, owned by atrium-page-classification (its #58);
* ``decisions/document_type.v1.json``, the AMČR ``dokument_typ`` list with its two questions, owned by
  atrium-digital-convert (its #8, ``tools/build_document_types.py``), which builds it from this
  repository's own ``data_samples/vocab/amcr_flat.json``. Asking the same questions with the same
  digest keeps the two tools' answers comparable.

Each copy is pinned by the SHA-256 of the owner's file (line endings normalised to ``\\n``). A
mismatch means a local edit to a vendored file -- make the change in the owner instead -- or a
re-vendor without updating the pin. To re-vendor: copy the file from the sibling checkout, run
``python3 tests/test_vendored_decision_parity.py`` and paste the printed pins over ``VENDORED``, and
the owner's revision over ``ADOPTED``. Commit the copies and the pins together (atrium-project
docs/shared_components.md). When a sibling checkout sits next to this repo, the copies are also
compared with it byte for byte; that check is skipped when the checkout is absent (CI).
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Per copy, its owner and the owner revision it was taken from. ``commit``/``release`` stay ``None``
#: until the owner's change has landed and been released; the pin by content (``VENDORED``) holds
#: from the first day.
ADOPTED = {
    "decision_client.py": {
        "owner": "atrium-page-classification",
        "commit": "39a7bea87080de6e0e7ed30eba600ce05f9bef8b",
        "release": "v1.13.0-beta",
    },
    "decisions/document_type.v1.json": {
        "owner": "atrium-digital-convert",
        "commit": None,  # the digital-convert commit that adds decisions/document_type.v1.json
        "release": None,
    },
}

VENDORED = {
    "decision_client.py": "7031b3f5bdb9f397318a396ab0042e662b8c21994afe230107e3e74c604e9604",
    "decisions/document_type.v1.json": "56a876f0bc13fd3fe805a6600ea4de0c12bce475e8a43d7df2fe57e602063830",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.parametrize("rel", sorted(VENDORED))
def test_vendored_copy_matches_its_pin(rel):
    assert _digest(REPO_ROOT / rel) == VENDORED[rel], (
        f"{rel} differs from the {ADOPTED[rel]['owner']} copy it was vendored from -- change it there "
        "and re-vendor (see this module's docstring)"
    )


@pytest.mark.parametrize("rel", sorted(VENDORED))
def test_vendored_copy_matches_a_sibling_owner_checkout(rel):
    upstream = REPO_ROOT.parent / ADOPTED[rel]["owner"] / rel
    if not upstream.is_file():
        pytest.skip(f"no ../{ADOPTED[rel]['owner']} checkout with {rel} next to this repo")
    assert _digest(REPO_ROOT / rel) == _digest(upstream), (
        f"{rel} drifted from {ADOPTED[rel]['owner']}"
    )


@pytest.mark.parametrize("rel", sorted(ADOPTED))
def test_adopted_names_an_immutable_owner_revision(rel):
    adopted = ADOPTED[rel]
    assert adopted["owner"] in ("atrium-page-classification", "atrium-digital-convert")
    if adopted["commit"] is None:
        assert adopted["release"] is None
        pytest.skip(
            f"{rel}: the owner's commit is still to be recorded (pinned by content meanwhile)"
        )
    assert re.fullmatch(r"[0-9a-f]{40}", adopted["commit"]), (
        "a full commit SHA, not a branch or a short SHA"
    )
    assert re.fullmatch(r"v\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", adopted["release"])


def test_the_list_is_shared_with_this_repository_and_its_digest_holds():
    import decision_client as dc

    content = json.loads(
        (REPO_ROOT / "decisions" / "document_type.v1.json").read_text(encoding="utf-8")
    )
    assert "ufal/atrium-keyword-extract" in content["shared_with"]
    assert "document_text" in content["states"]
    assert dc.schema_digest(dc.questions(content["questions"])) == content["schema_digest"]


def test_the_client_selftest_passes():
    import decision_client

    assert decision_client.selftest() == 0


if __name__ == "__main__":
    for rel in sorted(VENDORED):
        print(f'    "{rel}": "{_digest(REPO_ROOT / rel)}",')
