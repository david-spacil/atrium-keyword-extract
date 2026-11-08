"""tests/test_vendored_doctype_parity.py -- the AMČR document-type list vendored from atrium-digital-convert.

ufal/atrium-project#83: atrium-digital-convert (its #8) builds AMČR's ``dokument_typ`` list for decision
models from this repository's own ``data_samples/vocab/amcr_flat.json``
(``tools/build_document_types.py``) and asks it about PDF page images. This repository asks the same
list about a document's OCR text, in ``dm_run.py --doc-type`` (#3), through its text-only view:
the written-document types and the abstain option. The service never reads the list.

The copy is pinned by the SHA-256 of the digital-convert file it was taken from (line endings
normalised to ``\\n``). A mismatch means a local edit to a vendored file -- change the list in
atrium-digital-convert and re-vendor -- or a re-vendor without updating the pin.

To re-vendor: copy ``decisions/document_type.v<N>.json`` from a sibling ``../atrium-digital-convert``
checkout, point ``dm_run.DOC_TYPE_LIST`` at it, run ``python3 tests/test_vendored_doctype_parity.py``
and paste the printed pin over ``VENDORED`` and ``ADOPTED``. Commit the copy and the pins together
(atrium-project docs/shared_components.md). digital-convert's v2 (one view per input) is the next
adoption, once its commit is on GitHub.

When an atrium-digital-convert checkout sits next to this repo with the same file, the copy is also
compared with it byte for byte; the para-drift job checks the owner out at ``ADOPTED["commit"]``.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DIGITAL_CONVERT = REPO_ROOT.parent / "atrium-digital-convert"

#: The owner revision the copy was taken from: the digital-convert commit of 2026-11-08 that added
#: decisions/document_type.v1.json with the digest below, first released in v1.4.6-beta.
ADOPTED = {
    "owner": "atrium-digital-convert",
    "commit": "2bb3f3cb43027744cbc5a49cab468b51c22799fc",
    "release": "v1.4.6-beta",
}

VENDORED = {
    "decisions/document_type.v1.json": "56a876f0bc13fd3fe805a6600ea4de0c12bce475e8a43d7df2fe57e602063830",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.parametrize("rel", sorted(VENDORED))
def test_vendored_copy_matches_its_pin(rel):
    assert _digest(REPO_ROOT / rel) == VENDORED[rel], (
        f"{rel} differs from the atrium-digital-convert copy it was vendored from -- change it there and "
        "re-vendor (see this module's docstring)"
    )


@pytest.mark.parametrize("rel", sorted(VENDORED))
def test_vendored_copy_matches_a_sibling_digital_convert_checkout(rel):
    upstream = DIGITAL_CONVERT / rel
    if not upstream.is_file():
        pytest.skip(f"no ../atrium-digital-convert checkout with {rel} next to this repo")
    assert _digest(REPO_ROOT / rel) == _digest(upstream), (
        f"{rel} drifted from atrium-digital-convert"
    )


def test_adopted_names_an_immutable_owner_revision():
    assert ADOPTED["owner"] == DIGITAL_CONVERT.name
    assert re.fullmatch(r"[0-9a-f]{40}", ADOPTED["commit"]), (
        "a full commit SHA, not a branch or a short SHA"
    )
    assert re.fullmatch(r"v\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", ADOPTED["release"])


def test_the_list_is_shared_with_this_repository():
    content = json.loads(
        (REPO_ROOT / "decisions" / "document_type.v1.json").read_text(encoding="utf-8")
    )
    assert "ufal/atrium-keyword-extract" in content["shared_with"]


def test_the_text_view_offers_the_written_types_only_and_takes_no_image():
    import dm_run

    view = dm_run.load_doc_types()
    assert list(view["questions"]) == ["document_type"]
    assert {t["group"] for t in view["types"]} == {"text", "unspecified"}
    assert view["inputs"]["image"] == "none"


if __name__ == "__main__":
    for rel in sorted(VENDORED):
        print(f'    "{rel}": "{_digest(REPO_ROOT / rel)}",')
