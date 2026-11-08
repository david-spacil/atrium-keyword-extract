"""tests/test_vendored_doctype_parity.py -- the AMČR document-type list vendored from atrium-digital-convert.

ufal/atrium-project#83: atrium-digital-convert (its #8) builds AMČR's ``dokument_typ`` list for decision
models from this repository's own ``data_samples/vocab/amcr_flat.json``
(``tools/build_document_types.py``) and asks it about PDF page images. This repository asks the same
list about a document's OCR text, in ``dm_run.py --doc-type`` (#3), through its text-only view:
the written-document types and the abstain option. The service never reads the list.

The copy is pinned by the SHA-256 of the digital-convert file it was taken from (line endings
normalised to ``\\n``). A mismatch means a local edit to a vendored file -- change the list in
atrium-digital-convert and re-vendor -- or a re-vendor without updating the pin.

The copy is digital-convert's v2, with one view per input; ``dm_run.py`` reads its ``document_text``
view. It replaces v1, which no digital-convert release carried; the text view and its digest are the
same in both. To re-vendor: copy ``decisions/document_type.v<N>.json`` from a sibling ``../atrium-digital-convert``
checkout, point ``dm_run.DOC_TYPE_LIST`` at it, run ``python3 tests/test_vendored_doctype_parity.py``
and paste the printed pin over ``VENDORED``; set ``ADOPTED`` to the digital-convert commit and the
release that carry the file. Commit the copy and the pins together (atrium-project
docs/shared_components.md).

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

#: The owner revision the copy was taken from: the digital-convert commit of 2026-11-08 on `test`
#: that added decisions/document_type.v2.json with the digest below. Its tag v1.4.6-beta was never
#: released (GitHub's release downloads answered 504), so the first release to carry it is v1.4.7-beta.
ADOPTED = {
    "owner": "atrium-digital-convert",
    "commit": "427b690d66fd0ce1ce7b2454e9dd46fe719a4ca0",
    "release": "v1.4.7-beta",
}

VENDORED = {
    "decisions/document_type.v2.json": "8c1a2715bc7ee0fc8b3fddfbb9222983302af647a86bca5ff6db1e620f0712d7",
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
        (REPO_ROOT / "decisions" / "document_type.v2.json").read_text(encoding="utf-8")
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
