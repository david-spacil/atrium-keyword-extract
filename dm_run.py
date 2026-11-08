#!/usr/bin/env python3
"""dm_run.py — decision models over the controlled vocabulary (#3, Option 4 of #2; ufal/atrium-project#83).

A decision model cannot read the whole vocabulary at once (4,711 terms; at most 255 options per
``choice``), so it walks down it: is the line content at all (L0), which facet (L1), the geographic
gate where the guardrail covers the branch (L1g), which group (L2), then down the terms' own tree
(L3…), with ``<this term>`` where a node is itself a term and ``none_of_these`` at every step. An
answer is therefore always a vocabulary term or "unresolved" (#2's ruling D2).

This file builds that tree. ``--dump-tree OUT`` writes it as JSON with its digest. The digest goes
into every run's paradata, so a vocabulary change shows in the runs. Line mode (the descent's
requests) is not written yet.

**The document type** (``--doc-type``). Each document of a ``DOC_LINE_CATEG`` folder gets an AMČR
``dokument_typ``, chosen by a decision model from its text alone, never an image: the document's
``Clear`` and ``Noisy`` lines, in order, whole lines up to ``--max-chars``. The list is the one
atrium-digital-convert builds from this repository's ``amcr_flat.json`` and asks about PDF page
images (its #8), vendored as ``decisions/document_type.v1.json`` and pinned in
``tests/test_vendored_doctype_parity.py``; the client, ``decision_client.py``, is pinned from
atrium-page-classification in ``tests/test_vendored_decision_parity.py``. This tool asks the list's
**text-only view** (``document_text``): one ``document_type`` choice over the 6 written-document
types and ``neurčený dokument`` (the abstain option), with no ``group`` question. Photographs,
aerial photographs, plans, maps and drawings cannot be told from OCR text, so they are not offered.
``dokument_typ`` is withheld from the keyword vocabulary (``taxonomy_config.json``), so a type is
never a keyword: it is a document-level answer of its own. The table, ``DM_DOC_TYPE.csv``, and a
``.summary.json`` sit beside each other; with ``--gold`` (``file,document_type``) the summary scores
the types, also over the documents whose catalogue type the view offers. Nothing reaches the record.

How the tree is made, from files the vocabulary build already writes:

* **Terms, facets, groups:** ``prompt_template.vocabulary_terms()`` over ``data_samples/vocab/
  union_nested.json``, without the themes ``llm_client_shared.excluded_prompt_themes()`` withholds:
  the same terms the generative prompt offers. Facets and groups are routing levels only (P2), never
  an answer.
* **A term's parent** is its nearest ancestor that the union offers in the same facet. For TEATER
  terms it comes from ``data_samples/vocab/specificity_pairs.csv`` (``nearest_ancestor_id``, rows
  with ``same_facet``). For AMČR terms it is the first ``broader`` id of ``amcr_flat.json`` that
  the union offers in that facet, which links a ``*_druh`` term to its ``*_kat`` category. A term
  with no offered ancestor hangs from its group. An ancestor in another group of the facet pulls the
  term into that group's subtree. A group left with no terms of its own is dropped.
* **The geographic gate** marks the branches ``taxonomy_config.json`` ``_settings.geo_guardrail.covers``
  names (``heslar:<scheme>`` — the groups built from that AMČR heslář; ``teater:<id>`` — that term).
* **Chunks:** a node with more options than a ``choice`` takes is split, in Czech alphabetical order,
  into balanced chunks. Each chunk is asked with ``none_of_these``, then one ``choice`` is asked among
  the chunk winners. Chunks are plumbing and are never emitted (P1).

Run it from the repository root::

    python3 dm_run.py --dump-tree /tmp/dm_tree.json
    python3 dm_run.py --doc-type --input data_samples/DOC_LINE_CATEG --out /tmp/dm/DM_DOC_TYPE.csv \\
        --model clef-oQ8e --served-on ufal-endpoint        # URL from $DECISION_MODEL_URL
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
VOCAB = REPO_ROOT / "data_samples" / "vocab"
DEFAULT_INPUTS = {
    "vocabulary": VOCAB / "union_nested.json",
    "vocabulary_meta": VOCAB / "union_nested.meta.json",
    "specificity_pairs": VOCAB / "specificity_pairs.csv",
    "amcr_flat": VOCAB / "amcr_flat.json",
    "taxonomy_config": REPO_ROOT / "data_samples" / "taxonomy_config.json",
    "llm_config": REPO_ROOT / "llm_config.txt",
    "prompt_blocks": REPO_ROOT / "prompts" / "system_prompt.txt",
}

TREE_FORMAT = "atrium-dm-tree/1"
MAX_OPTIONS = 255  # a Jev/SystemOne choice
THIS_TERM = "<this term>"
NONE_OF_THESE = "none_of_these"
RESERVED = (THIS_TERM, NONE_OF_THESE)

#: Czech alphabetical order (ČSN 97 6030, primary level): č, ř, š, ž and "ch" are letters of their
#: own; the acute, the ring and ď/ť/ň's caron only break ties.
_CZ_LETTERS = "a b c č d e f g h ch i j k l m n o p q r ř s š t u v w x y z ž".split()
_CZ_RANK = {letter: rank for rank, letter in enumerate(_CZ_LETTERS)}
_CZ_SECONDARY = {
    "á": "a",
    "é": "e",
    "ě": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "ů": "u",
    "ý": "y",
    "ď": "d",
    "ť": "t",
    "ň": "n",
}


def czech_key(text: str) -> Tuple:
    """A sort key in Czech alphabetical order (case-insensitive; digits and others after letters)."""
    s = text.lower()
    primary: List[Tuple[int, str]] = []
    secondary: List[int] = []
    i = 0
    while i < len(s):
        if s.startswith("ch", i):
            primary.append((0, f"{_CZ_RANK['ch']:03d}"))
            secondary.append(0)
            i += 2
            continue
        ch = s[i]
        base = _CZ_SECONDARY.get(ch, ch)
        if base in _CZ_RANK:
            primary.append((0, f"{_CZ_RANK[base]:03d}"))
            secondary.append(0 if base == ch else 1)
        else:
            folded = unicodedata.normalize("NFKD", ch)[0]
            primary.append((1, folded))
            secondary.append(0)
        i += 1
    return (primary, secondary, text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(tree: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the tree, without its digest and timestamp."""
    body = {k: v for k, v in tree.items() if k not in ("digest", "generated_utc")}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


# ── building ──────────────────────────────────────────────────────────────────────────────────


def _load_inputs(paths: Mapping[str, Path]) -> Dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    import llm_client_shared  # noqa: PLC0415
    import prompt_template  # noqa: PLC0415
    from vocab_manager import VocabularyManager  # noqa: PLC0415

    vocab = json.loads(paths["vocabulary"].read_text(encoding="utf-8"))
    manager = VocabularyManager(
        vocab_path=str(paths["vocabulary"]), config_path=str(paths["taxonomy_config"])
    )
    excluded = sorted(llm_client_shared.excluded_prompt_themes(manager))
    terms = prompt_template.vocabulary_terms(vocab, excluded)
    config = llm_client_shared.load_config(str(paths["llm_config"]))
    blocks = prompt_template.load_blocks(paths["prompt_blocks"])
    taxonomy = json.loads(paths["taxonomy_config"].read_text(encoding="utf-8"))
    flat = json.loads(paths["amcr_flat"].read_text(encoding="utf-8"))["terms"]
    flat = list(flat.values()) if isinstance(flat, dict) else flat
    with paths["specificity_pairs"].open(newline="", encoding="utf-8") as handle:
        pairs = [row for row in csv.DictReader(handle)]
    meta = (
        json.loads(paths["vocabulary_meta"].read_text(encoding="utf-8"))
        if paths["vocabulary_meta"].is_file()
        else {}
    )
    return {
        "terms": terms,
        "excluded": excluded,
        "geo_mode": prompt_template.resolve_geo_guardrail(config),
        "blocks": blocks,
        "geo_covers": list(
            (taxonomy.get("_settings", {}).get("geo_guardrail") or {}).get("covers") or []
        ),
        "amcr": {str(rec["source_id"]): rec for rec in flat},
        "pairs": pairs,
        "meta": meta,
        "sentinel": prompt_template.META_TEXT_TERM,
    }


def build_tree(paths: Optional[Mapping[str, Path]] = None) -> Dict[str, Any]:
    """The descent tree, as a JSON-ready dict (see the module docstring for the rules)."""
    paths = {**DEFAULT_INPUTS, **(paths or {})}
    data = _load_inputs(paths)
    sentinel = data["sentinel"]
    terms = [t for t in data["terms"] if t["cs"] != sentinel["cs"]]

    # Node ids: "t:<facet>/<cs>", made unique within the facet if a label repeats.
    term_ids: List[str] = []
    seen_ids: Counter = Counter()
    for t in terms:
        base = f"t:{t['theme']}/{t['cs']}"
        seen_ids[base] += 1
        term_ids.append(base if seen_ids[base] == 1 else f"{base}#{seen_ids[base]}")
    clashes = [t["cs"] for t in terms if t["cs"] in RESERVED]
    if clashes:
        raise SystemExit(f"vocabulary terms named like a reserved option: {clashes}")

    by_source_id: Dict[Tuple[str, str], set] = defaultdict(set)
    for index, t in enumerate(terms):
        for ident in t.get("ids", []):
            by_source_id[(t["theme"], str(ident["id"]))].add(index)
    pairs = {
        (row["facet"], str(row["source_id"])): row
        for row in data["pairs"]
        if row.get("same_facet") == "True"
    }

    parent: Dict[int, int] = {}
    edges = Counter()
    for index, t in enumerate(terms):
        teater_parent = amcr_parent = None
        for ident in t.get("ids", []):
            key = (t["theme"], str(ident["id"]))
            if ident["source"] == "teater" and key in pairs:
                found = by_source_id.get(
                    (t["theme"], str(pairs[key]["nearest_ancestor_id"])), set()
                ) - {index}
                if found and teater_parent is None:
                    teater_parent = min(found)
            if (
                ident["source"] == "amcr"
                and str(ident["id"]) in data["amcr"]
                and amcr_parent is None
            ):
                for broader in data["amcr"][str(ident["id"])].get("broader") or []:
                    found = by_source_id.get((t["theme"], str(broader)), set()) - {index}
                    if found:
                        amcr_parent = min(found)
                        break
        if teater_parent is not None and amcr_parent is not None and teater_parent != amcr_parent:
            edges["teater_amcr_conflicts"] += 1
        chosen = teater_parent if teater_parent is not None else amcr_parent
        if chosen is None:
            continue
        parent[index] = chosen
        edges["from_teater" if teater_parent is not None else "from_amcr"] += 1
        if terms[chosen]["sub"] != t["sub"]:
            edges["cross_group"] += 1

    # A cycle would make the descent endless; break it at the edge that closes it (none today).
    for start in list(parent):
        path, node = [], start
        while node in parent:
            if node in path:
                del parent[node]
                edges["cycles_broken"] += 1
                break
            path.append(node)
            node = parent[node]

    def group_of(index: int) -> str:
        node = index
        while node in parent:
            node = parent[node]
        return terms[node]["sub"]

    nodes: Dict[str, Dict[str, Any]] = {}
    facets: List[str] = []
    groups_of: Dict[str, List[str]] = defaultdict(list)
    for t in terms:
        if t["theme"] not in facets:
            facets.append(t["theme"])
        if t["sub"] not in groups_of[t["theme"]]:
            groups_of[t["theme"]].append(t["sub"])

    children: Dict[int, List[int]] = defaultdict(list)
    roots: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, t in enumerate(terms):
        if index in parent:
            children[parent[index]].append(index)
        else:
            roots[(t["theme"], t["sub"])].append(index)

    geo = _geo_targets(data["geo_covers"], terms, data["amcr"])

    def sorted_terms(indices: Sequence[int]) -> List[str]:
        return [term_ids[i] for i in sorted(indices, key=lambda i: czech_key(terms[i]["cs"]))]

    dropped_groups = []
    for facet in facets:
        group_nodes = []
        for group in groups_of[facet]:
            members = roots.get((facet, group), [])
            if not members:
                dropped_groups.append(f"{facet}/{group}")
                continue
            gid = f"g:{facet}/{group}"
            nodes[gid] = {
                "kind": "group",
                "label": group,
                "facet": facet,
                "children": sorted_terms(members),
                "geo_gated": (facet, group) in geo["groups"],
            }
            group_nodes.append(gid)
        nodes[f"f:{facet}"] = {
            "kind": "facet",
            "label": facet,
            "children": sorted(group_nodes, key=lambda g: czech_key(nodes[g]["label"])),
        }
    for index, t in enumerate(terms):
        node = {
            "kind": "term",
            "cs": t["cs"],
            "en": t.get("en", t["cs"]),
            "facet": t["theme"],
            "group": group_of(index),
            "ids": t.get("ids", []),
            "children": sorted_terms(children.get(index, [])),
            "geo_gated": index in geo["terms"],
        }
        if t.get("bare_cs"):
            node["bare_cs"] = t["bare_cs"]
        nodes[term_ids[index]] = node
    nodes["root"] = {"kind": "root", "children": [f"f:{facet}" for facet in facets]}

    for node in nodes.values():
        extras = 1 + (
            1 if node["kind"] == "term" else 0
        )  # none_of_these, and <this term> for a term
        if len(node["children"]) + extras > MAX_OPTIONS:
            node["chunks"] = _chunks(node["children"], MAX_OPTIONS - 1)

    tree = {
        "format": TREE_FORMAT,
        "generated_by": "dm_run.py --dump-tree",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            name: {
                "path": str(path.relative_to(REPO_ROOT))
                if path.is_relative_to(REPO_ROOT)
                else str(path),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if path.is_file()
        },
        "vocabulary_meta": {
            k: data["meta"].get(k) for k in ("generated_utc", "generator", "schema_version")
        }
        | {"total": (data["meta"].get("counts") or {}).get("total")},
        "excluded_themes": data["excluded"],
        "geo_guardrail": {"mode": data["geo_mode"], "covers": data["geo_covers"]},
        "questions": _question_wording(data["blocks"], sentinel),
        "sentinel": {"cs": sentinel["cs"], "en": sentinel["en"]},
        "reserved_options": list(RESERVED),
        "nodes": nodes,
        "stats": {},
    }
    tree["stats"] = tree_stats(tree, edges, dropped_groups)
    tree["digest"] = tree_digest(tree)
    return tree


def _geo_targets(
    covers: Sequence[str], terms: Sequence[Mapping[str, Any]], amcr: Mapping[str, Any]
) -> Dict[str, set]:
    """The groups (``heslar:<scheme>``) and terms (``teater:<id>``) the geographic gate covers."""
    schemes = {c.split(":", 1)[1] for c in covers if c.startswith("heslar:")}
    teater = {c.split(":", 1)[1] for c in covers if c.startswith("teater:")}
    groups, gated = set(), set()
    for index, t in enumerate(terms):
        for ident in t.get("ids", []):
            if (
                ident["source"] == "amcr"
                and (amcr.get(str(ident["id"])) or {}).get("scheme") in schemes
            ):
                groups.add((t["theme"], t["sub"]))
            if ident["source"] == "teater" and str(ident["id"]) in teater:
                gated.add(index)
    return {"groups": groups, "terms": gated}


def _chunks(children: Sequence[str], size: int) -> List[List[str]]:
    count = -(-len(children) // size)
    step = -(-len(children) // count)
    return [list(children[i : i + step]) for i in range(0, len(children), step)]


def _question_wording(blocks: Mapping[str, str], sentinel: Mapping[str, str]) -> Dict[str, Any]:
    """The texts of the decision questions, taken from the generative prompt's own blocks."""
    return {
        "L0_content": {
            "instructions": "Is the marked line archaeological content?",
            "criteria": {
                "false": blocks.get("rule.metatext", "").strip(),
                "true": "The line carries archaeological content.",
            },
            "no_gives": sentinel["cs"],
        },
        "L1g_geographic": {
            "instructions": "Is the marked line genuinely about a place, a people or a dynasty?",
            "criteria": {"true": blocks.get("guardrail.geographic.preference", "").strip()},
        },
        "L1_facet": "Which part of the vocabulary does the marked line belong to?",
        "L2_group": "Which group of terms fits the marked line?",
        "L3_term": "Which term fits the marked line best? Choose the most specific one the line supports.",
        "chunk_final": "Which of these terms fits the marked line best?",
    }


# ── the questions a descent asks ──────────────────────────────────────────────────────────────


def node_options(
    tree: Mapping[str, Any],
    node_id: str,
    chunk: Optional[int] = None,
    among: Optional[Sequence[str]] = None,
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """``(instructions, [(option id, node id or reserved option, description)])`` asked at *node_id*.

    Over the node's children, one *chunk* of them, or the chunk winners given as *among*. Option ids
    are the labels a model reads (a term's Czech form); a label that repeats among the options gets
    its English form, then its node id, so no option is ever lost to a collision."""
    nodes = tree["nodes"]
    node = nodes[node_id]
    wording = tree["questions"]
    if among is not None:
        offered, instructions = list(among), wording["chunk_final"]
    elif chunk is not None:
        offered, instructions = node["chunks"][chunk], wording["L3_term"]
    else:
        offered = node["children"]
        instructions = {"root": wording["L1_facet"], "facet": wording["L2_group"]}.get(
            node["kind"], wording["L3_term"]
        )
    options: List[Tuple[str, str, str]] = []
    used = set(RESERVED)
    for child_id in offered:
        child = nodes[child_id]
        if child["kind"] == "facet":
            label = child["label"]
            description = "groups: " + "; ".join(nodes[g]["label"] for g in child["children"])
        elif child["kind"] == "group":
            label = child["label"]
            description = "for example: " + ", ".join(nodes[t]["cs"] for t in child["children"][:6])
        else:
            label, description = child["cs"], child["en"]
            if label in used:
                label = f"{child['cs']} [{child['en']}]"
        if label in used:
            label = f"{label} <{child_id}>"
        used.add(label)
        options.append((label, child_id, description))
    if node["kind"] == "term" and chunk is None:
        options.append((THIS_TERM, THIS_TERM, f"{node['cs']} itself; nothing narrower fits"))
    options.append((NONE_OF_THESE, NONE_OF_THESE, "none of the options fits the marked line"))
    return instructions, options


def node_choice(
    tree: Mapping[str, Any],
    node_id: str,
    chunk: Optional[int] = None,
    among: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """The ``choice`` asked at *node_id*, in the ``/v1/systemone`` wire format (see ``node_options``)."""
    instructions, options = node_options(tree, node_id, chunk, among)
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {o: d for o, _, d in options},
    }


def every_choice(tree: Mapping[str, Any]) -> List[Tuple[str, Optional[int], Dict[str, Any]]]:
    """``(node, chunk, choice)`` for every choice the tree can ask (chunk finals at their widest)."""
    out = []
    for node_id, node in tree["nodes"].items():
        if not node["children"]:
            continue
        if "chunks" in node:
            for index in range(len(node["chunks"])):
                out.append((node_id, index, node_choice(tree, node_id, chunk=index)))
            winners = [chunk[0] for chunk in node["chunks"]]  # one winner per chunk, at most
            out.append((node_id, None, node_choice(tree, node_id, among=winners)))
        else:
            out.append((node_id, None, node_choice(tree, node_id)))
    return out


def tree_stats(
    tree: Mapping[str, Any], edges: Mapping[str, int], dropped_groups: Sequence[str]
) -> Dict[str, Any]:
    nodes = tree["nodes"]
    term_nodes = [n for n in nodes.values() if n["kind"] == "term"]

    def depth(node_id: str) -> int:  # term levels below the group, the group's own terms being 1
        best = 0
        stack = [(node_id, 1)]
        while stack:
            current, level = stack.pop()
            best = max(best, level)
            stack += [(child, level + 1) for child in nodes[current]["children"]]
        return best

    deepest = max(depth(t) for g in nodes.values() if g["kind"] == "group" for t in g["children"])
    widest = sorted(((len(n["children"]), node_id) for node_id, n in nodes.items()), reverse=True)
    choices = every_choice(tree)
    return {
        "terms": len(term_nodes),
        "facets": sum(1 for n in nodes.values() if n["kind"] == "facet"),
        "groups": sum(1 for n in nodes.values() if n["kind"] == "group"),
        "groups_dropped": list(dropped_groups),
        "single_group_facets": sorted(
            n["label"] for n in nodes.values() if n["kind"] == "facet" and len(n["children"]) == 1
        ),
        "inner_term_nodes": sum(1 for n in term_nodes if n["children"]),
        "deepest_term_level": deepest,
        "longest_parent_chain": deepest - 1,
        "parent_edges": dict(sorted(edges.items())),
        "chunked_nodes": [node_id for node_id, n in nodes.items() if "chunks" in n],
        "nodes_over_64_children": [f"{node_id} ({size})" for size, node_id in widest if size > 64],
        "geo_gated": sorted(node_id for node_id, n in nodes.items() if n.get("geo_gated")),
        "choices": len(choices),
        "max_options_per_choice": max(len(c["criteria"]) for _, _, c in choices),
        "11-05_figures": {
            "inner_term_nodes": 797,
            "levels_below_a_group": 11,
            "note": (
                "measured by hand on 11-05; the generated tree is authoritative. The difference in inner "
                "nodes comes from how a term's parent is chosen (same facet, crossing groups, AMČR broader)"
            ),
        },
    }


# ── document type (#3; the list is shared with digital-convert#8) ────────────────────────────

DOC_TYPE_LIST = REPO_ROOT / "decisions" / "document_type.v1.json"
TEXT_VIEW = "document_text"  # the list's view for an item read as OCR text, never an image
TEXT_GROUPS = ("text",)  # the types OCR text can tell: the written documents
TEXT_INPUTS = {
    "image": "none",
    "text": "the document's OCR text (its first lines, in order), always",
}
TEXT_CATEGORIES = ("Clear", "Noisy")  # the lines whose text goes to the model, in document order
SERVED_ON = ("cluster-job", "ufal-endpoint", "local", "other")


def load_doc_types(path: Path = DOC_TYPE_LIST) -> Dict[str, Any]:
    """The text-only view of the vendored list, its questions checked and their digest recomputed.

    Photographs, aerial photographs, plans, maps and drawings cannot be told from OCR text, so the
    view offers the written-document types (group ``text``) and the abstain option, and no ``group``
    question. From a v2 list it is the list's own ``document_text`` view; from v1 it is derived the
    same way digital-convert's ``tools/build_document_types.py`` derives it (the same criteria text),
    so the digest is the same either way."""
    import decision_client as dc  # noqa: PLC0415 — vendored from atrium-page-classification

    content = json.loads(path.read_text(encoding="utf-8"))
    abstain = content["abstain"]
    offered = [t["id"] for t in content["types"] if t["group"] in TEXT_GROUPS or t["id"] == abstain]
    if "views" in content:
        if TEXT_VIEW not in content["views"]:
            raise SystemExit(
                f"{path}: no {TEXT_VIEW!r} view; this list is not shared with keyword-extract"
            )
        entry = content["views"][TEXT_VIEW]
        questions, item, inputs = entry["questions"], entry["item"], entry["inputs"]
        declared = entry["schema_digest"]
    else:
        if TEXT_VIEW not in content.get("states", {}):
            raise SystemExit(
                f"{path}: no {TEXT_VIEW!r} state; this list is not shared with keyword-extract"
            )
        page = content["questions"]["document_type"]
        questions = {
            "document_type": {
                "type": "choice",
                "instructions": page["instructions"],
                "criteria": {tid: page["criteria"][tid] for tid in offered},
            }
        }
        item, inputs = content["states"][TEXT_VIEW]["item"], TEXT_INPUTS
        declared = None
    questions = dc.questions(questions)
    digest = dc.schema_digest(questions)
    if declared is not None and declared != digest:
        raise SystemExit(
            f"{path}: the {TEXT_VIEW} questions do not match their digest; re-vendor the list"
        )
    if (
        list(questions) != ["document_type"]
        or list(questions["document_type"]["criteria"]) != offered
    ):
        raise SystemExit(
            f"{path}: the {TEXT_VIEW} view does not offer the written-document types only"
        )
    if inputs.get("image") != "none":
        raise SystemExit(
            f"{path}: the {TEXT_VIEW} view takes an image; keyword-extract reads text only"
        )
    by_id = {t["id"]: t for t in content["types"]}
    return {
        "version": content["version"],
        "view": TEXT_VIEW,
        "questions": questions,
        "item": item,
        "inputs": inputs,
        "schema_digest": digest,
        "types": [by_id[tid] for tid in offered],
        "all_types": content["types"],
        "abstain": abstain,
    }


def document_text(
    rows: Sequence[Mapping[str, Any]],
    max_chars: int = 6000,
    categories: Sequence[str] = TEXT_CATEGORIES,
) -> Tuple[str, int, int]:
    """The document's text for the state: its lines of *categories*, in order, whole lines only, up to
    *max_chars*. Returns (text, lines used, lines available). Counted here, never asked."""
    lines = [str(r.get("text") or "").strip() for r in rows if str(r.get("categ")) in categories]
    lines = [line for line in lines if line]
    used: List[str] = []
    size = 0
    for line in lines:
        extra = len(line) + (1 if used else 0)
        if size + extra > max_chars:
            break
        used.append(line)
        size += extra
    return "\n".join(used), len(used), len(lines)


def _gold_types(path: Path, doc_types: Mapping[str, Any]) -> Dict[str, str]:
    """``file -> type id`` from a CSV with ``file`` and ``document_type`` (an id, a HES number or a
    Czech label of the whole list: a catalogue type the text view does not offer still counts)."""
    by_any: Dict[str, str] = {}
    for t in doc_types["all_types"]:
        for key in (t["id"], t["hes"], t["cs"]):
            by_any[str(key).casefold()] = t["id"]
    out: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("document_type", "")).strip()
            if value:
                if value.casefold() not in by_any:
                    raise SystemExit(f"{path}: {value!r} is not a type of the list")
                out[str(row["file"])] = by_any[value.casefold()]
    return out


def doc_type_columns(doc_types: Mapping[str, Any]) -> List[str]:
    types = [f"P_type_{t['id']}" for t in doc_types["types"]]
    return (
        ["file", "lines_used", "lines_available", "chars"]
        + ["document_type", "type_group", "confidence", "top3", "lane"]
        + ["gold_type", "gold_group", "gold_offered"]
        + types
        + ["status", "error", "latency_ms", "input_tokens", "attempts", "schema_digest"]
    )


def decide_doc_type(
    client: Any,
    path: Path,
    doc_types: Mapping[str, Any],
    max_chars: int = 6000,
    gold: Optional[Mapping[str, str]] = None,
    accept_at: Optional[float] = None,
    review_below: Optional[float] = None,
) -> Dict[str, Any]:
    """One row of ``DM_DOC_TYPE.csv`` for one ``DOC_LINE_CATEG`` CSV. Never raises for a server problem.

    The request carries the document's text and no image."""
    import decision_client as dc  # noqa: PLC0415

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    text, used, available = document_text(rows, max_chars)
    group_of = {t["id"]: t["group"] for t in doc_types["all_types"]}
    offered = {t["id"] for t in doc_types["types"]}
    gold_type = (gold or {}).get(path.stem, "")
    row: Dict[str, Any] = {
        "file": path.stem,
        "lines_used": used,
        "lines_available": available,
        "chars": len(text),
        "gold_type": gold_type,
        "gold_group": group_of.get(gold_type, ""),
        "gold_offered": int(gold_type in offered) if gold_type else "",
    }
    if not text:
        d = dc.Decision(
            "rejected", doc_types["schema_digest"], error="no Clear or Noisy line to read"
        )
    else:
        state = {"item": doc_types["item"], "text": text, "lines": used}
        d = client.decide(state, doc_types["questions"])
    kind = d.answers.get("document_type")
    row["document_type"] = kind.choice if kind else ""
    row["type_group"] = group_of.get(row["document_type"], "")
    row["confidence"] = kind.confidence if kind else ""
    if kind:
        best = sorted(kind.probabilities.items(), key=lambda item: -item[1])[:3]
        row["top3"] = "|".join(f"{k}:{p:.4f}" for k, p in best)
    else:
        row["top3"] = ""
    for t in doc_types["types"]:
        row[f"P_type_{t['id']}"] = kind.probabilities.get(t["id"], "") if kind else ""
    row["lane"] = (
        dc.lane(kind.confidence, accept_at, review_below)
        if kind and accept_at is not None and review_below is not None
        else ""
    )
    row.update(
        {
            "status": d.status,
            "error": d.error or "",
            "latency_ms": d.latency_ms if d.latency_ms is not None else "",
            "input_tokens": d.input_tokens if d.input_tokens is not None else "",
            "attempts": d.attempts,
            "schema_digest": d.schema_digest,
        }
    )
    return row


def summarise_doc_types(rows: Sequence[Mapping[str, Any]], abstain: str) -> Dict[str, Any]:
    """Counts and, where gold is given, accuracy over all graded documents and over those whose
    catalogue type the text view offers (a photograph's type cannot be chosen from text)."""
    ok = [r for r in rows if r.get("status") == "ok"]
    latencies = sorted(float(r["latency_ms"]) for r in ok if str(r.get("latency_ms", "")) != "")
    out: Dict[str, Any] = {
        "documents": len(rows),
        "statuses": dict(Counter(str(r.get("status")) for r in rows)),
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
        "types": dict(Counter(r["document_type"] for r in ok)),
        "abstained": round(sum(r["document_type"] == abstain for r in ok) / len(ok), 4)
        if ok
        else None,
    }
    graded = [r for r in ok if r.get("gold_type")]
    if graded:
        offered = [r for r in graded if str(r.get("gold_offered")) == "1"]
        out["gold"] = {
            "documents": len(graded),
            "gold_not_offered": len(graded) - len(offered),
            "type_accuracy": round(
                sum(r["document_type"] == r["gold_type"] for r in graded) / len(graded), 4
            ),
            "type_accuracy_offered": round(
                sum(r["document_type"] == r["gold_type"] for r in offered) / len(offered), 4
            )
            if offered
            else None,
            "type_group_accuracy": round(
                sum(r["type_group"] == r["gold_group"] for r in graded) / len(graded), 4
            ),
        }
    return out


def served_on(url: Optional[str], given: Optional[str] = None) -> str:
    """Where the documents went, for the paradata, never the URL: the internal endpoint is not published."""
    if given:
        return given
    from urllib.parse import urlsplit  # noqa: PLC0415

    host = urlsplit(url or "").hostname or ""
    return "local" if host in ("127.0.0.1", "localhost", "::1") else "other"


def run_doc_types(
    client: Any,
    inputs: Sequence[Path],
    out: Path,
    doc_types: Mapping[str, Any],
    max_chars: int = 6000,
    gold: Optional[Mapping[str, str]] = None,
    accept_at: Optional[float] = None,
    review_below: Optional[float] = None,
    resume: bool = False,
    progress: bool = False,
) -> List[Dict[str, Any]]:
    """Decide every document; rows are appended to *out* as they come. Returns the new rows."""
    if out.exists() and not resume:
        raise SystemExit(f"{out} exists; pass --resume to continue it or choose another --out")
    done = set()
    if resume and out.is_file():
        with out.open(newline="", encoding="utf-8") as handle:
            done = {r["file"] for r in csv.DictReader(handle)}
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    rows: List[Dict[str, Any]] = []
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=doc_type_columns(doc_types), extrasaction="ignore"
        )
        if new_file:
            writer.writeheader()
        for index, path in enumerate(inputs, start=1):
            if path.stem in done:
                continue
            row = decide_doc_type(client, path, doc_types, max_chars, gold, accept_at, review_below)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if progress:
                print(
                    f"[{index}/{len(inputs)}] {path.stem}: {row['status']} {row['document_type']}",
                    file=sys.stderr,
                )
    return rows


def main_doc_type(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import decision_client as dc  # noqa: PLC0415

    if not args.input:
        parser.error("--doc-type needs --input (a DOC_LINE_CATEG directory or one CSV)")
    if (args.accept_at is None) != (args.review_below is None):
        parser.error("--accept-at and --review-below go together")
    inputs = [args.input] if args.input.is_file() else sorted(args.input.glob("*.csv"))
    if not inputs:
        parser.error(f"no CSV under {args.input}")
    client = dc.DecisionClient(
        args.url,
        args.model,
        timeout_s=args.timeout,
        retries=args.retries,
        use_env_proxy=not args.no_proxy,
    )
    if not client.endpoint or not client.model:
        parser.error("set --url/--model (or DECISION_MODEL_URL/DECISION_MODEL_NAME)")
    doc_types = load_doc_types()
    gold = _gold_types(args.gold, doc_types) if args.gold else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out = args.out or REPO_ROOT / "result" / "dm" / f"{stamp}_DM_DOC_TYPE.csv"
    where = served_on(args.url, args.served_on)

    from atrium_paradata import ParadataLogger  # noqa: PLC0415

    paradata = ParadataLogger(
        program="keyword-extract",
        config={
            "script": "dm_run",
            "mode": "doc_type",
            "decision_model": args.model,
            "decision_revision": args.model_revision,
            "served_on": where,
            "decision_schema_digest": doc_types["schema_digest"],
            "list": f"{DOC_TYPE_LIST.relative_to(REPO_ROOT)} v{doc_types['version']} ({TEXT_VIEW})",
            "max_chars": args.max_chars,
            "gold": str(args.gold) if args.gold else None,
            "accept_at": args.accept_at,
            "review_below": args.review_below,
        },
        paradata_dir=str(out.parent / "paradata"),
        output_types=["csv"],
        config_dir=str(REPO_ROOT),
    )
    paradata.log_component("decision_model")
    paradata.log_component("amcr_vocab")
    rows = run_doc_types(
        client,
        inputs,
        out,
        doc_types,
        args.max_chars,
        gold,
        args.accept_at,
        args.review_below,
        args.resume,
        progress=True,
    )
    paradata.log_success("csv", len(rows))
    with out.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    summary = summarise_doc_types(all_rows, doc_types["abstain"])
    summary.update(
        {
            "model": args.model,
            "model_revision": args.model_revision,
            "served_on": where,
            "schema_digest": doc_types["schema_digest"],
            "table": str(out),
        }
    )
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paradata.finalize(input_total=len(inputs), processed_total=summary["statuses"].get("ok", 0))
    print(
        json.dumps(
            {k: summary[k] for k in ("documents", "statuses", "types")},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decision models over the controlled vocabulary and the document type (#3)."
    )
    parser.add_argument(
        "--dump-tree", type=Path, metavar="OUT", help="write the descent tree (JSON) and exit"
    )
    parser.add_argument(
        "--lines", action="store_true", help="line mode, the vocabulary descent (not yet)"
    )
    doc = parser.add_argument_group("the document type (--doc-type)")
    doc.add_argument(
        "--doc-type",
        action="store_true",
        help="ask each document's AMČR type from its text (decisions/document_type.v1.json)",
    )
    doc.add_argument("--input", type=Path, help="a DOC_LINE_CATEG directory or one CSV")
    doc.add_argument("--max-chars", type=int, default=6000, help="the text budget per document")
    doc.add_argument(
        "--gold", type=Path, help="a CSV of file,document_type (id, HES or Czech label)"
    )
    doc.add_argument("--accept-at", type=float, help="lane threshold (set on dev)")
    doc.add_argument("--review-below", type=float, help="lane threshold (set on dev)")
    doc.add_argument(
        "--out", type=Path, help="the table (default result/dm/<stamp>_DM_DOC_TYPE.csv)"
    )
    doc.add_argument(
        "--resume", action="store_true", help="continue --out, skipping documents in it"
    )
    server = parser.add_argument_group("server")
    server.add_argument(
        "--url", default=os.environ.get("DECISION_MODEL_URL"), help="($DECISION_MODEL_URL)"
    )
    server.add_argument(
        "--model", default=os.environ.get("DECISION_MODEL_NAME"), help="($DECISION_MODEL_NAME)"
    )
    server.add_argument("--model-revision", default="", help="the served build or HF revision")
    server.add_argument("--served-on", choices=SERVED_ON, help="where the documents go")
    server.add_argument("--timeout", type=float, default=_env_float("DECISION_TIMEOUT_S", 120.0))
    server.add_argument("--retries", type=int, default=2)
    server.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args(argv)
    if args.lines:
        print(
            "line mode (the vocabulary descent over the tree) is not written yet (#3, Phase C)",
            file=sys.stderr,
        )
        return 2
    if args.doc_type:
        return main_doc_type(args, parser)
    if not args.dump_tree:
        parser.error("nothing to do: pass --dump-tree OUT or --doc-type")
    tree = build_tree()
    args.dump_tree.parent.mkdir(parents=True, exist_ok=True)
    args.dump_tree.write_text(
        json.dumps(tree, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    stats = {
        k: v for k, v in tree["stats"].items() if k not in ("geo_gated", "nodes_over_64_children")
    }
    print(json.dumps({"digest": tree["digest"], **stats}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
