# Using keyword-extract

How to install keyword-extract, run the service, run the batch command line for the statistical kind, run the
controlled kind from the command line, and use the Docker images. What each kind does:
[how it works](how-it-works.md).

**On this page:**

- [Set it up](#set-it-up)
- [Run the service](#run-the-service)
- [Extract keywords from a folder](#extract-keywords-from-a-folder), and [which setting wins](#which-setting-wins)
- [Run the controlled kind from the command line](#run-the-controlled-kind-from-the-command-line)
- [Run the Docker images](#run-the-docker-images)
- [The command line and the service](#the-command-line-and-the-service): where they differ
- [Decision models, offline](#decision-models-offline): `dm_run.py`, the descent tree and the document type

## Set it up

```bash
git clone https://github.com/ufal/atrium-keyword-extract.git
cd atrium-keyword-extract
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt                      # KeyBERT, YAKE, the shared contract
pip install -r service/requirements.txt              # + the service (fastapi, uvicorn)
```

KeyBERT downloads its embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) from the Hugging Face Hub on first
use; set `HF_HOME` to keep the cache. A GPU is used when `torch` finds one. The environment of the service is
[`.env.example`](../.env.example).

## Run the service

[`service/README.md`](../service/README.md) is the reference: the endpoints, the options, the response, the limits,
the errors and the OpenAPI contract. In short:

```bash
python -m service.api                  # PORT / HOST; default 0.0.0.0:8000

curl -s -X POST localhost:8000/extract_keywords \
  -F "document_json=@CTX000000001.document.json;type=application/json" \
  -F kind=statistical -F method=keybert -F num_keywords=10

# both kinds, with the controlled one's backend configured
OPENROUTER_API_KEY=sk-... OPENROUTER_MODEL=openai/gpt-4o-mini python -m service.api
curl -s -X POST localhost:8000/extract_keywords \
  -F "document_json=@CTX000000001.document.json;type=application/json" -F kind=both
```

## Extract keywords from a folder

> [!NOTE]
> `keywords.py` is the command-line form of the statistical kind, for a whole collection at once. It reads what the
> [service](#run-the-service) reads, with the same code (`kw_inputs.py`): the CoNLL-U and TEITOK files
> [nlp-enrich](https://github.com/ufal/atrium-nlp-enrich) writes, ATRIUM records (`*.document.json`), line tables
> (`.csv`) and plain text (`.txt`). It writes CSV tables and, when asked, the records. Both run the same three methods.

Run `keywords.py` on a directory of such files; other files are passed over.

### Which setting wins

`keywords.py` takes its settings from three places, from the highest priority to the lowest:

1. **Command-line flags** (e.g., `-m yake`, `-w 3`) always override everything else.
2. **`kw_config.txt`** (the `[DEFAULTS]` section) is read automatically if placed next to the script.
3. **Hardcoded fallbacks** are used if no config file or flags are provided.

So once your settings are in `kw_config.txt`, you can simply run:

```bash
python3 keywords.py
```

### Choose a method

| Flag value            | Method                                                   | Dependencies                                | Score semantics                       | Best for                                  |
|-----------------------|----------------------------------------------------------|---------------------------------------------|---------------------------------------|-------------------------------------------|
| `legacy`              | NOUN/PROPN/ADJ lemma frequency, named after KER          | none (stdlib only)                          | raw occurrence count                  | reproducing original ATRIUM results       |
| `yake`                | YAKE — unsupervised statistical, CPU-only (**AGPL-3.0**) | `pip install yake`                          | normalised inverse YAKE score, [0, 1] | fast CPU runs, no model download          |
| `keybert` *(default)* | KeyBERT — embedding-based, GPU-accelerated               | `pip install keybert sentence-transformers` | cosine similarity, [0, 1]             | highest semantic quality, GPU recommended |

You can override any `kw_config.txt` setting via the command line:

```bash
python3 keywords.py -i <input_dir> -m <method> -l <lang> -w <integer> \
                    -n <integer> -d <output_dir> -o <output_file>.csv
```

All available flags:

| Flag | Long form                 | Default in `kw_config.txt`                     | Description                                                                               |
|------|---------------------------|------------------------------------------------|-------------------------------------------------------------------------------------------|
| `-i` | `--input_dir`             | `data_samples/UDP`                             | The folder to read: CoNLL-U, TEITOK, records, line tables, `.txt`                         |
| `-m` | `--method`                | `keybert`                                      | Backend: `legacy`, `yake`, or `keybert`                                                   |
| `-l` | `--lang`                  | `cs`                                           | Language code for YAKE stopwords (`cs`, `en`, `de`, …). Ignored by `legacy` and `keybert` |
| `-w` | `--max_words`             | `3`                                            | Maximum words per keyword phrase (n-gram upper bound)                                     |
| `-n` | `--num_keywords`          | `20`                                           | Number of keywords to extract per document                                                |
| `-d` | `--per_doc_out_dir`       | `data_samples/KW_PER_DOC_{METHOD}`             | Output directory for per-document CSV files (`{METHOD}`: `L`, `Y` or `KB`)                |
| `-o` | `--output_file`           | `data_samples/keywords_summary_{method}.csv`   | Master keywords CSV (`{method}`: `l`, `y` or `kb`)                                        |
|      | `--per-page`              | *(off)*                                        | Also extract each page's keywords, for inputs with pages (the service's `per_page`)       |
| `-p` | `--per_page_out_dir`      | `data_samples/KW_PER_PAGE_{METHOD}`            | Output directory for the per-page CSV files (`page`, `keyword`, `score`)                  |
|      | `--document-json-out-dir` | *(none)*                                       | Write each document's record with its `keywords` block here, as the service writes it     |
|      | `--document-json-dir`     | *(none)*                                       | Baselines `<doc_id>.document.json` for inputs that are not records themselves             |
|      | `--paradata-dir`          | `PARADATA_DIR`, else `<input_dir>/../paradata` | Where the run's paradata file goes                                                        |
|      | `--keybert-model`         | `paraphrase-multilingual-MiniLM-L12-v2`        | Sentence-Transformer model name (KeyBERT only)                                            |
|      | `--no-mmr`                | *(False)*                                      | Disable Maximal Marginal Relevance diversification (KeyBERT only)                         |
|      | `--diversity`             | `0.5`                                          | MMR diversity parameter, 0 = max relevance → 1 = max diversity (KeyBERT only)             |
|      | `--batch_size`            | `16`                                           | Documents per KeyBERT batch (KeyBERT only)                                                |
|      | `--workers`               | `0` *(Auto / CPU count)*                       | Parallel worker processes. Auto-forced to 1 for KeyBERT + GPU                             |

Examples:

**YAKE** — Czech, up to 3-word phrases, 20 keywords per document

```bash
python3 keywords.py -i OUTPUT_DIR/UDP -m yake -l cs -w 3 -n 20 \
        -o keywords_summary.csv -d KW_PER_DOC
```

**KeyBERT** — multilingual model, GPU-accelerated (the default)

```bash
python3 keywords.py -i OUTPUT_DIR/UDP -m keybert -w 3 -n 20 \
        --keybert-model paraphrase-multilingual-MiniLM-L12-v2 \
        -o keywords_summary.csv -d KW_PER_DOC
```

**Legacy (KER)** — (English/Czech) the original ATRIUM lemma-frequency approach, no extra dependencies. It is named
after ÚFAL's KER but, unlike KER, does not weight lemmas by IDF ([licences](reference/licences.md#licences))

```bash
python3 keywords.py -i OUTPUT_DIR/UDP -m legacy -n 20 \
        -o keywords_summary.csv -d KW_PER_DOC
```

**The record** — per-page keywords into the records, as the service writes them:

```bash
python3 keywords.py -i records/ -m yake --per-page --document-json-out-dir records_kw/
```

> [!WARNING]
> For **KeyBERT with a GPU**, the script automatically forces `--workers 1` to
> prevent competing CUDA context initialisation across subprocesses.  On CPU,
> any worker count is safe.

## Run the controlled kind from the command line

The three paths and what they read and write are in [how it works](how-it-works.md#the-controlled-kind-and-the-vocabulary).

```bash
pip install -r requirements_remote.txt
export OPENROUTER_API_KEY=sk-...
python openrouter_client.py --input CTX000000001.document.json --model openai/gpt-4o-mini \
  --document-json-out CTX000000001.kw.document.json
python ollama_client.py --input data_samples/DOC_LINE_CATEG --model qwen2.5:14b
```

The research run on a GPU node:

```bash
pip install -r requirements.txt -r requirements_llm.txt   # vLLM brings its own torch pin
$EDITOR llm_config.txt     # MODEL_KEY; INPUT_DIR (line CSVs or *.teitok.xml), OUTPUT_DIR, PARADATA_DIR
python llm_run.py --download   # where huggingface.co is reachable: MODEL_KEY into the HF cache
python llm_run.py          # or: python llm_run.py <another config>
```

A GPU node without internet access (`HF_HUB_OFFLINE=1`) loads a model only from the Hugging Face cache. Fetch it first
on a machine that has access, into the same `HF_HOME`; `--download KEY ...` takes several keys. The download fetches
what a run reads (configs, tokenizer, weights; not README or LICENSE) and skips files already cached, for which
huggingface_hub then reports 0 B. It ends with what the cache holds. A run whose model files are not all cached stops
before loading and names what is missing.

What the research run does:

- It writes `<OUTPUT_DIR>_<model>/<doc_id>_enriched.json` per document, and `<doc_id>_enriched.abort.json` after ten
  consecutive errors.
- It skips a document whose output already exists, so a stopped job resumes. To redo an aborted document, delete its
  two files.
- It writes a paradata record that names the vocabulary build and the sha256 of the prompt it used.

The whole prompt is about 58k tokens. A model served with a smaller window sees only a prefix of the vocabulary, and
the paradata records how many terms were cut. The large vLLM models do this at their default `MAX_MODEL_LEN` of 16384.

To keep the run fast and within memory:

- vLLM computes the prompt's shared part once and reuses it (prefix caching). So does `BACKEND=transformers` with
  `REUSE_PROMPT_CACHE=true`; without it, every line re-reads the whole prompt.
- `PREFILL_CHUNK_SIZE` keeps the transformers prefill within a 24 GB card.
- Hybrid models (Qwen3.5, Qwen3.6) need `flash-linear-attention`, which `requirements_llm.txt` lists. Without it their
  linear-attention layers run on a reference kernel that is far slower and does not fit.

The `llm` image carries no sample inputs, and whatever it writes outside `/data` goes with the container. Point
`INPUT_DIR`, `OUTPUT_DIR` and `PARADATA_DIR` under `/data` in the `llm_config.txt` the compose file mounts.

## Run the Docker images

The Dockerfile builds three images, one per target:

| Target | Image                                     | What it is                                                          |
|--------|-------------------------------------------|---------------------------------------------------------------------|
| `api`  | `ghcr.io/ufal/atrium-keyword-extract-api` | the service, both kinds — **the production image** (no LLM weights) |
| `base` | `ghcr.io/ufal/atrium-keyword-extract`     | the batch CLI (`keywords.py`)                                       |
| `llm`  | `ghcr.io/ufal/atrium-keyword-extract-llm` | the research LLM batch run, GPU                                     |

The batch image passes its arguments to `keywords.py` (`docker run IMAGE -i /data/UDP -m yake`). Since `v1.6.5-beta`
its entry point is the hub-canonical `container_entrypoint.py`, which names `keywords.py` by its absolute path: a
first argument that names a shell or Python runs that program instead (`docker run IMAGE /bin/sh job.sh`), which is
how Galaxy starts a tool's container. Only those interpreters run, so an input folder that happens to be called `sh`
is still a folder; the `llm` image does the same for `llm_run.py`.

```bash
docker compose --profile api up                       # the service on :8000
docker compose run --rm kw -i /data/UDP -m yake       # the batch CLI
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml --profile llm run --rm kw-llm
```

## The command line and the service

`keywords.py` and `POST /extract_keywords` read every input through [kw_inputs.py](../kw_inputs.py). They run the same
methods (`keywords.extract_from_texts`) and write the record's `keywords` block through the same writer
(`llm_client_shared.write_document_record`). So the same input gives the same keywords, per document and per page,
and the same licence components in the paradata; [tests/test_cli_service_parity.py](../tests/test_cli_service_parity.py)
checks this. Both take their defaults from `kw_config.txt`: `METHOD` and `NUM_KEYWORDS` (the service's
`DEFAULT_KW_METHOD` setting overrides its method), and the KeyBERT model, `MAX_WORDS`, MMR and diversity. Where they
differ:

| What                         | Command line                                                       | Service                                                           | Why they differ                                              |
|------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------------|--------------------------------------------------------------|
| How much per run             | a folder                                                           | one document per request                                          | the service is stateless                                     |
| The record                   | `*.document.json` in the folder; `--document-json-out-dir`         | `document_json`                                                   |                                                              |
| CoNLL-U, TEITOK, CSV, `.txt` | files in the folder                                                | the `file` part (with `document_json`, the record gets the block) |                                                              |
| Lines as JSON                | none; write them to a `.txt`                                       | `POST /extract_keywords_text`                                     | an HTTP convenience                                          |
| Per page                     | `--per-page` (off by default)                                      | `per_page` (on by default)                                        | the CSV outputs of a batch run stay as they were             |
| Pages of a CoNLL-U file      | none                                                               | none                                                              | CoNLL-U carries no pages: send the record or the TEITOK file |
| KER without lemmas           | an empty row                                                       | 422                                                               | a batch run goes on; a call is told why it got nothing       |
| A document with no text      | an empty row                                                       | 422                                                               | the same                                                     |
| Language                     | `-l`: any language YAKE has stop words for                         | `lang`: `cs` only                                                 | the service takes Czech only                                 |
| KeyBERT and YAKE settings    | `-w`, `--keybert-model`, `--no-mmr`, `--diversity`, `--batch_size` | `kw_config.txt` in the image                                      | deployment settings, not request fields                      |
| Limits                       | none                                                               | `MAX_DOCUMENT_WORDS` (413), `MAX_KEYWORDS` (422), `MAX_UPLOAD_MB` | the service bounds what one request may cost                 |
| The controlled kind          | `openrouter_client.py`, `ollama_client.py`; `llm_run.py` on a GPU  | `kind=controlled\|both`                                           | the LLM backend is configured per deployment in the service  |
| Outputs                      | CSV tables, the records                                            | JSON, with the record                                             |                                                              |
| Paradata                     | a file in `--paradata-dir` (program `keyword-extract`)             | the response's `paradata`, a `CreateAction`                       | the service keeps no files                                   |

KER (`legacy`) counts lemmas, and where they come from depends on the input. CoNLL-U and TEITOK give every noun's,
proper noun's and adjective's lemma. A record gives `lines[].lemma`, which no ATRIUM tool writes today. A CSV or a
`.txt` gives none. The pages' lemmas come from a record only.

## Decision models, offline

`dm_run.py` tries decision models on two questions of [#3](https://github.com/ufal/atrium-keyword-extract/issues/3)
(atrium-project#83). A decision model reads a state, such as a text, and questions with fixed options, and returns
a probability for every option; it writes no text. The tool is offline and experimental: the service does not run
it, no image's entry point reaches it, and nothing it finds reaches the record. Run it from the repository root,
set up as [above](#set-it-up).

### The descent tree: `--dump-tree`

A decision model cannot take the whole vocabulary at once (4,711 terms; a choice takes at most 255 options), so the
line mode of #3 is to walk down it, one choice a step: is the line content at all, which facet, the geographic
gate, which group, then down the terms' own tree. Every step also offers `none_of_these`, so an answer is always a
vocabulary term or unresolved. `--dump-tree OUT` builds that tree and writes it as JSON; the line mode itself
(`--lines`) is not written yet and exits 2.

```bash
python3 dm_run.py --dump-tree /tmp/dm_tree.json
```

The tree is built from files the vocabulary build already writes (`data_samples/vocab/union_nested.json`,
`specificity_pairs.csv`, `amcr_flat.json`, `data_samples/taxonomy_config.json`) and from the prompt's blocks and
settings (`prompts/system_prompt.txt`, `llm_config.txt`). The file records each input's SHA-256, the questions'
wording, every node and the tree's statistics, and a digest of all of it but the time it was made. The command
prints the digest and the statistics. How a term's parent is chosen, and how a node with too many options is
split, is in the module's docstring.

### The document type: `--doc-type`

`--doc-type` asks a decision model the AMČR document type (`dokument_typ`) of each document, from its text and
never an image. The input is a `DOC_LINE_CATEG` folder, one line table per document with its `text` and `categ`
columns, or one such CSV. A request carries the document's `Clear` and `Noisy` lines, in order, whole lines up to
`--max-chars`.

The list of types is atrium-digital-convert's `decisions/document_type.v2.json`, which it builds from this
repository's `amcr_flat.json`; the copy here is pinned by its SHA-256 (`tests/test_vendored_doctype_parity.py`).
The client, `decision_client.py`, is atrium-page-classification's, pinned the same way
(`tests/test_vendored_decision_parity.py`). The tool asks the list's text-only view, `document_text`: one
`document_type` question over the 6 written-document types and `unspecified` (`neurčený dokument`, the abstain
option), with no `group` question. Photographs, aerial photographs, plans, maps and drawings cannot be told from
OCR text, so the view does not offer them. `dokument_typ` is withheld from the keyword vocabulary, so a type is
never a keyword.

```bash
export DECISION_MODEL_URL=...      # the server; never recorded
python3 dm_run.py --doc-type --input data_samples/DOC_LINE_CATEG --out /tmp/dm/DM_DOC_TYPE.csv \
    --model clef-oQ8e --served-on ufal-endpoint
```

| Option                          | Meaning                                                                                                                       |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `--input`                       | a `DOC_LINE_CATEG` folder (its `*.csv`) or one CSV (required)                                                                 |
| `--out`                         | the table (default `result/dm/<stamp>_DM_DOC_TYPE.csv`)                                                                       |
| `--max-chars`                   | the text a request carries, in whole lines (default 6000)                                                                     |
| `--gold`                        | a CSV of `file,document_type` to score against; a type as its id, its HES number or its Czech label, from the whole list      |
| `--accept-at`, `--review-below` | the lane thresholds, both or neither (below)                                                                                  |
| `--resume`                      | continue `--out`, skipping the documents it holds; without it, an existing table stops the run                                |
| `--url`, `--model`              | the server and the model (default `DECISION_MODEL_URL`, `DECISION_MODEL_NAME`)                                                |
| `--model-revision`              | the served build or Hugging Face revision, for the paradata                                                                   |
| `--served-on`                   | where the documents go: `cluster-job`, `ufal-endpoint`, `local` or `other` (default `local` for a loopback URL, else `other`) |
| `--timeout`, `--retries`        | per request: seconds (default `DECISION_TIMEOUT_S`, else 120) and retries (default 2)                                         |
| `--no-proxy`                    | ignore the proxy variables for this server                                                                                    |

**The lanes.** With both thresholds, each row gets a `lane`: `accept` at or above `--accept-at`, `review` below
`--review-below`, `escalate` in between. Each threshold is between 0 and 1, and `--review-below` is not above
`--accept-at`. The thresholds belong to one model revision and one question wording, so set them on dev data.

A bad option ends the run before the first request, with exit code 2: `--doc-type` without `--input`, a threshold
out of range or without the other, no CSV at `--input`, or no server or model.

It writes:

* **The table**, `DM_DOC_TYPE.csv`, a row per document: `lines_used`, `lines_available` and `chars` (the text
  sent); the chosen `document_type` with `type_group`, `confidence`, `top3`, `lane` and every offered type's
  probability (`P_type_*`); the gold, `gold_type`, `gold_group` and `gold_offered` (whether the view offers that
  type); and the request's `status`, `error`, `latency_ms`, `input_tokens`, `attempts` and `schema_digest`. Rows
  are written as the answers come. A document with no `Clear` or `Noisy` line gets a `rejected` row and no request.
* **The summary** beside it, the table's name ending in `.summary.json`: the statuses, the median latency, the
  spread of types and the share that abstained; with gold, the type and type-group accuracy over every graded
  document, and the type accuracy over those whose type the view offers.
* **The paradata record** in `paradata/` beside it: program `keyword-extract`, components `decision_model` and
  `amcr_vocab`, and the model, its revision, the list with its version and view, the digest, `--max-chars`,
  `--gold` and the thresholds. The server is recorded by its kind (`--served-on`), never by its URL.
