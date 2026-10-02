# FKG.in Recipe Extraction Verification Pipeline

A 5-stage pipeline for verifying LLM-extracted recipe JSON against its source text:

1. **Schema** - structural validation against the recipe JSON schema.
2. **Vocabulary** - flags ingredients not found in the known ingredient vocabulary.
3. **Numerical consistency** - flags ingredient weight/quantity outliers via IQR bounds.
4. **Semantic coherence** - a Set Transformer over ingredient embeddings flags ingredients that don't belong.
5. **Source fidelity** - NLI + relation extraction + rule-based graph alignment check whether the extracted quantities/forms/existence are entailed by the source text.

No trained weights ship with this repository. You must train the three models under `Models/` before the pipeline will run end to end (the pipeline will error without them).

See `CASE_WALKTHROUGH.md` for a worked example of the full detection + resolution loop running on a single recipe, end to end.

## Data availability

Our scraped recipe corpus, and everything derived from it, is not included in this repository. What ships is a one-row sample of each input, to show the format and let you dry-run the pipeline:

- `data/eval_sample.json` - a single `(source text, extracted JSON)` pair.
- `data/vocab/cleaned_ingredients_n_v2.txt` - two hand-written entries in the vocabulary format. With only these, stage 2 flags almost every ingredient and stage 3 has weight bounds for almost none, so those two stages are a format demo until you supply a full vocabulary.

Not included:

- **The full ingredient vocabulary** (stages 2 and 3). It is described in our paper: https://ceur-ws.org/Vol-3882/ifow-4.pdf. Build your own in the format of the sample file, or request ours (below).
- **The 500-recipe evaluation sample** behind `RESULTS.md`. Those numbers cannot be reproduced without it; `RESULTS.md` documents how they were produced.
- **The auto-corrector's corrections database** (`data/corrections_db.jsonl`, only read when `AUTO_CORRECT_ENABLED=true`). It is JSON Lines with one record per human correction: `recipe_id`, `source_text_snippet`, `original_error_path`, `patch`, and `context_embedding` (the `thenlper/gte-large` embedding of the snippet). Without the file the auto-corrector does nothing. You can build one from corrections recorded in the correction tool (section 6).
- **The recipe-card JSONs** for training the semantic-coherence model (section 1, `suspicious_ingredient`).

To request any of the above, email foodcomputing.ashoka@gmail.com, saransh.gupta@ashoka.edu.in, or armaan.shah@alumni.ashoka.edu.in.

## Citation

This repository accompanies:

Saransh Kumar Gupta, Armaan Shah, Lipika Dey, Partha Pratim Das, and Ramesh Jain.
Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge.
arXiv:2608.29249, 2026.
https://doi.org/10.48550/arXiv.2608.29249

If you use this software in academic or scientific work, please cite the paper above and, where appropriate, the specific software release archived on Zenodo.

Software: https://doi.org/10.5281/zenodo.23097205

## Setup

```
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` as needed (see the environment variable table below). Only `LLM_PROVIDER`/`GEMINI_API_KEY`/`OPENAI_API_KEY` matter if you enable the auto-corrector; everything else has a sane default.

---

## 1. Getting training data

We have not included our training data with this repository (see "Data availability" above). `ingredient_ner` and `relation_extraction` use publicly available datasets with minor augmentations; `suspicious_ingredient` needs recipe-card JSONs of the shape described below, which you must supply.

### `ingredient_ner` - needs a CoNLL-format NER file

A tab/space-separated CoNLL file, one token-tag pair per line, blank lines separating sentences, tags using the `B-`/`I-` prefix convention over the label set `ING`, `QUANTITY`, `STATE`, `UNIT`, `PRODUCT`. We used "https://figshare.com/articles/dataset/Food_Ingredient_Named-Entity_Data_Construction_using_Semi-supervised_Multi-model_Prediction_Technique/20222361". 

### `relation_extraction` - needs TASTEset

A CSV with (at minimum) `ingredients` and `ingredients_entities` columns, where `ingredients_entities` is a Python-literal list of `{"entity": ..., "type": ..., "span": "(start, end)"}` dicts with types `FOOD`/`QUANTITY`/`UNIT`. This matches the public **TASTEset** recipe entity dataset (1,000 manually annotated recipes, ~19,500 entities): https://github.com/taisti/TASTEset-2.0

### `suspicious_ingredient` - needs recipe-card JSON files

A directory of `*.json` files, each shaped `{"recipes": [...]}`, where each recipe has an `ingredients` list of `{"heading": ..., "items": {<ingredient name>: {"category": ..., ...}}}` groups (or `items` as a list of `{"ingredient": ..., "category": ..., ...}` - both shapes are tolerated). Example of one ingredient group entry:

```json
{
    "heading": "For the Spicy Red Beans",
    "items": {
        "Rajma": {
            "category": "PulsesAndLegumes",
            "form": "cooked",
            "quantity": 1.0,
            "unit": "cup",
            "estimated_weight_in_grams": 177.0,
            "notes": "Large Kidney Beans"
        },
        "Garlic": {
            "category": "Vegetables",
            "form": "finely chopped",
            "quantity": 4.0,
            "unit": "count",
            "estimated_weight_in_grams": 40.0,
            "notes": "NA"
        }
    }
}
```

Each recipe also carries top-level metadata alongside `ingredients` (`recipeID`, `recipeItem`, `recipeURL`, `prepTime`, `fermentTime`, `cookTime`, `totalTime`, `cuisine`, `course`, `difficulty`, `diet`, `servings`) - `prepare_dataset.py` only reads the `ingredients` groups, but the full shape is what you'll get scraping recipe-card sites in this format.

---

## 2. Training the models

Run in this order - `suspicious_ingredient`'s dataset prep imports the trained `ingredient_ner` model, so NER must be trained first.

```bash
# --- ingredient_ner ---
python -m Models.ingredient_ner.prepare_dataset --conll path/to/your.conll --output-dir data/ingredient_ner --augment --add-negatives
python -m Models.ingredient_ner.train --data-dir data/ingredient_ner --output-dir Models/ingredient_ner/out
# wraps `spacy train` with Models/ingredient_ner/config.cfg (roberta-base + spaCy NER head, 20k steps, GPU).
# roberta-base is fetched from Hugging Face on first run. Best checkpoint lands in Models/ingredient_ner/out/model-best.

# --- relation_extraction ---
python -m Models.relation_extraction.prepare_dataset --tasteset-csv path/to/TASTEset.csv --output Models/relation_extraction/re_train_data.jsonl
python -m Models.relation_extraction.train --data Models/relation_extraction/re_train_data.jsonl --output-dir Models/relation_extraction/checkpoints
# trainer writes the final model to Models/relation_extraction/checkpoints/finetuned_re_model

# --- suspicious_ingredient (run after ingredient_ner above) ---
python -m Models.suspicious_ingredient.prepare_dataset --recipe-dir path/to/recipe_card_jsons --output-dir Models/suspicious_ingredient
python -m Models.suspicious_ingredient.train --data Models/suspicious_ingredient/training_data_with_negatives.json --embeddings Models/suspicious_ingredient/ingredient_embeddings.pt --output Models/suspicious_ingredient/best_model_classification.pt
```

Notes:
- All three `train.py` scripts take `--epochs`/`--n-iter` (naming differs slightly per script - check `--help`); the values above are sane defaults for a real run, not a smoke test.
- `config.py` expects `NER_MODEL_DIR`/`RE_MODEL_DIR`/`SUSPICIOUS_MODEL_PATH`/`SUSPICIOUS_EMBEDDINGS_PATH` at the exact output paths shown above by default. If you train to a different location, point the pipeline at it via the matching `MMFOOD_*` environment variable instead of moving files around - see the table below.
- `nli` and `graph_alignment` are not trained: NLI uses a public pretrained HuggingFace checkpoint (`NLI_MODEL_NAME`, default `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`), and graph alignment is rule-based.

---

## 3. Getting data to verify

The pipeline verifies **pairs** of `(source_text, llm_extracted_json)` - it does not do the LLM extraction itself. You need to bring both halves:

- `text`: the raw recipe article/page text the extraction was drawn from.
- `extracted`: the LLM's JSON output for that text, matching the schema in `verification/schema.py` (see `examples/sample_input.json` for a minimal valid example and `data/eval_sample.json` for a real-shaped one; `data/vocab/cleaned_ingredients_n_v2.txt` is a two-entry sample of the vocabulary stage 2 checks against).

Input file shape for `run_pipeline.py`:

```json
{"text": "...", "extracted": { ... }}
```

or a JSON array of such objects for batch mode. If you're producing these yourself, run your extraction LLM over your source text first, save its raw JSON output as `extracted`, and pair it with the original `text` in one of the two shapes above.

---

## 4. Running verification

```bash
python run_pipeline.py --input examples/sample_input.json --output report.json
```

`--output` is optional - omit it to print the report to stdout. Batch mode (a JSON array as input) writes back a JSON array of reports, one per input document, in the same order.

---

## 5. Checking verification results

`report.json` (or stdout) has one object per input document:

```json
{
  "results": {
    "auto_correction": [...],
    "stages": {
      "stage_1": {"errors": [...], "warnings": [...]},
      "stage_2": {"new_ingredients": [...]},
      "stage_3": {"outliers": [...]},
      "stage_4": {"outliers": {...}},
      "stage_5": {"report": [...]}
    }
  },
  "extracted": { ... }
}
```

| Field | Meaning |
|---|---|
| `stage_1.errors` / `.warnings` | JSON Schema violations against `RECIPE_SCHEMA` (e.g. `"X is not of type 'string'"`). Errors are schema-breaking; warnings are soft (missing non-essential fields). |
| `stage_2.new_ingredients` | Ingredient names in `extracted` not found in the vocabulary file - candidates for being hallucinated, misspelled, or genuinely novel. |
| `stage_3.outliers` | Ingredients whose `estimated_weight_in_grams` per unit falls outside the IQR bounds seen historically for that ingredient+unit pair; each entry includes the `expected_range` and computed `mean` for context. |
| `stage_4.outliers` | Ingredients the Set Transformer scores as not plausibly belonging in this recipe (keyed by ingredient name → suspicion probability), only for scores above the IQR upper bound. |
| `stage_5.report` | Per-ingredient (and per-metadata-field) fidelity checks against the source text. Each `error_type: "ingredient_fidelity"` entry reports `existence`/`quantity`/`form` as `ENTAILMENT` (source supports the claim), `CONTRADICTION` (source disagrees), or `NEUTRAL` (source doesn't confirm either way); `source_quantity` shows what was actually matched in the text, or a fallback reason (`"NLI Fallback"`, `"Implicit (Verified Staple)"`, etc.) when no direct textual match was found. `error_type: "metadata_field_not_in_source"` / `"metadata_field_not_in_extracted"` / `"metadata_value_mismatch"` cover prep/cook/total time and serving size. |

A document with an empty `stage_1.errors`, no `stage_2`/`stage_3`/`stage_4` entries, and all `stage_5` fields `ENTAILMENT` is one the pipeline found no issues with. Everything else is a candidate for human review - which is what the correction tool below is for.

---

## 6. Human-in-the-loop review

`CorrectionTool/correction_tool/` is a Flask + React app for a human to review documents (optionally pre-annotated with `detected_errors` from the pipeline above) and record corrections, building a labeled dataset you can use to improve the models.

### Running it

The supported path is Docker (it bundles the frontend build, backend, and an nginx reverse proxy in one container):

```bash
cd CorrectionTool/correction_tool
./run_image.sh
```

This builds and runs a container exposing the app at `http://localhost:7301`. It needs a MongoDB instance reachable from inside the container - by default it connects to `mongodb://host.docker.internal:27017`, i.e. a MongoDB running on your host machine (`docker run -d -p 27017:27017 mongo` is the quickest way to get one). Override the connection string with the `MONGO_URI` environment variable if yours lives elsewhere.

For local iteration without Docker, run the backend directly (`python server/main.py`, listens on `PORT`, default `5000`) and the frontend dev server (`cd frontend && npm install && npm run dev`) - but note the frontend calls the backend via the relative path `/api/...`, which only the Docker image's nginx config proxies to port 5000. Running the two processes standalone is useful for testing the backend API directly but the frontend won't be able to reach it unless you configure your own proxy.

### Default login

A single placeholder account is included in `server/local_collections/users.json`:

- **username:** `johndoe`
- **password:** `changeme`

Change this password (or add real accounts via the signup form) before using the tool for anything beyond a local test - it's plaintext-compared, single-user-friendly, and not meant for public deployment (see Known Limitations).

### Workflow

1. **Log in** with the account above (or sign up a new one from the same screen).
2. **Pick or create a project.** A project is just a named Mongo collection of documents to review.
3. **Upload data to the project**, either via the in-app uploader or the `/upload_data` API directly: a CSV of source rows (must include a `text` column and whatever key you're matching on) and a JSON array of extracted objects (each with that same matching key, and optionally a `detected_errors` array - this is where you'd feed in a `stage_5.report`-shaped list, or your own flat `{field, error_type, source_value}` list, to pre-flag fields for the reviewer).
4. **Correction page** (`/correction`): shows one uncorrected document at a time - the raw extracted JSON on one side, the source text on the other, with search/highlighting between them. Click any field to select it, describe the error type and corrected value, and add it to that document's correction log. Two ways to finish a document: **Save & Next** (persists your correction log + edited JSON) or **Mark Correct & Next** (no corrections needed). Either way, the next fetch (`/get_correction_data`) excludes documents that already have a `corrected_json`, so you're always shown fresh ones.
5. **History page** (`/history`): review documents you've already corrected.
6. **Admin overview** (`/admin_overview`): cross-user view - all users, all projects, and all corrected documents, with the ability to review/approve individual corrections.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini` \| `openai` \| `local` - backend for the optional auto-corrector |
| `LLM_MODEL` | Model name for the selected provider |
| `LLM_BASE_URL` | Base URL for a local OpenAI-compatible endpoint (e.g. a llama.cpp server) |
| `GEMINI_API_KEY` | API key when `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | API key when `LLM_PROVIDER=openai` |
| `OPENAI_BASE_URL` | Optional override for the OpenAI base URL |
| `AUTO_CORRECT_ENABLED` | `true`/`false` - gates the LLM auto-corrector (default: off, for a deterministic pipeline run) |
| `MONGO_URI` | Mongo connection string for the correction tool |
| `CORRECTION_TOOL_PORT` | Port for the correction tool's Flask backend |
| `MMFOOD_MODELS_ROOT` | Override for the root directory holding all three trained models |
| `MMFOOD_VOCAB_PATH` | Override for the stage-2 ingredient vocabulary file |
| `MMFOOD_CORRECTIONS_DB` | Override for the auto-corrector's historical-fix lookup file |
| `MMFOOD_NER_MODEL_DIR` | Override for the trained `ingredient_ner` model directory |
| `MMFOOD_RE_MODEL_DIR` | Override for the trained `relation_extraction` model directory |
| `MMFOOD_SUSPICIOUS_MODEL_PATH` | Override for the trained `suspicious_ingredient` checkpoint file |
| `MMFOOD_SUSPICIOUS_EMBEDDINGS_PATH` | Override for the `suspicious_ingredient` ingredient-embeddings file |
| `NLI_MODEL_NAME` | HF model id for the NLI checkpoint used in stage 5 |

## Known limitations

- Stage 5 (source fidelity) accuracy has not been independently benchmarked against a ground-truth set.
- The correction tool stores user passwords in plaintext and compares them directly; it is intended for internal annotation use only, not public deployment. Change the default `johndoe` / `changeme` credentials before real use.
- The correction tool's `detected_errors` UI highlighting matches on a `field` key; if you feed it a pipeline `stage_5.report` directly, note it uses `path` instead, so pipeline output won't auto-highlight in the UI without a small key-rename step on your end.

## Repository layout

- `run_pipeline.py` - pipeline entry point.
- `config.py` - environment-variable-driven paths and settings.
- `verification/` - the 5-stage pipeline, recipe schema, and the LLM provider abstraction.
- `Models/` - one subdirectory per model, each with `inference.py` and (where applicable) `prepare_dataset.py` / `train.py`.
- `data/vocab/` - a two-entry sample of the ingredient vocabulary stages 2 and 3 use.
- `data/eval_sample.json` - a one-row sample input for `run_pipeline.py`.
- `examples/` - a minimal valid `run_pipeline.py` input fixture.
- `CorrectionTool/correction_tool/` - the human-in-the-loop correction web app.
- `CASE_WALKTHROUGH.md` - a worked example of the detection + resolution pipeline on one recipe.
