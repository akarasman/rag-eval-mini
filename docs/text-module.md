# Text module (`src/rag_eval/text.py`)

Design choices for turning documents into strings and strings into tokens. Used by the retrievers.

## Overview
- `doc_to_text(doc)` builds the string that gets indexed from a `{"title", "text"}` dict.
- `tokenize(text, stem=False)` splits text into tokens for BM25.
- Named `text.py` rather than `tokenize.py` to avoid shadowing the stdlib module.

## `doc_to_text`
| Choice | Decision | Why |
|---|---|---|
| Format | `title + ". " + text` | Title is plain context; no `"Title:"` label, which would add a zero-information token to every document |
| Empty/missing title | Returns just the text, no leading `". "` | Avoids a stray separator in the string the dense model sees |
| Empty/missing text | Returns `"T. "` (trailing space kept) | Harmless for BM25 (the tokenizer drops it); trailing whitespace could be stripped later if it matters |
| Missing keys | `.get(..., "")` | `load_corpus` already guarantees both keys, so this is only a safety net |

**Open:** one `doc_to_text` feeds both retrievers. Context that helps the dense model (a title, punctuation) may be noise for BM25, and the comparison between them is only fair if each gets its best input. If the experiments show a difference, give the retrievers separate text preparation, or an argument to select it.

## `tokenize`
Lowercase, then `findall` with one compiled regex: a token is a run of letters/digits (Unicode-aware, `_` excluded), optionally joined to more runs by a hyphen or by a decimal point between digits.

| Input | Tokens | Rule |
|---|---|---|
| `Cancer` | `cancer` | lowercased |
| `(p53).` | `p53` | punctuation dropped, identifiers kept whole |
| `IL-6`, `well-known` | `il-6`, `well-known` | hyphenated compounds stay together |
| `3.5`, `2.0.1` | `3.5`, `2.0.1` | a dot joins only when digits are on both sides |
| `10.1038/nature` | `10.1038`, `nature` | `/` splits |
| `1,000` | `1`, `000` | `,` splits |
| `mg/kg` | `mg`, `kg` | `/` splits |
| `snake_case` | `snake`, `case` | `_` splits |
| `β-catenin` | `β-catenin` | Unicode letters are kept |
| `Alzheimer's` | `alzheimer`, `s` | apostrophe splits |

- **Single-character tokens are kept.** A filter was tried and removed. "Vitamin D", "type 1" and "hepatitis B" depend on them. The cost is stray `s` tokens from possessives; their idf is near zero, so the effect is small, and it could add false overlap between unrelated "...'s" phrases. To be checked in the experiments.
- **Stopwords are not removed.** idf already gives them almost no weight.
- **Stemming is optional** (`stem=False` by default, Porter via `nltk`). Aggressive stemming can merge scientific terms with different meanings (gene/genetic), so it is a flag to compare in the experiments, not a default.
- **The same call must be used for documents and queries.** A mismatch (e.g. stemming only at query time) causes no error; queries just stop matching and recall drops quietly. The BM25 retriever should own the `stem` setting and apply it to both.
- **Known imperfections:** `1,000` becomes `1` and `000`, and DOIs split at the slash. Not expected to matter while documents and queries go through the same tokenizer.

## Tests (`tests/test_text.py`)
- `doc_to_text`: exact-string checks for the main join and for each empty/missing-field case, plus a no-mutation check.
- `tokenize`: one parametrized test with one row per rule above, plus a stemming test.
- The stemmer is replaced via `monkeypatch.setattr` on the module global with a recording fake, to check that `stem=False` never calls it and `stem=True` calls it for every token. A global makes this patching necessary; passing the stemmer as an argument would not.
- An extra-keys test for `doc_to_text` was written and deleted: the loader only ever produces `title` and `text`, so it could not fail.

## Open items
- Tests for Unicode (`β-catenin`), `2.0.1`, a trailing dot after a number, and edge hyphens are still missing.
- Whitespace-only titles are not treated as empty.
- Compare stemming on/off and the title format on the real SciFact queries.
- Look at the most frequent tokens and a few random abstracts once the data is downloaded.
