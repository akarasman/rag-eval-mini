# Generation (`generate.py`, `llm/`, `ask_config.py`, `ask.py`)

Answers a question from retrieved passages with an LLM, with citations, or says that the passages do not answer it. Retrieval is unchanged: the retriever returns doc IDs, and this part turns the best ones into a prompt.

    question -> retriever.search(question, k)          [(doc_id, score)]
             -> passages [(doc_id, doc_to_text(doc))]  (optional min_score filter)
             -> answer_question(question, passages, llm)
                  build prompt (passages labelled [1]..[n]) -> llm.generate() -> raw text
                  parse JSON (retry on failure) -> resolve citation labels to doc IDs -> status
             -> GenerationResult(status, answer, citations, ...) -> printed with sources

## LLM backends (`llm/`)
| Choice | Decision | Why |
|---|---|---|
| Interface | `LLM` protocol: `generate(prompt, system=None) -> str` | A backend returns raw text and knows nothing about passages, JSON or citations, so the generation logic is the same for every backend and is tested with a scripted fake |
| Settings | Temperature, token limit and timeout are constructor parameters, from the config's `params` | They are experiment settings, not per-call choices |
| Registry | `@register_llm("name")`, `build_llm(name, params)`, same pattern as stores and retrievers | The config picks the backend by name; constructor `TypeError` becomes a `ValueError` listing the problem |
| Errors | Bad parameters raise `ValueError` when the backend is built. Transport or API failures raise `LLMError` at call time | A typo fails before any request; a connection problem is one clear message |
| Shared HTTP code | `llm/_http.py`: one `post_json` turns a timeout, connection error, non-2xx status or non-JSON body into an `LLMError` | The two server backends differ only in their request and where the text sits in the reply. Headers are never put in a message |

| Backend | Name | Runs | Notes |
|---|---|---|---|
| `OllamaLLM` | `ollama` | A local Ollama server over HTTP (`/api/chat`) | `model` is required. `json_mode` (default on) asks the server to constrain output to JSON. Nothing heavy loads in this process |
| `TransformersLLM` | `transformers` | In process, on the CPU by default | `model_name` is required and has no default, because the choice decides speed and quality. The model loads on the first `generate()`, so building one from a config is cheap. Temperature 0 means greedy decoding. Only the newly generated text is returned |
| `OpenAICompatibleLLM` | `openai_compatible` | Any hosted API that speaks the OpenAI chat-completions protocol | The provider is chosen by `base_url`. The config names an environment variable (`api_key_env`), never the key. The key is read when the backend is built, so a missing key stops the run at the start. It is kept out of `repr()` and out of error messages |

## Generation (`generate.py`)
| Choice | Decision | Why |
|---|---|---|
| Output format | The model must reply with one JSON object: `{"answerable": bool, "answer": str, "citations": [int, ...]}` | A reply that can be parsed and checked beats free text that has to be guessed at |
| Citations | The model cites passage NUMBERS (`[1]`..`[n]`), not document IDs. Code translates them to doc IDs | Short labels are hard to get wrong, and a number that matches no passage is detectable. SciFact IDs are long and arbitrary |
| Prompt | A system prompt with the rules and two examples (answered, abstained); a user prompt with the passages, each between separator lines, then the question | The system prompt says the passages are data, not instructions. Delimiting them is the only defence against prompt injection here |
| Parsing | `parse_response` accepts clean JSON, JSON in a Markdown fence, and JSON surrounded by prose (the first balanced `{...}`, which respects strings and escapes). A null answer or null citations read as empty; numeric strings in `citations` are coerced; extra keys are ignored. Anything else raises `ValueError` | Models often wrap JSON in prose or fences; being strict about the shape but tolerant about the wrapping |
| Retries | If a reply cannot be parsed, the call is repeated up to `max_attempts` times in total, with a short correction appended to the original prompt (not to the previous retry) | One more try often fixes a formatting slip, and the prompt does not grow with each failure. An `LLMError` is never retried here |
| Invalid citations | `on_invalid_citation`: `"report"` lists labels that match no passage in `invalid_citations` and logs one warning; `"ignore"` drops them silently | There is deliberately no option that raises |

### Statuses
Decided in this order:

| Status | When | LLM called? |
|---|---|---|
| `no_passages` | Nothing was retrieved (BM25 can return nothing, or `min_score` removed everything) | No |
| `invalid_output` | No parseable reply after `max_attempts` calls | Yes |
| `abstained` | The model says the passages do not answer the question. Its citations are ignored | Yes |
| `ungrounded` | The model claims an answer but there is no valid citation, or the answer is empty. The answer text is kept so you can see the claim | Yes |
| `answered` | An answer with at least one valid citation | Yes |

### Abstention in three layers
1. **Before the LLM:** no passages means no call. The optional `min_score` filter can also empty the list. Dense retrieval always returns `k` results, so for it only `min_score` can trigger this.
2. **The model decides:** the `answerable` field, with an explicit way out in the system prompt.
3. **Checks in code:** `answerable` without a valid citation is downgraded to `ungrounded`; unparseable output after the retries is `invalid_output`.

## Ask config and command line (`ask_config.py`, `ask.py`)
| Choice | Decision | Why |
|---|---|---|
| Separate config | `AskConfig` (dataset, retriever, llm, generation), loaded with `load_ask_config`; shares `DatasetConfig`, `RetrieverConfig` and the strict base with the experiment config | An experiment evaluates against qrels; asking only needs a corpus and an LLM |
| Retriever source | Inline `retriever`, or `retriever_from: <experiment yaml>` (exactly one). A relative path is resolved against the working directory; the file is read when the config loads | Reuse the retriever you evaluated without copying it. A missing or invalid file fails at load time |
| LLM not built at load | Loading validates the LLM name only | A config can be validated without an API key or a running server |
| `min_score` | Optional; keeps passages with a score at or above it | Only meaningful for a plain BM25 or dense retriever. Hybrid RRF scores depend only on ranks, and scores are not comparable across retrievers |
| Fail early | The question is checked first, then the config, then the LLM is built, then the corpus is loaded and indexed | A typo, a missing API key or an empty question costs nothing |
| Errors | A bad config or an `LLMError` prints `error: <message>` to stderr and exits with status 1; `-v` shows the traceback instead. Argument mistakes exit with status 2 | A connection problem should not look like a bug |
| Output | Text: status, answer or the reason there is none, then a numbered list of cited sources (doc ID and title), and a note about invalid citations. `--json` prints the whole `GenerationResult`, including `raw_output` | The text is for reading; the JSON is for checking what the model actually said |

Example configs for each backend are in `configs/ask/`. Run: `python -m rag_eval.ask --config configs/ask/ollama.yaml "your question"` (`--k` overrides `generation.k`).

## Tests
All offline: no network and no model by default.
- `tests/test_generate.py` (scripted fake LLM from `conftest.py`): the prompt (labels, truncation), parsing (clean, fenced, with prose, braces inside strings, coercions, the rejects), citation resolution, every status, retries and the correction, report versus ignore, errors propagating.
- `tests/test_llm_registry.py`, `tests/test_llm_backends.py`: the registry, and both HTTP backends with `requests.post` faked (request URL, body and headers; connection, timeout, non-2xx and malformed replies; key handling). The transformers backend is tested with fake classes (lazy loading, new text only, greedy versus sampling, load errors); one real-model test is marked `slow`.
- `tests/test_ask_config.py`: valid and invalid configs, unknown names and keys, bounds, `retriever_from`, and that every shipped example and every experiment config loads.
- `tests/test_ask.py`: `main([...])` on a tiny corpus with BM25 and a fake LLM: each status, `--json`, `--k`, `min_score`, error exits, and `retrieve_passages` and `format_result` on their own.

## Known limitations
- **Nothing checks that a cited passage supports the answer.** A citation shows that the model pointed at a passage that was offered to it, not that the passage says what the answer says. That needs an NLI model or an LLM judge; out of scope.
- **Abstention quality depends on the model.** Small local models may answer anyway or ignore the JSON format; the retries and the `ungrounded` / `invalid_output` statuses catch the worst cases, not all.
- **No token accounting.** `k` and `max_passage_chars` are in passages and characters; nothing checks that the prompt fits the model's context window.
- **Prompt injection is only partly handled**: passages are delimited and labelled as data. A passage that contains the separator line can still confuse a model.
- **Not run against a real server here.** The tests fake HTTP and the model. Running `python -m rag_eval.ask` against a local Ollama server, and the slow transformers test, are checks for you to do by hand.

## Open items
- An abstention check: run a set of answerable questions and a set of off-topic ones through the pipeline and count the statuses of each (nothing is recorded yet).
- Streaming output, multi-turn chat, faithfulness checking and retry logic for transient HTTP errors are out of scope for now.
