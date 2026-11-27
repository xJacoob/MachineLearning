# AGH-RAG — Experiment Results

## Setup

- **Documents:** `regulamin.pdf`, `harmonogram.pdf`, `isi.pdf`, `stypendia.pdf` (84 pages after skipping, see `SKIP_PAGES` in `ingest.py`)
- **Embedding model:** `bge-m3` (Ollama), cosine distance in ChromaDB
- **Chat model:** `qwen2.5:7b` (Ollama), `temperature=0`
- **Test set:** `eval/questions.json`, 6 questions (2 easy, 1 medium, 2 hard, 1 out-of-scope)
- **Raw outputs:** `eval/results/<run_name>.json`

## Metrics

- **Retrieval hit:** at least one of the `k_top` retrieved chunks comes from the expected document and page. Not computed for out-of-scope questions.
- **Answer grade (manual):**
  - `correct`: complete and accurate, in Polish, with a valid citation
  - `partial`: accurate but incomplete
  - `wrong`: incorrect, wrong language, hallucinated citation, or a refusal when the answer was in the context

One change per experiment. Previous results are never overwritten.

---

## Run 1 — Baseline (`baseline_c1000_k3`)

**Config:** chunk_size 1000, overlap 150, k_top 3, prompt v1

| # | Type | Question (short) | Hit | Grade | Notes |
|---|------|------------------|-----|-------|-------|
| 1 | easy | Days off (statutory + additional) | ✅ | correct | Correctly separated 28 May as an additional day off |
| 2 | easy | Number of exam attempts | ✅ | wrong | Answer in **German**; cited page 17, which was not in the context (got 19, 33, 19), so a **hallucinated citation** |
| 3 | hard | Scholarship achievements ≥ 30 pts | ✅ | partial | Listed 2 of 8 items. Retrieved pages 20, 18, 19; page 17 missing |
| 4 | medium | Reasons for removal from student list | ✅ | correct | Also included mandatory reasons (§ ust. 1), which are valid too; expected answer was too narrow |
| 5 | hard | Assessment in Digital Image Processing | ✅ | wrong | Correct page 34 was in the context, but the model answered "not found" |
| 6 | out_of_scope | Dormitory cost | — | correct | Correct refusal |

**Retrieval:** 5/5 · **Answers:** 3 correct / 1 partial / 2 wrong

### Observations

- **High hit rate ≠ good answers.** Retrieval found the right page every time; all failures happened in generation.
- **Hit rate is lenient for multi-page answers.** Q3 counted as a hit with only one of the two expected pages retrieved.
- **Language drift:** the English system prompt plus a 7B model occasionally answers in another language, even at temperature 0.
- **Hallucinated citations:** the model can cite a page that was never in its context.
- **A distance threshold won't work for "I don't know":** the out-of-scope question (0.44) scored a *better* distance than an answerable one (Q5: 0.46–0.48).
- **Duplicate pages:** two chunks from the same page (p. 19) took 2 of 3 context slots in Q2.

---
## Run 2 — `c1000_k5`

**Config:** chunk_size 1000, overlap 150, **k_top 5**, prompt v1
**Hypothesis:** more context should fix Q3 (missing page 17); risk: more noise for the 7B model.

| # | Type | Question (short) | Hit | Grade | Notes |
|---|------|------------------|-----|-------|-------|
| 1 | easy | Days off (statutory + additional) | ✅ | correct | Same as baseline |
| 2 | easy | Number of exam attempts | ✅ | partial | Now in Polish and content correct, but still cites page 17 (context: 19, 33, 20), so hallucinated citation |
| 3 | hard | Scholarship achievements ≥ 30 pts | ✅ | wrong | Page 17 now retrieved, but the model misread the table: listed "3 chapters [1.5]" (actually item 1.4 = 24 pts) |
| 4 | medium | Reasons for removal from student list | ✅ | correct | All 8 reasons, cites pages 25–26 |
| 5 | hard | Assessment in Digital Image Processing | ✅ | wrong | Still "not found"; more context didn't help |
| 6 | out_of_scope | Dormitory cost | — | correct | Still refuses, extra chunks didn't cause hallucination |

**Retrieval:** 5/5 · **Answers:** 3 correct / 1 partial / 2 wrong

### Observations

- **k_top 5 improved retrieval** (page 17 now in context for Q3) without breaking the out-of-scope refusal. Keeping k_top 5 as the new default.
- **Tables are the main failure source.** pypdf flattens tables, so row numbers, descriptions and point values end up on separate lines; the model mixed up rows 1.4 and 1.5 in Q3. Q5 (`isi.pdf` pp. 28–38, also tables) likely fails for the same reason.
- **Language fix is not reliable:** Q2 switched from German to Polish only because the context changed, not because of a real fix.
- **Hallucinated citations persist:** page 17 again, and the number doesn't even appear in the retrieved text.
- **Neighbouring chunks crowd the context:** page 26 appeared 3 times in Q4, page 33 twice in Q2.

## Run 3 — `c1000_k5_prompt_v2`

**Config:** chunk_size 1000, overlap 150, k_top 5, **prompt v2**
**Changes in prompt v2:** cite sources only from fragment headers, stronger "answer in Polish" rule, list all matching items (no 1–4 sentence limit for lists), "several fragments" instead of "three chunks".
**Also changed:** fixed typo in Q5 ("cyforwych" → "cyfrowych").

| # | Type | Question (short) | Hit | Grade | Notes |
|---|------|------------------|-----|-------|-------|
| 1 | easy | Days off (statutory + additional) | ✅ | partial | Content correct, but cited "[1] regulamin.pdf, s. 19", copied verbatim from the example in the prompt (fragment [1] was harmonogram.pdf, s. 2) |
| 2 | easy | Number of exam attempts | ✅ | correct | Polish, cites pages 19 and 33 (both in context); hallucinated page 17 is gone |
| 3 | hard | Scholarship achievements ≥ 30 pts | ✅ | partial | 2 of 3 listed items correct; still includes "3 chapters" (24 pts, table misread); grants and awards from pp. 18–19 missing |
| 4 | medium | Reasons for removal from student list | ✅ | partial | Over-inclusive: added a sub-case of "not starting studies" as a separate reason, missed disciplinary expulsion; cites non-existent fragments like [7], [17] |
| 5 | hard | Assessment in Digital Image Processing | ✅ | wrong | Still "not found", even with the typo fixed |
| 6 | out_of_scope | Dormitory cost | — | correct | Still refuses |

**Retrieval:** 5/5 · **Answers:** 2 correct / 3 partial / 1 wrong

### Observations

- **The model copies examples from the prompt.** A realistic example ("regulamin.pdf, s. 19") was reused as the citation for an unrelated answer. Use placeholders (`<file>, s. <page>`) in prompt examples.
- **Header-only citation rule partially works:** the hallucinated page 17 in Q2 disappeared.
- **"List all items" trades precision for completeness:** longer lists in Q3, but over-inclusion in Q4.
- **Prompt changes shift errors rather than remove them:** with a 7B model, each fix tends to break something else.
- **Q5 is not a prompt problem:** the chunk with the answer (middle of page 34, no table header) is still not retrieved.

## Run 4 — `c2000_k5_prompt_v2`

**Config:** **chunk_size 2000, overlap 300**, k_top 5, prompt v2, `num_ctx 8192`
**Hypothesis:** larger chunks keep table headers together with their rows (Q5) and keep table rows intact (Q3).

| # | Type | Question (short) | Hit | Grade | Notes |
|---|------|------------------|-----|-------|-------|
| 1 | easy | Days off (statutory + additional) | ✅ | correct | Correct citation this time (harmonogram.pdf, s. 2) |
| 2 | easy | Number of exam attempts | ❌ | wrong | Page 19 no longer retrieved; answered "twice", based on the general field exam on p. 33 |
| 3 | hard | Scholarship achievements ≥ 30 pts | ✅ | partial | Table now read correctly (40 / 24 / 30 pts), but the 24-pt item is still listed as ≥ 30; items from pp. 18–19 missing |
| 4 | medium | Reasons for removal from student list | ✅ | partial | 14 items: all 8 correct ones + 6 over-inclusive (sub-cases, unrelated provisions); no citation |
| 5 | hard | Assessment in Digital Image Processing | ✅ | partial | **Content correct for the first time** (all 5 forms); cited p. 37 instead of p. 34 |
| 6 | out_of_scope | Dormitory cost | — | correct | Still refuses |

**Retrieval:** 4/5 · **Answers:** 2 correct / 3 partial / 1 wrong

## Run 5 — `c1500_k5_prompt_v2`

**Config:** **chunk_size 1500, overlap 225**, k_top 5, prompt v2, `num_ctx 8192`
**Hypothesis:** a middle ground: page 34 header and the target row fit in one chunk, while chunks stay small enough to avoid the Q2 retrieval miss from Run 4.

| # | Type | Question (short) | Hit | Grade | Notes |
|---|------|------------------|-----|-------|-------|
| 1 | easy | Days off (statutory + additional) | ✅ | correct | Correct content and citation |
| 2 | easy | Number of exam attempts | ✅ | partial | Page 19 retrieved again, content correct; cited only fragment [2] (p. 33) instead of [1] (p. 19) |
| 3 | hard | Scholarship achievements ≥ 30 pts | ✅ | wrong | 2 correct items, 2 wrong ("article 40–70" = 20 pts; a sports medal is not a scientific achievement); key items missing |
| 4 | medium | Reasons for removal from student list | ✅ | partial | Listed only the sub-cases of "lack of progress", not the main reasons |
| 5 | hard | Assessment in Digital Image Processing | ✅ | partial | 4 of 5 forms ("Egzamin" missing); cited p. 38 instead of p. 34 |
| 6 | out_of_scope | Dormitory cost | — | correct | Still refuses |

**Retrieval:** 5/5 · **Answers:** 2 correct / 3 partial / 1 wrong

### Observations

- **Middle ground works for retrieval:** page 19 is back for Q2 (no dilution), and the Q5 table row is retrieved together with its header.
- **Remaining errors are mostly citations and list completeness,** except Q3, where the model mixed in items from the wrong table (sports) and the wrong row (40–70 pts).

---

## Summary

| Run | Config | Retrieval | Correct | Partial | Wrong |
|-----|--------|-----------|---------|---------|-------|
| 1 | c1000, k3, prompt v1 | 5/5 | 3 | 1 | 2 |
| 2 | c1000, k5, prompt v1 | 5/5 | 3 | 1 | 2 |
| 3 | c1000, k5, prompt v2 | 5/5 | 2 | 3 | 1 |
| 4 | c2000, k5, prompt v2 | 4/5 | 2 | 3 | 1 |
| 5 | c1500, k5, prompt v2 | 5/5 | 2 | 3 | 1 |

**Final choice: chunk_size 1500, overlap 225, k_top 5, prompt v2.**
It is the only configuration where both easy questions get the correct content *and* the hardest table question (Q5) is almost fully answered. Chunk 1000 fails Q5 completely; chunk 2000 dilutes embeddings and gives a confidently wrong answer to Q2. The remaining errors are mostly citation and list-completeness issues rather than invented facts.

**Caveat:** with only 6 questions, small config changes reshuffle individual answers a lot (Q4 went correct → partial → over-inclusive → incomplete). Differences between runs are close to noise level; the evaluation set will be extended to 10 questions.

### Key lessons

- **Retrieval hit rate alone is misleading:** 5/5 retrieval still produced wrong answers; a page-level hit also counts when the right page but the wrong chunk was retrieved.
- **PDF tables are the hardest case:** flattened rows lose their headers and column alignment.
- **Chunk size is a trade-off:** small chunks give precise retrieval of narrow facts, large chunks give tables their context; 1500 balanced both here.
- **Prompt changes shift errors rather than remove them;** realistic examples in a prompt get copied verbatim.
- **A 7B model has limits:** numerical comparisons (≥ 30 pts) and complete lists are unreliable.
- **Small eval sets are noisy:** one run per config with 6 questions can't separate close configurations.