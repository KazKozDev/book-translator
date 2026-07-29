# Prompts

Every word the models are sent, one file per role. Nothing here is generated:
what you read in a file is what the model receives, once its `$placeholders`
are filled in.

Edit a file and the next run uses it — there is nothing to rebuild. The
[golden tests](../tests/test_prompts.py) will fail until you update the
matching file under `tests/fixtures/prompts/`, which is deliberate: a prompt is
the program here, and a changed prompt is a changed translation.

## The files

| File | Role | Called from |
| --- | --- | --- |
| `stage0_prepare/rendering_from_candidates.md` | Stage 0 — name the proper nouns found in the text, one rendering each | `BookTranslator._rendering_prompt` |
| `stage0_prepare/rendering_from_excerpt.md` | Stage 0 — the same question when extraction found no candidates (a script with no capitalisation to go on) | `BookTranslator._rendering_prompt` |
| `stage0_prepare/rendering_rules.md` | The answer format and rules appended to both of the above | `BookTranslator._rendering_prompt` |
| `stage0_prepare/cluster_adjudication.md` | Stage 0 — rule on what the embedding model merged or could not decide | `BookTranslator.adjudicate_entity_clusters` |
| `stage1_translate/default.md` | Stage 1 — the draft translation, for general instruct models | `BookTranslator._stage1_prompt_default` |
| `stage1_translate/translategemma.md` | Stage 1 — the draft translation in the shape TranslateGemma's model card documents | `BookTranslator._stage1_prompt_translategemma` |
| `stage2_refine/estimate.md` | Stage 2a — report the errors in a draft, as spans, without rewriting it | `BookTranslator._estimate_prompt` |
| `stage2_refine/verify.md` | Stage 2c — did the patch improve accuracy? Asked twice, versions swapped | `BookTranslator.stage2_verify` |
| `quality/pairwise_editor.md` | Stage 3 — draft vs final, on accuracy and readability separately | `QualityTests.eval_llm_judge_stage2` |
| `quality/adequacy_fluency.md` | Stage 3 — score one translation 1–5 on adequacy and fluency | `QualityTests._adequacy_fluency_prompt` |
| `shared/terminology.md` | The verified-glossary block spliced into Stage 1 and Stage 2 | `TerminologyManager.prompt_context` |
| `manual/glossary_verification.md` | Ready-to-paste glossary mode review for an external frontier model | `/glossary-verification-prompt` |

## The format

`$name` is a placeholder the caller fills in. Everything else is literal —
including the `{` and `}` of the JSON shapes the prompts ask for, which is why
these are `$name` and not `{name}`.

A line starting with `## ` opens a named section. Text before the first one is
the prompt itself; the sections after it are the optional blocks a prompt grows
when there is something to add — the previous paragraph, the glossary, the list
of violated terms. `prompts.render('stage2_refine/estimate')` gives the prompt,
`prompts.render('stage2_refine/estimate', 'terminology_violations', ...)` gives
that block.

Blank lines around a heading are markup, not content: every section is stripped
of leading and trailing blank lines. Where a block has to join its prompt with a
specific number of newlines, the code does the joining — the files hold wording,
the code holds layout.

A placeholder the caller forgot raises instead of rendering an empty string. A
prompt that quietly loses its source text is worse than one that fails.
