# docs/taxonomy.md — Phase 2 Failure Taxonomy Reference

This document catalogs the 7 failure modes identified by the RLAIF Critic, their formal definitions, and negative examples from real student generations.

---

## 1. Failure Categories & Definitions

| Category | Definition | Example Bad Output |
|---|---|---|
| **repetition** | Model enters degenerated loops repeating words, phrases, n-grams, or sentences without informational progress. | `"The ocean was blue and calm and blue and calm and calm and blue and calm..."` |
| **off_topic** | Model diverges from the prompt's premise, injecting unrelated trivia, tangential musings, or unsolicited opinions. | *Prompt: "Summarize the history of the steam engine."*<br>`"Steam engines are cool, but my favorite vehicle is the Tesla Model S..."` |
| **instruction_ignored** | Model actively ignores constraints (e.g., asked to correct grammar but instead answers the statement as a conversation). | *Prompt: "Correct the grammar: 'She go to store.'"*<br>`"Yes, she went there to buy groceries yesterday morning."` |
| **grammar_error** | Model generates syntactically broken sentences, subject-verb disagreements, incorrect tenses, or malformed clauses. | `"The cars was driving fastly down the streets without no headlights on."` |
| **punctuation_error** | Model omits vital punctuation marks (run-on sentences), uses erratic commas, or fails capitalization rules. | `"when the bell rang everyone left the room they ran outside into the rain"` |
| **hallucination** | Model invents nonexistent entities, produces contradictory factual claims, or hallucinates citations. | `"Albert Einstein won his Nobel Prize in 1984 for inventing the personal computer."` |
| **format_violation** | Model fails to respect structural instructions (e.g. JSON format, bullet list, exact word count, one sentence limit). | *Prompt: "Summarize in exactly one sentence."*<br>`"Here is the summary. First, the dog barked. Second, it ran."` |

---

## 2. Severity Hierarchy

1. **CRITICAL:** Total task collapse, incoherent degeneration, severe safety hazard, or completely inverted instruction.
2. **HIGH:** Definite instruction or factual failure that renders the response unusable.
3. **MEDIUM:** Detectable stylistic defect or minor grammatical inaccuracy that weakens quality.
4. **LOW:** Cosmetic blemish, stylistic preference difference, or negligible punctuation quirk.

Only **CRITICAL** and **HIGH** failures with `priority <= 5` pass the Severity Quality Gate into synthetic data generation.
