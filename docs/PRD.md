# Canonical: project design doc (one page)

**Owner / PM:** solo (Wednesday section). **Demo of Demos:** Wed Dec 2. **Final presentation:** Wed Dec 9. **Demo Night:** Mon Dec 14, 2026. **Feature freeze:** Dec 1.

## Problem
Students learn better when a concept is tied to something they care about. Many already ask chatbots to
"explain recursion through Naruto." Research shows those analogies are fluent but often structurally wrong, and
LLMs invent fictional canon. No existing tool checks either problem: not GPT-store anime tutors, Charademia,
BrainBinge, or Google's Learn Your Way.

## Users
CS1/CS2 students who watch anime, and creators who teach CS through anime (for example, The Surgeon of Code).

## Solution
An anime-to-CS lesson generator that checks its own analogies:
- It is grounded in a cited canon corpus.
- A logistic regression / random forest trained on hand labels ranks the analogies.
- A faithfulness gate verifies every claim.

## Scope (frozen)
- **Anime:** Naruto, Hunter x Hunter, Jujutsu Kaisen.
- **Topics:** stack, queue, recursion, BFS, DFS, hash table, binary search, deadlock. That gives 24 pairs.
- **Output:** a lesson with citations, a role-to-element mapping table, a "where it breaks" box, a quiz and a
  source list.

## Data
- Fandom wiki text: 82 pages, about 400k words, CC BY-SA.
- 8 hand-checked CS concept cards.
- About 200 hand-labelled analogy candidates (good / ok / bad, plus canon-error, CS-error and shallow flags).
- 51 labelled retrieval queries.

## Models
- **Taught in class:** logistic regression (Week 7) and random forest (Week 9), as the analogy-quality
  classifier.
- **Retrieval:** BM25 plus embeddings.
- **LLM (Gemini):** proposes and writes only. It never grades itself in the final system.

## Success metrics
1. **Ablation, condition D vs A:** fewer unsupported canon claims and fewer CS contradictions.
2. **Classifier:** ROC-AUC above the embedding-only and LLM self-rating baselines (grouped CV, CI reported).
3. **Retrieval:** Recall@5 and MRR for BM25 vs dense vs hybrid.
4. **Optional:** a pilot crossover quiz with 12-20 students, each learning one topic with an anime lesson and one
   with a generic lesson, followed by an unassisted post-test.

## Milestones
| Date | Milestone |
|---|---|
| Nov 18-19 | Corpus, cards, pipeline, app skeleton, system design diagram |
| Nov 19-21 | Gemini key, embeddings, hybrid retrieval eval, GitHub repo |
| Nov 22-25 | About 240 candidates; label about 200; send 60 to a second rater |
| Nov 26-28 | Train and evaluate the classifier; end-to-end lessons through the gate |
| Nov 29-Dec 1 | Cache all 24 pairs, logging, suggestions, deploy; feature freeze Dec 1 |
| Dec 2 | Demo of Demos |
| Dec 2-8 | Ablation sweep, gate spot-check, Evidence page, pilot (optional) |
| Dec 9 | Final presentation |
| Dec 9-13 | Polish, backup video, pitch practice |
| Dec 14 | Demo Night |

## Risks and what we cut first
- **Labeling runs long:** cap it at 150 labels.
- **The pilot can't run:** drop it.
- **The free tier hits its limit:** serve from cache.
- **Never cut:** the deployed app, the labelled dataset and classifier, and the ablation table.

## Open questions for the instructor
1. Is a solo project eligible for Demo Night awards?
2. Is there a judging scorecard?
3. Is consent or IRB approval needed for a classmate quiz?
