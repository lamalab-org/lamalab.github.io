---
title: "lamalab picks papers of the month - jan. 2026"
date: 2026-01-31T10:00:00+00:00
draft: false
author: "gordan prastalo"
description: "a collection of research papers picked by lamalab members as interesting or important for january 2026."
tags: ["paper_of_month", "chemistry", "machine_learning", "research"]
intro: ""
---


## we will name it after some of the research papers

for the new paper of the month everyone should submit one or more papers. the paper should ideally be published in the last month or so, but it’s fine if it’s a bit older. each of the papers should be followed by a short description, one or two paragraphs. the description should contain some of the following: why did you pick this paper? what do you like/what do you not like? what is interesting in there for you? make the descriptions personal and give your opinions rather than just describing the paper. i will collect all of the responses into one blog post and send it to you before it gets released for the final check. 

i will prepare a proper template once we have the first post out, since i am still unsure how the post would exactly look like.

Deadline: 29th of Jan

@ Sadra

@ Kevin


My pick this month is a [paper from DeepSeek](conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models) that highlights how simple n‑gram mechanisms, combined with careful hardware–software co‑design, can improve LLM behaviour and efficiency.

Summary
- n‑grams are one of the simplest language models: a lookup of how likely a token is given the preceding n tokens. Prior work (e.g. [TinyStories](https://arxiv.org/abs/2305.07759)) has shown structural similarities between scaled n‑gram models and large transformers; we observed related effects in our [MatText](https://arxiv.org/abs/2406.17295) study.
- DeepSeek take this further by injecting n‑gram knowledge into an LLM via a new layer they call the Engram. Instead of forcing model parameters to memorize facts, the Engram performs a hash‑based lookup to retrieve embeddings derived from the input token sequence; those embeddings are then mixed and gated with context from the transformer.
- Because the lookup is hashing‑based, the system can prefetch n‑gram contributions from cheaper memory tiers (for example, CPU memory) and schedule retrievals to overlap with transformer computation. Based on [language statistics](https://en.wikipedia.org/wiki/Zipf%27s_law) they can build a caching hierarchy to make this efficient.

Key takeaways
- Model design: Engram offloads factual memorization to an external, lookup‑style mechanism while letting transformer parameters focus more on reasoning and higher‑level computation.
- Engineering: hashing + prefetching + cache hierarchy enables using large n‑gram stores without prohibitive on‑device memory costs.
- Empirical effect: improved performance on some reasoning tasks, attributed to relieved parameter capacity (parameters are less occupied with rote retrieval).
- Broader implication: this is another example of how open models and system-level co‑design are diverging across ecosystems; see this writeup on the developing China–Western gap in open models for context: [open‑model ecosystem divide](https://substack.com/redirect/3f51978c-1574-4129-aaf8-33916eac3a24?j=eyJ1IjoiMzBrbWZ0In0.oHpO_VmDNCXJgfbvu8nA5EKFQpbikjInwzUhD8fXD94).

Why I find it interesting
- It revives a very simple idea (n‑grams) but combines it with practical systems engineering to gain useful trade‑offs in cost, latency and capability. 



@ Adrian

https://arxiv.org/pdf/2601.02671 (Ahmed, Cooper et al.) Extracting books from production language models

This paper demonstrates that it is possible to extract copyrighted material from Large Language Models. Interestingly, not all production-ready models required jail-breaking to achieve this (Gemini-2.5-Pro and Grok 3 required, Claude-3.7-Sonnet and GPT-4.1 models did not). Often, LLM providers argue, for legal reasons, that their models do not memorize the data, and this paper clearly shows that at least they memorize part of it. For me one limitation of the paper hides in its very study cases, which are books that are widely known, and thus spread on the Internet. Additional work would be required to see if the same is possible for science books.

https://arxiv.org/pdf/2512.13668 (Liu et al.) A Scientific Reasoning Model for Organic Synthesis Procedure Generation

This paper demonstrates the application of Reinforcement Learning from Verifiable Rewards (RLVR) to a real chemical problem: the prediction of procedural information for organic synthesis. GRPO worked in this case, because the authors designed a set of actions which can occur in organic synthesis, thus verifiable and usable in the RVLR paradigm. More ablations of the reward function would be valuable to learn from, and additionally the ablation of the supervised finetuning (is it valuable compared to starting from the base model?).

@ Meiling

@ Mara

@ Sreekanth

https://doi.org/10.26434/chemrxiv-2025-6c250 (Qiu et al.) A Robust and Versatile Generative Model for Inverse Design of Polymers

This work tackles the fragility of polymer generative models, specifically their tendency to hallucinate invalid structures or lose track of functional group requirements. By replacing standard text representations with Group SELFIES, the authors integrate chemical validity and polymer class definitions directly into the tokenization, achieving 100% structural validity. They pair this with PRECISE, a physics-informed reinforcement learning strategy that intelligently balances novelty and synthesizability, even when trained on small datasets (~1,000 polyimides). While the validation against DFT calculations for dielectric constants is promising, the current limitation to homopolymers leaves a gap for complex co-polymer designs, where the most interesting material properties often emerge.

@ Nawaf

@ Martiño

my pick for this month is [terminal-bench](https://arxiv.org/pdf/2601.11868). this paper provides a systematic evaluation of the already-public dataset across diverse swe tasks, ranging from dataset manipulation and analysis to replicating cobol legacy code.

summary:

- they created 89 very diverse tasks that can be solved using a terminal. these tasks were selected from the 210 proposed ones, based on their difficulty and uniqueness. task creation involved multiple human reviews to ensure quality and robustness.
- they evaluated 16 different frontier models using different scaffolds and clis (swe-agent, claude code, codex, etc).
- they carried out an automatic evaluation (verified by human review) to extract the most common failure cases, finding that models primarily still fail in execution over coherence and verification. [previous works already did this](https://openai.com/index/paperbench/), but with greater human involvement, which does not scale well.

key takeaways:

- llm evaluation can aid in the development and verification of the tasks.
- llms can also be used for fine-grained evaluation through very detailed workflows.
- they acknowledge that not releasing a private dataset might lead to fast overfitting.

@ Viktor

@ Rostislav
