---
title: "lamalab picks papers of the month - jan. 2026"
date: 2026-01-31T10:00:00+00:00
draft: false
author: "gordan prastalo"
description: "A short comment on the best research environment award we received, and the application that we submitted"
tags: ["paper_of_month", "chemistry", "machine_learning", "research"]
intro: ""
---


## we will name it after some of the research papers

for the new paper of the month everyone should submit one or more papers. the paper should ideally be published in the last month or so, but it’s fine if it’s a bit older. each of the papers should be followed by a short description, one or two paragraphs. the description should contain some of the following: why did you pick this paper? what do you like/what do you not like? what is interesting in there for you? make the descriptions personal and give your opinions rather than just describing the paper. i will collect all of the responses into one blog post and send it to you before it gets released for the final check. 

i will prepare a proper template once we have the first post out, since i am still unsure how the post would exactly look like.

Deadline: 29th of Jan

@ Sadra

@ Kevin

@ Adrian

https://arxiv.org/pdf/2601.02671 (Ahmed, Cooper et al.) Extracting books from production language models

This paper demonstrates that it is possible to extract copyrighted material from Large Language Models. Interestingly, not all production-ready models required jail-breaking to achieve this (Gemini-2.5-Pro and Grok 3 required, Claude-3.7-Sonnet and GPT-4.1 models did not). Often, LLM providers argue, for legal reasons, that their models do not memorize the data, and this paper clearly shows that at least they memorize part of it. For me one limitation of the paper hides in its very study cases, which are books that are widely known, and thus spread on the Internet. Additional work would be required to see if the same is possible for science books.

https://arxiv.org/pdf/2512.13668 (Liu et al.) A Scientific Reasoning Model for Organic Synthesis Procedure Generation

This paper demonstrates the application of Reinforcement Learning from Verifiable Rewards (RLVR) to a real chemical problem: the prediction of procedural information for organic synthesis. GRPO worked in this case, because the authors designed a set of actions which can occur in organic synthesis, thus verifiable and usable in the RVLR paradigm. More ablations of the reward function would be valuable to learn from, and additionally the ablation of the supervised finetuning (is it valuable compared to starting from the base model?).

@ Meiling

@ Mara

https://doi.org/10.26434/chemrxiv-2025-6c250 (Qiu et al.) A Robust and Versatile Generative Model for Inverse Design of Polymers

This work tackles the fragility of polymer generative models, specifically their tendency to hallucinate invalid structures or lose track of functional group requirements. By replacing standard text representations with Group SELFIES, the authors integrate chemical validity and polymer class definitions directly into the tokenization, achieving 100% structural validity. They pair this with PRECISE, a physics-informed reinforcement learning strategy that intelligently balances novelty and synthesizability, even when trained on small datasets (~1,000 polyimides). While the validation against DFT calculations for dielectric constants is promising, the current limitation to homopolymers leaves a gap for complex co-polymer designs, where the most interesting material properties often emerge.

@ Nawaf

@ Martiño

@ Viktor

@ Rostislav
