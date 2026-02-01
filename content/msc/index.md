
---
title: "Master's thesis topic ideas"
---

this list is not exhaustive and only supposed to give starting points for collaborative development of specific project ideas.

# project ideas
| project | lead/supervisor | description |
|---------|-----------------|-------------|
| mechanism parsing & lift training | martiño ríos-garcía | parse mechanism explanations from books/webpages for common reactions, then train using [language interfaced finetuning (lift)](https://arxiv.org/abs/2206.06565) to potentially improve performance of models like [flower](https://www.nature.com/articles/s41586-025-09426-9) |
| organic data extraction | martiño ríos-garcía | multiple directions possible: extract all protocols from orgsyn (high-value data), or other organic chemistry extraction tasks |
| flexible polymer data extraction | mara schilling-wilhelmi | focus on flexible extraction pipelines rather than extraction quality improvement, leveraging strong capabilities of recent models (e.g., gemini) |
| chempile subset & model training | adrian mirza | create or contribute to building a curated subset of [chempile](https://chempile.lamalab.org) and train a model on it |
| iupac-smiles bootstrapping | adrian mirza | bootstrapping iupac-smiles conversion (and potentially nmr bootstrapping) using verifable rewards |
| llm-agent for data schema updates | | if we extract data, we might need to adjust the data schema on the fly or adjust synonym maps. llm-based agents should be able to implement this. |
| finetuning of open models for data extraction | | we have now many results from extraction pipelines. based on those we should be able to perform distillation to have our "own" open models to perform  data extraction | 
| implement tooling that can count electrons in inorganic compounds | sadra aghajani | for various downstream tasks - e.g., test and use of chemical heuristics - it would be great to have tooling that can reliably count electrons in inorganic compounds | 
| llm-based materials descriptions as gnn node embeddings | |  in some cases, we have only "fuzzy" material descriptions. llms can provide embeddings for those that we can use in gnns |
| parserbench | | writing file-conversion tools is one of the most promising applications for making research-data management scale. to be able to develop those systems, we need a reliable benchmark for the parser-writing performance | 
| text-to-dsl | | tune models to formalize their reasoning as domain-specific language that can more formally be verified and scaled | 