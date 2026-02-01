---
title: "Team"
---

if you want to join the team, please contact us [via this form](https://forms.fillout.com/t/eoGA7AhnAKus) - we are always happy to speak to potential new team members at all levels.
each new hire is an opportunity for us to bring in a different perspective, and we are always eager to diversify our team further.

we value doing impactful work. [but we do not believe in rushing things, but instead aim for a sustainable pace.](https://kjablonka.com/blog/posts/take_it_easy/)

for students pursuing a masters degree, we compiled a list of potential topics. however, we are happy to also explore projects beyond this list. 

# project ideas
| project | lead/supervisor | description |
|---------|-----------------|-------------|
| mechanism parsing & lift training | martiño ríos-garcía | parse mechanism explanations from books/webpages for common reactions, then train using [language interfaced finetuning (lift)](https://arxiv.org/abs/2206.06565) to potentially improve performance of models like [flower](https://www.nature.com/articles/s41586-025-09426-9) |
| organic data extraction | martiño ríos-garcía | multiple directions possible: extract all protocols from orgsyn (high-value data), or other organic chemistry extraction tasks |
| flexible polymer data extraction | mara schilling-wilhelmi | focus on flexible extraction pipelines rather than extraction quality improvement, leveraging strong capabilities of recent models (e.g., gemini) |
| chempile subset & model training | adrian mirza | create or contribute to building a curated subset of [chempile](chempile.lamalab.org) and train a model on it |
| iupac-smiles bootstrapping | adrian mirza | bootstrapping iupac-smiles conversion (and potentially nmr bootstrapping) using verifable rewards |
| llm-agent for data schema updates | | if we extract data, we might need to adjust the data schema on the fly or adjust synonym maps. llm-based agents should be able to implement this. |
| finetuning of open models for data extraction | | we have now many results from extraction pipelines. based on those we should be able to perform distillation to have our "own" open models to perform  data extraction | 
| implement tooling that can count electrons in inorganic compounds | sadra aghajani | for various downstream tasks - e.g., test and use of chemical heuristics - it would be great to have tooling that can reliably count electrons in inorganic compounds | 
| llm-based materials descriptions as gnn node embeddings | |  in some cases, we have only "fuzzy" material descriptions. llms can provide embeddings for those that we can use in gnns |
| parserbench | | writing file-conversion tools is one of the most promising applications for making research-data management scale. to be able to develop those systems, we need a reliable benchmark for the parser-writing performance | 
| text-to-dsl | | tune models to formalize their reasoning as domain-specific language that can more formally be verified and scaled | 