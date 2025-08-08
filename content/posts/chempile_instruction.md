---
title: "how we built a 400k-instruction chemistry dataset for $150?"
date: 2025-08-04T10:00:00+00:00
draft: false
author: "martiño rios garcia"
description: "a short introduction to the chempile-instruction dataset, a collection of 390k+ instruction-following examples for chemistry"
tags: ["chempile", "dataset", "instruction-following", "chemistry"]
intro: "we are pleased to announce the release of chempile-instruction, a dataset comprising over 390k instruction-following examples for chemistry. this dataset is the largest multi-turn chemistry instruction resource available, making it ideal for training llms for chemical applications. the dataset was created by rephrasing selected subsets of chempile using an llm, yielding diverse multi-turn conversations suitable for training and evaluating instruction-following models in chemistry. below we provide a dataset overview, content analysis, and discussion of potential applications."
header_image: "instruction_image_1.svg"
---

[the chempile collection](https://huggingface.co/collections/jablonkagroup/chempile-6824e88c60d3286ba9b0dae1), shown in the figure below, represents the largest chemistry-focused dataset collection available. originally containing over 75 billion tokens spanning all chemistry education levels, from high school to research literature, it serves as an exceptional resource for training chemistry domain language models. however, the original chempile did not include instruction-following tasks, which are essential for training language models.

![datasets token count](/images/posts/instruction_image_2.svg)

*figure 1. token count of the largest chemistry datasets.*

while existing chemistry instruction datasets like llasmol exist[^ref5], they typically focus narrowly on specific tasks such as property prediction, limiting both their diversity and the resulting models' general capabilities. this creates a need for more comprehensive instruction-following datasets covering diverse chemistry topics and tasks.

generating high-quality instruction data often employs templating approaches with tabular data[^ref4], yet these can lack diversity and introduce biases. a more flexible alternative uses llms to reformat existing data into instruction-following structures[^ref1], enabling scalable creation of diverse, high-quality training conversations.

we introduce the chempile-instruction dataset, created by rephrasing chempile subsets using [gpt-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/). this post demonstrates how this simple approach (~150$) produces high-quality data for training and evaluating chemistry instruction-following models. we further provide preliminary dataset analysis and application potential.

## dataset curation

as described, the curation process primarily leveraged an llm to transform existing data into instruction-following format. the chempile-instruction dataset curation involved three sequential steps:

- **selection of subsets**: we identified chempile subsets containing longer documents to yield conversations with more turns, specifically selecting:
  - **chempile-education** for its textbooks and course transcripts.
  - **chempile-reasoning** for reasoning traces and detailed explanations.
  - **chempile-paper** randomly sampling 100 million tokens from its research articles.
- **rephrasing with llm**: using *gpt-4o-mini-2024-07-18*, we rephrased selected content into instruction-following conversations through tailored prompts inspired by Pieler et al.[^ref2]. we employed four distinct stylistic approaches:
  - **wiki**: the conversations have to be written in a style similar to wikipedia articles, with clear, formal and concise explanations.

    ```markdown
    Use formal, encyclopedic English resembling Wikipedia.
    Original text:
    {text}

    Now, rephrase this text into a multi-turn conversation about chemistry.
    ```

  - **hard**: the conversations have to be written in an obscure and complex style, with many technical terms and jargon.

    ```markdown
    Use esoteric vocabulary and complex syntax suitable for academics.
    Original text:
    {text}

    Now, rephrase this text into a multi-turn conversation about chemistry.
    ```

  - **engaging**: the conversations have to be written in an engaging and interactive style, with questions and answers that encourage participation.

    ```markdown
    Present information in a clear yet engaging manner, suitable for a broad audience.
    Original text:
    {text}

    Now, rephrase this text into a multi-turn conversation about chemistry.
    ```

  - **none**: the model is not asked to follow any specific style, and the conversations are generated in a more natural way (at least for the model).

    ```markdown
    Original text:
    {text}

    Now, rephrase this text into a multi-turn conversation about chemistry.
    ```

  we processed documents in 10k batches via openai's batch api with structured outputs to enforce consistent conversation formats. each conversation contains message lists (with "user"/"assistant" roles) and dual tagging (following Mirza et al.[^ref3]) for required skills and chemical subdomains. the schema implemented was:

  ```python
  class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


  class Conversation(BaseModel):
      first_tag: Optional[
          list[
              Literal["requires-knowledge", "requires-calculation", "requires-reasoning"]
          ]
      ] = Field(
          default=None,
          description=(
              "List of possible skill types required for the conversation. "
              "- 'requires-knowledge': Requires factual or domain-specific knowledge. "
              "- 'requires-calculation': Requires mathematical or computational reasoning. "
              "- 'requires-reasoning': Requires logical or deductive reasoning."
          ),
      )
      second_tag: Optional[
          list[
              Literal[
                  "Analytical Chemistry",
                  "General Chemistry",
                  "Inorganic Chemistry",
                  "Materials Science",
                  "Organic Chemistry",
                  "Physical Chemistry",
                  "Technical Chemistry",
              ]
          ]
      ] = Field(
          default=None,
          description=(
              "List of possible domains for tagging the conversation.\n"
              "- 'Analytical Chemistry': Focuses on methods and techniques for analyzing substances, such as spectroscopy, chromatography, and mass spectrometry, to identify or quantify chemical compounds.\n"
              "- 'General Chemistry': Covers basic chemistry concepts, including the study of atoms, molecules, reactions, stoichiometry, and the periodic table, suitable for beginners or foundational learning.\n"
              "- 'Inorganic Chemistry': Deals with the properties and behaviors of inorganic compounds, including metals, metal complexes, salts, and minerals, excluding organic chemistry.\n"
              "- 'Materials Science': Involves the study of materials and their properties, including metals, polymers, ceramics, and composites, as well as their applications in various industries.\n"
              "- 'Organic Chemistry': Focuses on the study of carbon-containing compounds, including hydrocarbons and their derivatives, along with the mechanisms, reactions, and synthesis of organic molecules.\n"
              "- 'Physical Chemistry': Explores the principles and theories that explain chemical behavior, including thermodynamics, kinetics, quantum mechanics, and the interaction of matter and energy.\n"
              "- 'Technical Chemistry': Pertains to applied chemical knowledge, focusing on the industrial and practical use of chemical processes, technologies, and techniques in manufacturing and engineering."
          ),
      )
      messages: list[Message]
  ```

- **processing**: we standardized llm outputs into a structured format aligned with [litellm message conventions](https://docs.litellm.ai/docs/completion/prompt_formatting).

as demonstrated by Maini et al.[^ref1], smaller models like *mixtral-8x7b* excel at such rephrasing tasks. thus, while gpt-4o-mini is a smaller model, its ability to generate high-quality instruction-following data at scale makes it an optimal choice for this project. this tradeoff between data volume and model size allows us to create a diverse dataset without incurring prohibitive costs.

with the curation process outlined, let's dive into the key characteristics and statistics of the chempile-instruction dataset.

## dataset overview

the chempile-instruction dataset is unique in its multi-turn structure, enabling more complex and interactive dialogues than existing chemistry instruction-following datasets. it comprises 396,298 conversations averaging 7.74 turns each. the multi-turn structure of the chempile-instruction dataset allows models to engage in more complex dialogues[^ref6], improving their ability to handle sophisticated chemistry interactions and enhance agentic capabilities.

*Table 1. Comparison of general and chemistry-specific instruction-following datasets.*

| dataset             | scale (rows)  | conversation length (turns/row)| domain/specialty       |
|:-------------------:|:-------------:|:----------------------:|:------------------------------:|
| **general datasets**|               |                        |                                |
|  &nbsp;             |  &nbsp;       |  &nbsp;                |  &nbsp;                        |
| coqa                | ~127k         | ~16                    | mixed (news, literature, wiki) |
| quac                | ~100k         | ~7                     | wikipedia articles             |
| topiocqa            | ~3900         | ~13                    | wikipedia / retrieval setting  |
| scigraphqa          | ~295k         | ~2–3                   | academic graphs/text           |
|  &nbsp;             |  &nbsp;       |  &nbsp;                |  &nbsp;                        |
| **chemistry datasets** |            |                        |                                |
|  &nbsp;             |  &nbsp;       |  &nbsp;                |  &nbsp;                        |
| llasmol                          | ~3.3m   | 1              | chemistry (properties)         |
| chempile-instruction (education) | 66858   | 9.75           | chemistry (education)          |
| chempile-instruction (paper)     | 256434  | 7.52           | chemistry (papers)            |
| chempile-instruction (reasoning) | 73006   | 6.65           | chemistry (reasoning)         |
| chempile-instruction             | 396298  | 7.74           | chemistry (all, aggregate)    |

table 1 shows a comparison of general and chemistry-specific instruction-following datasets, illustrating the scale and domain diversity of the chempile-instruction dataset relative to other resources. chempile-instruction achieves a favorable scale-conversation length ratio, compensating for its smaller scale relative to other chemistry datasets. regarding data sources, chempile-instruction exhibits greater diversity through textbooks, course transcripts, research papers, and reasoning traces. conversely, general datasets predominantly focus on news, literature, and wiki articles, with notable absence of books or long documents.

however, general datasets benefit from real conversational sources, whereas chempile-instruction relies on rephrased documents. llasmol further contrasts by using templated tabular data with limited diversity.

![embeddings comparison](/images/posts/instruction_image_3.svg)

*figure 2. comparison of the embeddings of the different datasets. the embeddings were produced with the model [qwen3-embedding-0.6b](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) sampling 1000 rows for each dataset.*

figure 2 compares the embeddings of different datasets. this comparison highlights how chempile-instruction achieves greater diversity in its latent space compared to other datasets. this divergence likely stems from chempile-instruction's exclusion of smiles and other tabular data, instead prioritizing long documents and dialogues. consequently, chempile-instruction better supports general chemistry instruction tasks, while smolinstruct targets specialized applications like property/reaction prediction.

## comparison with the original chempile

![original vs rephrased embeddings](/images/posts/instruction_image_4.svg)

*figure 3. comparison of the embeddings of the original chempile subsets and the rephrased chempile-instruction subsets. the embeddings were produced with the model [qwen3-embedding-0.6b](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) sampling 1000 rows for each dataset.*

figure 3 reveals that embeddings for the reasoning and education subsets of chempile-instruction closely align with those of original chempile subsets, suggesting minimal content alteration during rephrasing. however, embeddings for the paper subset exhibit greater dispersion in chempile-instruction, indicating increased conversational diversity. this likely stems from summarization of lengthy original documents during rephrasing.

### token statistics summary

*table 2. token statistics for the chempile-instruction dataset and the original subsets. note the differences in the number of documents are because the model not following the schema.*

| dataset      | total examples | total tokens | average tokens per row |
|:---------------------------------:|:--------:|:-----------:|:--------:|
| chempile-education                | 66861    | 129904942   | 1,942.91 |
| chempile-instruction (education)  | 66858    |  51726259   |   773.67 |
|  &nbsp;                           |  &nbsp;  |  &nbsp;     |  &nbsp;  |
| chempile-paper                    | 269968   | 102859977   |   381.01 |
| chempile-instruction (paper)      | 269933   | 138549172   |   513.27 |
|  &nbsp;                           |  &nbsp;  |  &nbsp;     |  &nbsp;  |
| chempile-reasoning                | 73010    |  29566307   |   404.96 |
| chempile-instruction (reasoning)  | 73006    |  41158636   |   563.77 |

no consistent trends emerge in average tokens per row between original and rephrased subsets. a substantial token reduction occurs in the education subset, attributable to the challenge of preserving information from lengthy textbooks and transcripts. for the paper subset, the modest token increase suggests embedding dispersion results from rephrasing-induced diversity rather than length changes.

## is chempile-instruction a good dataset?

to evaluate the dataset, we trained a lora adaptor for [qwen2.5-7b-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and assessed performance on [chembench](https://chembench.lamalab.org/). direct training on chempile-instruction alone yields no base model improvement. however, fine-tuning after prior training on other chempile data boosts overall performance by over 3%, with notable 10% gains in materials science and analytical chemistry tasks. this demonstrates chempile-instruction's utility for enhancing chemistry instruction-following models when combined with complementary data, though further task-specific validation remains necessary.

## limitations

while the chempile-instruction dataset represents a significant advancement for comprehensive chemistry instruction datasets, several limitations warrant consideration:

- **llm dependency**: the dataset's quality hinges on the rephrasing llm's capabilities, with model selection critically influencing output quality. though generated via gpt-4o-mini (a relatively small llm), we assert the dataset retains sufficient quality for training and evaluating chemistry instruction-following models.
- **training results**: extensive evaluation across diverse chemistry instruction-following tasks remains pending. while initial results demonstrate utility for model training and evaluation, further validation is required for specific domains and tasks.
- **diversity of topics**: despite broad topic coverage, certain chemistry subfields remain unrepresented, potentially restricting applicability to niche domains.
- **tags**: conversation tags derive from llm interpretations and may exhibit inconsistencies or inaccuracies.

## future work

subsequent efforts will include full fine-tuning experiments with varied data volumes to evaluate dataset quality and its impact on chemistry instruction-following model performance. we further intend to expand dataset diversity through additional subsets and investigate alternative rephrasing llms to assess quality variations. we are also considering integrating sota llms to test quality improvements and better understand the model's impact.

## takeaways

generating high-quality chemistry instruction datasets remains challenging yet achievable through llm-based rephrasing of existing data. the chempile-instruction dataset marks a significant step forward in creating high-quality, scalable instruction-following data for chemistry. by leveraging cost-effective methods and llms, it provides a foundational resource for developing sophisticated ai systems capable of tackling complex chemistry problems.

![in the future we will add more subsets to the chempile-instruction dataset](/images/posts/instruction_image_end.svg)

## bibliography

[^ref1]: Maini, P.; Seto, S.; Bai, H.; Grangier, D.; Zhang, Y.; Jaitly, N. Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling. [arXiv preprint arXiv: 2401.16380](https://arxiv.org/abs/2401.16380) 2024.

[^ref2]: Pieler, M.; Bellagente, M.; Teufel, H.; Phung, D.; Cooper, N.; Tow, J.; Rocha, P.; Adithyan, R.; Alyafeai, Z.; Pinnaparaju, N.; Zhuravinskyi, M.; Riquelme, C. Rephrasing natural text data with different languages and quality levels for Large Language Model pre-training. [arXiv preprint arXiv: 2410.20796](https://arxiv.org/abs/2410.20796) 2024.

[^ref3]: Mirza, A. et al. A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists. Nature Chemistry 2025, 17, 1027–1034, DOI: 10.1038/s41557-025-01815-x, [http://dx.doi.org/10.1038/s41557-025-01815-x](https://www.nature.com/articles/s41557-025-01815-x).

[^ref4]: Gonzales, C.; Pieler, M. M.; Jablonka, K. M.; Miret, S. In AI for Accelerated Materials Design - NeurIPS 2024, 2024, [https://openreview.net/forum?id=cEkUia8neA](https://openreview.net/forum?id=cEkUia8neA).

[^ref5]: Yu, B.; Baker, F. N.; Chen, Z.; Ning, X.; Sun, H. LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset. [arXiv preprint arXiv:2402.09391](https://arxiv.org/abs/2402.09391) 2024.

[^ref6]: Laban, P.; Hayashi, H.; Zhou, Y.; Neville, J. LLMs Get Lost In Multi-
Turn Conversation. [arXiv preprint arXiv: 2505.06120](https://arxiv.org/abs/2504.13837) 2025.
