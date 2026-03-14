# Lost Circulation LLM Framework

A knowledge-augmented large language model (LLM) framework for intelligent diagnosis and decision support of lost circulation problems in drilling engineering.

The framework integrates retrieval-augmented generation (RAG), mechanistic knowledge graphs, and agent-based reasoning to enable structured reasoning and explainable decision support.

## Overview

Lost circulation is one of the most common and costly problems in drilling operations. Traditional approaches rely on empirical rules or standalone machine learning models, which often lack interpretability and domain knowledge integration.

This framework proposes a mechanism–data collaborative AI approach that combines:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Mechanistic Knowledge Graphs
- Agent-based reasoning

to support intelligent lost circulation diagnosis and engineering decision-making.

## Project Structure
## Project Structure

```
lost-circulation-llm-framework
│
├── data_example
│
├── docs
│   ├── example for Agent Q&A
│   └── example for RAG Q&A
│
├── experiments
│   ├── run_ablation_eval
│   └── test_kg_build
│
├── framework
│   ├── hybrid_agent
│   ├── rag_chain
│   ├── kg_mechanism
│   └── kg_data
│
├── MechanismRules_300.xlsx
├── questions_ablation.xlsx
├── README.md
└── LICENSE
```


## Features

- Knowledge-augmented reasoning using mechanistic rules
- Retrieval-Augmented Generation for engineering knowledge retrieval
- Knowledge graph construction for drilling knowledge representation
- Agent-based reasoning framework for structured diagnosis
- Explainable lost circulation diagnostic reports

## Applications

- Lost circulation risk assessment  
- Loss type diagnosis  
- Mechanism interpretation  
- Intelligent drilling decision support  

## Citation

If you use this framework in your research, please cite:
```bibtex
@software{mu2026lostcirculation,
  title={Lost Circulation LLM Framework},
  author={Huayan Mu and Guancheng Jiang},
  year={2026},
  url={https://github.com/mhy333/lost-circulation-llm-framework}
}
```


## License

MIT License

