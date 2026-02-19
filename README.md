# Agentic RAG Implementation with LangGraph 🦜🔍

Implementation of Adaptive RAG for visual impaired users learning LangGraph🦜🕸️.

This code was guided by Gemini-CLI.

This repository contains a refactored version of the original [LangChain's Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain),

See Original YouTube video:[Advance RAG control flow with Mistral and LangChain](https://www.youtube.com/watch?v=sgnrL7yo1TE)

of [Sophia Young](https://x.com/sophiamyang) from Mistral & [Lance Martin](https://x.com/RLanceMartin) from LangChain


## Features

An accessible RAG CLI tool to summarize or query files from (default) input directory
and store results in (default) output directory.

```bash
% poetry run python ragagent2.py --help     
usage: ragagent2.py [-h] [--query [QUERY]] [--summarize] [--input INPUT] [--output OUTPUT] [--rich]

options:
  -h, --help       show this help message and exit
  --query [QUERY]  The question to ask the RAG agent.
  --summarize      Summarize all documents in the input directory.
  --input INPUT    Directory for source documents (default 'input').
  --output OUTPUT  Directory to save results (default 'output').
  --rich           Enable rich text output for non-screen reader users.


% poetry run python ragagent2.py --summarize
% poetry run python ragagent2.py --query 'Was is a multi-agent environment?' 

Both cases are using default 'input' and 'output' directory
```

## Environment Variables

To run this project, need to add the following environment variables to your .env file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # For web search capabilities
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional, for tracing
LANGCHAIN_TRACING_V2=true                      # Optional
LANGCHAIN_PROJECT=your-agentic-rag             # Optional
```

> **Note**: For query tracing via a LangSmith, to inspect
queries and their results between LLM calls, the setting 
`LANGCHAIN_TRACING_V2=true` must be defined.
Also, a valid LangSmith API key must be defined in `LANGCHAIN_API_KEY`. 
Without a valid API key, the application will throw an error.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/kds215/RagAgent2.git
cd RagAgent2
git checkout project/RagAgent2
```

Install dependencies:

```bash
pip install -r requirements.txt
# or if using Poetry:
poetry install
```

Run:

```bash
% poetry run python ragagent2.py --help 
```

## Acknowledgements

(1) this RagAgent2 code was GGBK published: “Gemini-Guided-By-Klaus”.

(2) Original LangChain repository: [LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)
By [Sophia Young](https://x.com/sophiamyang) from Mistral & [Lance Martin](https://x.com/RLanceMartin) from LangChain
