
import sys
import os
import argparse
import datetime
import re
from rich.traceback import install
install(show_locals=False) # Hides API keys but shows the code flow

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_openai")

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from graph.graph import app
from ingestion import ingest_documents, retriever as doc_retriever

console = Console()

script_name = os.path.basename(sys.argv[0])

def get_display_name(file_path):
    """
    Returns the base filename, but handles macOS .rtfd bundles by returning 
    the bundle name if the file is 'TXT.rtf' inside it.
    """
    basename = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path)
    if basename == "TXT.rtf" and parent_dir.lower().endswith(".rtfd"):
        return os.path.basename(parent_dir)
    return basename

def display_rag_result(result):
    # 1. Clear terminal & show question (optional)
    #console.clear()
    #console.print(f"Question: " + ??)

    # 2. Display the main Answer
    console.print("\n")
    console.print(Panel(
        Text(result["generation"], style="white"),
        title="[bold green]GPT-5.2 Agent Response",
        subtitle=f"[dim]Web Search Used: {result['web_search']}[/dim]",
        border_style="green",
        expand=False
    ))

    # 3. Display the Sources in a Table
    if result.get("documents"):
        table = Table(
            title="\n[bold blue]Supporting Evidence (Retrieved Documents)",
            show_header=True,
            header_style="bold cyan",
            border_style="dim"
        )
        table.add_column("Source #", justify="center", style="dim")
        table.add_column("Content Snippet", ratio=1)

        for i, doc in enumerate(result["documents"]):
            # Clean up the text for the table
            content = doc.page_content.replace("\n", " ").strip()
            snippet = (content[:150] + "...") if len(content) > 150 else content
            table.add_row(f"Doc {i+1}", snippet)

        console.print(table)

    console.print(f"\n\n[End of {script_name} output]")

def print_accessible_result(result):
    # JAWS users benefit from clear section headers
    print("\n" + "= AI RESPONSE =")
    print(result["generation"])

    print(f"\n= SOURCES USED ({len(result.get('documents', []))} documents) =")

    if result.get("documents"):
        for i, doc in enumerate(result["documents"]):
            # Use simple numbering and clear labels
            print(f"\n=Document {i+1}:")
            # Stripping newlines to keep the reader from pausing unnecessarily
            clean_content = " ".join(doc.page_content.split())
            print(clean_content[:300] + "...")
    else:
        print("No supporting documents found.")

    print(f"\n[End of {script_name} output]")

def to_camel_case(text):
    words = re.sub(r'[^a-zA-Z0-9\s]', '', text).split()
    if not words:
        return ""
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:5])

def archive_output(output_dir, query, response_text):
    now = datetime.datetime.now()
    year_dir = os.path.join(output_dir, str(now.year))
    os.makedirs(year_dir, exist_ok=True)

    # Naming Convention: mmdd.hhmm-CamelCaseOfFirstFiveWords.txt
    camel_case_query = to_camel_case(query)
    filename = f"{now.strftime('%m%d.%H%M')}-{camel_case_query}.txt"
    filepath = os.path.join(year_dir, filename)

    with open(filepath, 'w') as f:
        f.write(response_text)
    print(f"INFO: Output archived to {filepath}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An accessible RAG CLI tool.")
    parser.add_argument("--query", nargs="?", help="The question to ask the RAG agent.")
    parser.add_argument("--summarize", action="store_true", help="Summarize all documents in the input directory.")
    parser.add_argument("--input", default="./input", help="Directory for source documents.")
    parser.add_argument("--output", default="./output", help="Directory to save results.")
    parser.add_argument("--rich", action="store_true", help="Enable rich text output for non-screen reader users.")

    args = parser.parse_args()
    
    if not args.query and not args.summarize:
        print("Please provide a --query or use the --summarize flag.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    print(f"INFO: Starting ingestion from '{args.input}'...", file=sys.stderr)
    ingest_documents(args.input)
    print("INFO: Ingestion complete.", file=sys.stderr)

    if args.summarize:
        print("--- SUMMARIZING ALL DOCUMENTS ---")
        
        # Directly access the vector store to get all documents
        vectorstore = Chroma(
            collection_name="rag-chroma",
            persist_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db"),
            embedding_function=OpenAIEmbeddings(),
        )
        results = vectorstore.get()
        
        # Define a simple summarization chain
        prompt_template = ChatPromptTemplate.from_template(
            "Summarize the following document concisely:\n\n---\n\n{document_content}"
        )
        llm = ChatOpenAI(temperature=0)
        summarize_chain = {"document_content": RunnablePassthrough()} | prompt_template | llm
        
        all_summaries = []
        source_map = {}

        # 1. Group content by source
        if results.get('documents'):
            for i in range(len(results['documents'])):
                content = results['documents'][i]
                metadata = results['metadatas'][i]
                source = os.path.abspath(metadata.get('source', 'Unknown'))
                
                # We only need one chunk to summarize the document for now, 
                # or we could collect all. Preserving existing logic of "first seen" 
                # (but now "first seen" will be consistent after sort if we collect first).
                # To be safe and simple, let's store the first chunk we encounter for each source.
                if source not in source_map:
                    source_map[source] = content

        # 2. Sort the sources
        sorted_sources = sorted(source_map.keys())

        # 3. Iterate and summarize
        for i, source in enumerate(sorted_sources, 1):
            content = source_map[source]
            # Use the helper to handle .rtfd and other special cases
            display_filename = get_display_name(source)
            
            print(f"\n#{i}. Summarizing: \"{display_filename}\":")
            
            try:
                summary_result = summarize_chain.invoke(content)
                summary_text = summary_result.content
                # Append with the same format as the console output
                all_summaries.append(f"#{i}. Summarizing: \"{display_filename}\":\n{summary_text}\n")
                print(summary_text)
            except Exception as e:
                print(f"FAILED to summarize \"{display_filename}\": {e}")

        # Archive the combined summaries
        final_summary_output = "\n".join(all_summaries)
        archive_output(args.output, "summarize-all", final_summary_output)

    elif args.query:
        print(f"Hello Advanced RAG - Running {script_name}")
        inputs = {"question": args.query, "retry_count": 0}
        final_state = app.invoke(inputs)

        if args.rich:
            display_rag_result(final_state)
        else:
            print_accessible_result(final_state)

        archive_output(args.output, args.query, final_state.get("generation", "No content generated."))
