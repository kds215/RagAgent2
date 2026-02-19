from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredRTFLoader,
    UnstructuredPowerPointLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
import os
import argparse

load_dotenv()

# Set the path to your vector DB directory
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db")

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

def ingest_documents(input_dir: str):
    """
    Loads documents from the specified directory by manually iterating through
    files, determining their type, and using the appropriate loader. This
    approach is compatible with older versions of LangChain.
    """
    print(f"STARTING INGESTION FROM {input_dir}")
    
    all_docs = []
    
    # Define a mapping from file extensions to loader classes
    loader_map = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".rtf": UnstructuredRTFLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".html": BSHTMLLoader,
        ".md": UnstructuredMarkdownLoader,
        ".log": TextLoader,
        ".mhtml": UnstructuredLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
    }

    # Collect all files first to sort them
    all_files = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.startswith('.'):
                print(f"SKIPPING dot file: \"{os.path.join(dirpath, filename)}\"")
                continue
            all_files.append(os.path.join(dirpath, filename))

    all_files.sort()

    for i, file_path in enumerate(all_files, 1):
        file_ext = os.path.splitext(file_path)[-1].lower()
        display_filename = get_display_name(file_path)

        if file_ext in loader_map:
            loader_class = loader_map[file_ext]
        else:
            # Fallback to Unstructured for "others"
            print(f"#{i}. Attempting unstructured load for \"{display_filename}\":")
            loader_class = UnstructuredLoader

        print(f"#{i}. Loading: \"{display_filename}\":")
        try:
            # Special handling for JSONLoader if we want it to be more specific, 
            # but for now basic loading is fine.
            if loader_class == JSONLoader:
                # Basic JSONLoader requires jq_schema. For simplicity in a generic ingestor, 
                # we might prefer Unstructured or a specific schema.
                # Let's fallback to Unstructured for JSON if it fails or use a simple schema.
                loader = loader_class(file_path, jq_schema=".", text_content=False)
            else:
                loader = loader_class(file_path)
            
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            if "magic" in str(e).lower() or "libmagic" in str(e).lower():
                 print(f"SKIPPING \"{display_filename}\": Missing libmagic dependency.")
            else:
                 print(f"FAILED to load \"{display_filename}\": {e}")

    if not all_docs:
        print(f"No processable documents found in {input_dir}. Ingestion skipped.")
        return

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(all_docs)

    print(f"CREATING VECTOR STORE AT {PERSIST_DIR}")
    
    # Create the vector store and persist it
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIR,
    )
    
    print(f"INGESTION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("input_dir", help="The directory containing documents to ingest.")
    args = parser.parse_args()
    
    ingest_documents(args.input_dir)

# This part is for when the module is imported elsewhere, exposing the retriever
retriever = Chroma(
        collection_name="rag-chroma",
        persist_directory=PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
    ).as_retriever()
