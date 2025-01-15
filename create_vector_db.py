import re
from typing import List, Dict, Tuple
from pathlib import Path
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def read_markdown_files(folder_path: str) -> Dict[str, str]:
    """
    Read all markdown files from the specified folder.
    Returns a dictionary with filenames as keys and content as values.
    """
    markdown_files = {}
    folder = Path(folder_path)
    
    # Find all markdown files in the folder
    for file_path in folder.glob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                markdown_files[file_path.name] = content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return markdown_files

def extract_sections(text: str, filename: str) -> List[Document]:
    """
    Extract sections from the text while maintaining context.
    Returns list of Langchain Document objects.
    """
    # Split on main section headers
    sections = re.split(r'(?=\n## SECTION \d+\n)', text)
    
    # Remove empty sections and strip whitespace
    sections = [section.strip() for section in sections if section.strip()]
    
    # Further split sections if they're too large
    final_chunks = []
    for section in sections:
        # Get the main section header if it exists
        main_header = ""
        header_match = re.match(r'(## SECTION \d+.*?)(?=\n)', section)
        if header_match:
            main_header = header_match.group(1)
        
        # Split on subsection headers but maintain context
        subsections = re.split(r'(?=\n## \d+\.\d+)', section)
        
        for i, subsection in enumerate(subsections):
            if i == 0 and not subsection.startswith('##'):
                # This is the first subsection and doesn't start with a header
                chunk_text = subsection.strip()
            else:
                # Add the main section header for context if it exists
                if main_header:
                    chunk_text = f"{main_header}\n{subsection.strip()}"
                else:
                    chunk_text = subsection.strip()
            
            # Create metadata for the chunk
            metadata = {
                "source": filename,
                "section": main_header if main_header else "General"
            }
            
            # Create Langchain Document object
            doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            
            final_chunks.append(doc)
    
    return final_chunks

def create_and_save_db(documents: List[Document], db_path: str) -> Chroma:
    """
    Create a Chroma vector database from the documents and save it to disk.
    """
    # Initialize the embedding function
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},  # or 'cuda:0' for specific GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create and persist the database
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    # Persist the database
    vectorstore.persist()
    
    return vectorstore

def load_and_query_db(db_path: str, query: str, n_results: int = 3) -> List[Document]:
    """
    Load the database from disk and perform a query.
    """
    # Initialize the embedding function
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},  # or 'cuda:0' for specific GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load the existing database
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    # Perform the similarity search
    results = vectorstore.similarity_search(
        query=query,
        k=n_results
    )
    
    return results

# Example usage
if __name__ == "__main__":
    # Read all markdown files from the MarkdownDocs folder
    docs = read_markdown_files("MarkdownDocs")
    print(f"Found {len(docs)} markdown files")
    
    # Extract chunks from all documents and convert to Langchain Documents
    all_chunks = []
    for filename, content in docs.items():
        chunks = extract_sections(content, filename)
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} total chunks")
    
    # Create and save the database
    db_path = "./markdown_docs_db"
    vectorstore = create_and_save_db(all_chunks, db_path)
    print(f"Created and saved database to {db_path}")
    
    # Example query
    query = "How are port dues calculated?"
    results = load_and_query_db(db_path, query)
    
    # Print results
    print("\nQuery Results:")
    for doc in results:
        print("\nRelevant section:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Section: {doc.metadata['section']}")
        print("-" * 40)
        print(doc.page_content)
        print("-" * 80)
