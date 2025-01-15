from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI
import os, hashlib
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

def load_vectorstore(db_path: str) -> Chroma:
    """
    Load the Chroma vector database from disk.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    return vectorstore

def format_docs(docs: List[Document]) -> str:
    """
    Format the retrieved documents into a single string context.
    """
    formatted_docs = []
    print("\nRetrieved documents:")
    # print("-"*50)
    for i, doc in enumerate(docs, 1):
        formatted_doc = (
            f"\nDocument {i}:\n"
            f"Source: {doc.metadata.get('source', 'N/A')}\n"
            f"Section: {doc.metadata.get('section', 'N/A')}\n"
            f"Content:\n{doc.page_content}\n"
            f"{'-' * 80}"
        )
        formatted_docs.append(formatted_doc)
        print(formatted_doc)
    
    return "\n".join(formatted_docs)

def filter_documents(docs: List[Document], query: str, llm: AzureChatOpenAI) -> List[Document]:
    """
    Filter documents by removing duplicates and irrelevant content.
    
    Args:
        docs: List of retrieved documents
        query: Original user query
        llm: AzureChatOpenAI model instance
    
    Returns:
        List[Document]: Filtered list of documents
    """
    # First remove exact duplicates using content hash
    unique_docs = []
    seen_hashes = set()
    
    for doc in docs:
        # Create hash of the document content
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    print(f"Removed {len(docs) - len(unique_docs)} duplicate documents")
    
    # Now check relevance of each remaining document
    relevant_docs = []
    
    relevance_prompt = """Task: Determine if the following document section is relevant to answering the user's question.
    Rate the relevance on a scale of 0-10, where:
    - 0-3: Not relevant
    - 4-7: Somewhat relevant
    - 8-10: Highly relevant
    
    Provide your response in the following format only:
    Score: [number]
    Reason: [brief explanation]
    
    User Question: {query}
    
    Document Section:
    {document}
    """
    
    for doc in unique_docs:
        try:
            # Get relevance score from LLM
            response = llm.invoke(
                relevance_prompt.format(
                    query=query,
                    document=doc.page_content
                )
            )
            
            # Extract score from response
            try:
                score_line = response.content.split('\n')[0]
                score = int(score_line.split(':')[1].strip())
            except (IndexError, ValueError):
                print(f"Error parsing score from LLM response: {response.content}")
                continue
                
            # Keep documents with relevance score >= 4
            if score >= 4:
                relevant_docs.append(doc)
                print(f"\nKept document with score {score}:")
                # print(f"Section: {doc.metadata.get('section', 'N/A')}")
                # print(f"Reason: {response.content.split('Reason:')[1].strip()}")
            else:
                print(f"\nFiltered out document with score {score}:")
                # print(f"Section: {doc.metadata.get('section', 'N/A')}")
                # print(f"Reason: {response.content.split('Reason:')[1].strip()}")
                
        except Exception as e:
            print(f"Error processing document: {e}")
            continue
    
    print(f"\nKept {len(relevant_docs)} relevant documents out of {len(unique_docs)} unique documents")
    return relevant_docs

# Modified create_retrieval_chain function to use the document filter
def create_retrieval_chain(vectorstore: Chroma, llm: AzureChatOpenAI):
    """
    Create a retrieval chain with document filtering.
    """
    # Create the retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Increased k since we'll be filtering
    )
    
    # Create the prompt template
    template = """You are an expert on port tariffs, fees, and calculations. Use the provided context to deduce the correct 
    formulae and calculate the requested charges accurately.  

    Context:  
    {context}  
    
    Question: {question}  
    
    Instructions:  
    1. Carefully read through all sections of the provided context to identify relevant formulae, rates, and conditions for 
    calculating the fees or charges mentioned in the query.  
    2. Deduce the correct formula or method for the calculation based on the provided context.  
    3. Use the values or inputs provided in the question to perform the calculation.  
    4. Clearly explain the formula or method used, including specific rates, conditions, and steps where applicable.  
    5. Provide the final result of the calculation in a clear and professional manner.  
    6. If you cannot find the formula, rates, or any specific information required for the calculation, say 
    "I don't find this information in the provided context."  
    
    Answer: 
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def retrieve_and_filter(query: str):
        # Get initial documents
        docs = retriever.get_relevant_documents(query)
        # Filter documents
        filtered_docs = filter_documents(docs, query, llm)
        return filtered_docs
    
    def combine_documents(query: str):
        docs = retrieve_and_filter(query)
        return {"context": format_docs(docs), "question": query}
    
    # Construct the chain
    chain = (
        RunnablePassthrough()
        | combine_documents
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def setup_llm():
    """
    Set up the Azure OpenAI GPT-4 model.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not azure_endpoint or not api_key:
        raise ValueError(
            "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables"
        )
    
    try:
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            openai_api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-02-15-preview",
            temperature=0,
        )
        return llm
    except Exception as e:
        raise Exception(f"Error initializing Azure OpenAI: {str(e)}")

def query_database(query: str, chain) -> str:
    """
    Query the database using the retrieval chain.
    """
    # print(f"\nProcessing query: {query}")
    print(f"\nProcessing query...")
    try:
        response = chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error during query processing: {e}")
        raise

if __name__ == "__main__":
    # Load the vector database
    db_path = "./markdown_docs_db"
    vectorstore = load_vectorstore(db_path)
    print("Loaded vector database")
    
    try:
        # Setup LLM
        llm = setup_llm()
        print("Set up LLM successfully")
        
        # Create the retrieval chain
        chain = create_retrieval_chain(vectorstore, llm)
        print("Created retrieval chain")
        
        vessel_info = """
        Vessel Details

        General
        Vessel Name
        SUDESTADA
        Built
        2010
        Flag
        MLT - Malta
        Classification Society
        Registro Italiano Navale
        Call Sign
        9HA5631
        Main Details
        Lloyds / IMO No.
        9426087
        Type
        Bulk Carrier
        DWT
        93,274
        GT / NT
        51,300 / 31,192
        LOA (m)
        229.2
        Beam (m)
        38
        Moulded Depth (m)
        20.7
        LBP
        222
        Drafts SW S / W / T (m)
        14.9 / 0 / 0
        Suez GT / NT
        - / 49,069
        Communication
        E-mail

        Commercial E-mail

        DRY
        Number of Holds
        7

        Cargo quantity: 40000 MT

        Number of days alongside: 3.396 days
        """

        # User query
        query = f"What would be the port dues of a vessel that arrives at the port of Durban with the following vessel details: {vessel_info}?"

        try:
            response = query_database(query, chain)
            print("\nAnswer:")
            print(response)
            print("-" * 80)
        except Exception as e:
            print(f"Error processing query: {e}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
