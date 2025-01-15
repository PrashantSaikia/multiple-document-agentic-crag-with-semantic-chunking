import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

class PortTariffQA:
    def __init__(
        self,
        docs_dir: str = "markdown_docs_processed",
        llm: AzureChatOpenAI = None
    ):
        self.docs_dir = docs_dir
        self.llm = llm or AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2024-08-01-preview",
            temperature=0,
            streaming=True,
        )
        
        # Document type mapping
        self.doc_types = {
            "light": "light_dues.md",
            "vts": "vts_dues.md",
            "pilotage": "pilotage_dues.md",
            "towage": "towage_dues.md",
            "line": "line_handling_dues.md",
            "port": "port_dues.md"
        }

    def _determine_doc_type(self, query: str) -> str:
        """Determine which document type to query based on the user's question."""
        query_lower = query.lower()
        
        # Map keywords to document types
        keyword_mapping = {
            "light": ["light", "lighthouse"],
            "vts": ["vts", "vessel traffic", "traffic service"],
            "pilotage": ["pilot", "pilotage"],
            "towage": ["tow", "towage", "tug", "tugs"],
            "line": ["line", "handling", "mooring"],
            "port": ["port dues", "port fee", "port charge"]
        }
        
        for doc_type, keywords in keyword_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return doc_type
                
        # Default to port dues if no specific type is mentioned
        return "port"

    def _get_document_content(self, doc_type: str) -> str:
        """Load the content of the relevant markdown document."""
        filename = self.doc_types[doc_type]
        file_path = os.path.join(self.docs_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: Document {filename} not found."
        except Exception as e:
            return f"Error loading document: {str(e)}"

    def query(self, question: str) -> str:
        """Answer a question using the relevant document content."""
        # Determine which document to use
        doc_type = self._determine_doc_type(question)
        
        # Get document content
        content = self._get_document_content(doc_type)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template("""You are a helpful port tariff assistant. Use the following port tariff document to answer the user's question.
                                                  
        If the user query asks to calculate a charge in a location and if the document provided to you doesn't mention the charges for that location specifically but instead 
        mentions the charges for all other locations, then use that value for all other locations to calculate the charges.
                                                  
        Also, line handling dues, pilotage dues and towage dues depend on how many operations are done;
        so if the user query mentions that, say, 2 operations were done (i.e., loading and unloading), then you need to multiply these three charges by 2. 
                                                  
        The following terminilogies are relevant when calculating line handling dues - Berthing services, mooring serrices and running of vessel lines.

        Ignore the VAT and potential reductions, if any. If you don't know the answer or can't find it in the document, say so.
        
        And do not use latex notation in your calcultions.
                                                  
        Document content:
        {content}
        
        Question: {question}
        
        Answer: """)
        
        # Create and run chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "content": content,
            "question": question
        })


# Example usage:
if __name__ == "__main__":
    # Initialize with custom LLM if needed
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-08-01-preview",
        temperature=0,
        streaming=True,
    )
    
    # Create QA system
    qa = PortTariffQA(llm=llm)
    
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

    Number of operations: 2
    """

    charges = ["light dues", "port dues", "line handling dues", "pilotage dues", "towage dues", "vts dues"]
    response = ''
    # User query
    for charge_type in charges:
        query = f"What would be the {charge_type} of a vessel that arrives at the port of Durban with the following vessel details: {vessel_info}?"
        response += f"# {charge_type}\n"
        response += qa.query(query)
        response += "\n---\n"
        print(response)

    with open('Response.md', 'w') as md:
        md.write(response)