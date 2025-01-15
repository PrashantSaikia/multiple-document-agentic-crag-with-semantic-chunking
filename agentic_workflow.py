from typing import Sequence, List, Dict, Literal, Any, Annotated, Tuple
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import os, re, asyncio
from dataclasses import dataclass
from rich import print as rprint
from rich.panel import Panel
# Create DocumentStore with PyMuPDFLoader or PyPDFLoader and HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

@dataclass
class TariffResult:
    detailed_breakdown: str
    total_amount: float
    individual_charges: Dict[str, float]

class MainGraphState(TypedDict):
    messages: Sequence[BaseMessage]
    sub_questions: List[str]
    sub_responses: Dict[str, str]

class SubgraphState(TypedDict):
    messages: Sequence[BaseMessage]
    sub_responses: Dict[str, str]
    rewrite_count: int
    original_query: str

def log_node_output(node_name: str, output: Any = None, show_output: bool = False):
    rprint(f"\n[bold blue]--- {node_name.upper()} ---[/bold blue]")
    if show_output and output:
        if isinstance(output, dict):
            for key, value in output.items():
                rprint(f"[green]{key}[/green]: {value}")
        else:
            rprint(Panel(str(output), title="Output"))

class DocumentStore:
    """Handles document retrieval using an InMemoryVectorStore from a PDF."""
    def __init__(self, pdf_path: str):
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # self.embedding_function = AzureOpenAIEmbeddings(model="text-embedding-3-large")
        self.retriever = self._build_retriever(pdf_path)

    def _build_retriever(self, pdf_path: str):
        loader = PyMuPDFLoader(file_path=pdf_path)
        pages = asyncio.run(self._load_pages(loader))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False,
            add_start_index=True,
        )

        split_docs = text_splitter.split_documents(pages)
        vector_store = InMemoryVectorStore.from_documents(split_docs, self.embedding_function)

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4, "score_threshold": 0.5}
        )
        return retriever

    async def _load_pages(self, loader: PyMuPDFLoader):
        # PyMuPDFLoader.load() returns a list of documents; we can directly call it
        # If async is needed, just run it in a blocking manner
        docs = loader.load()
        return docs

    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        results = self.retriever.get_relevant_documents(query)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]

class MaritimeTariffCalculator:
    def __init__(self, pdf_path: str):
        self.doc_store = DocumentStore(pdf_path)
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2024-08-01-preview",
            temperature=0,
            streaming=True
        )
        self.max_rewrites = 3
        self.latest_result = None

    def grade_documents(self, state: SubgraphState) -> Literal["generate", "rewrite"]:
        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        llm_with_tool = self.llm.with_structured_output(Grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool
        scored_result = chain.invoke({"question": question, "context": docs})
        log_node_output("grade_documents", {"score": scored_result.binary_score})

        return "generate" if scored_result.binary_score == "yes" else "rewrite"

    def rewrite_query(self, state: SubgraphState) -> SubgraphState:
        messages = state["messages"]
        question = messages[0].content
        rewrite_count = state.get("rewrite_count", 0)

        if rewrite_count >= self.max_rewrites:
            return {**state, "rewrite_count": rewrite_count}

        msg = [
            HumanMessage(
                content=f"""Look at the input and try to reason about the underlying semantic intent / meaning.
                Here is the initial question:
                \n ------- \n
                {question}
                \n ------- \n
                Formulate an improved question focusing specifically on the maritime tariff calculation aspects:"""
            )
        ]

        response = self.llm.invoke(msg)
        log_node_output("rewrite_query", {"original": question, "rewritten": response.content})

        return {
            "messages": [HumanMessage(content=response.content)],
            "sub_responses": state.get("sub_responses", {}),
            "rewrite_count": rewrite_count + 1,
            "original_query": state.get("original_query", question)
        }

    def generate_response(self, state: SubgraphState) -> SubgraphState:
        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        prompt = PromptTemplate(
            template="""You are a maritime port tariff expert. Using the provided context, calculate the requested charge accurately.
            For vessel specifications, use ONLY the values provided in the vessel details.
            Show ALL calculations step by step and cite the exact rates/formulas you're using from the tariff documentation.
            Be extremely precise with mathematical calculations.

            Context: {context}
            Question: {question}

            Requirements:
            1. Quote the specific rates and formulas from the tariff documentation
            2. Show each mathematical step clearly
            3. Show all unit conversions if needed
            4. Present the final amount in South African Rand (R)

            Response Format:
            1. Applicable Rates: (quote from tariff)
            2. Calculation Steps:
               - Step 1: ...
               - Step 2: ...
            3. Final Amount: R X,XXX.XX
            """,
            input_variables=["context", "question"]
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})

        # Show only the output of the generate node in a box
        log_node_output("generate_response", response, show_output=True)

        return {
            "messages": messages + [AIMessage(content=response)],
            "sub_responses": {**state.get("sub_responses", {}), question: response},
            "rewrite_count": state.get("rewrite_count", 0)
        }

    def create_subgraph(self) -> Any:
        workflow = StateGraph(SubgraphState)

        def retrieve_node(state: SubgraphState):
            messages = state["messages"]
            last_message = messages[-1].content

            docs = self.doc_store.get_relevant_documents(last_message)
            docs_content = "No relevant documents found." if not docs else "\n\n".join(doc["page_content"] for doc in docs)

            log_node_output("retrieve", {
                "query": last_message,
                "num_docs": len(docs) if docs else 0
            })

            return {
                **state,
                "messages": messages + [AIMessage(content=docs_content)]
            }

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("rewrite", self.rewrite_query)
        workflow.add_node("generate", self.generate_response)

        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            },
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "retrieve")

        return workflow.compile()

    def decompose_query(self, state: MainGraphState) -> MainGraphState:
        messages = state["messages"]
        original_query = messages[-1].content

        # Extract vessel info by splitting at the first colon
        if ":" in original_query:
            _, vessel_info = original_query.split(":", 1)
        else:
            vessel_info = original_query
        vessel_info = vessel_info.strip()

        # Only create sub-queries for individual charges, not for the total again
        charge_types = [
            "light dues",
            "port dues",
            "vessel traffic service charges",
            "pilotage dues",
            "running of vessel lines charges",
            "berth dues"
        ]

        sub_queries = []
        for charge_type in charge_types:
            sub_query = f"Calculate the {charge_type} incurred by the following vessel at the port of Durban:\n{vessel_info}"
            sub_queries.append(sub_query)

        log_node_output("decompose_query", {"number_of_sub_queries": len(sub_queries)})

        return {
            "messages": state["messages"],
            "sub_questions": sub_queries,
            "sub_responses": {},
        }

    def _process_sub_queries(self, subgraph):
        def process(state: MainGraphState) -> MainGraphState:
            sub_questions = state["sub_questions"]
            all_responses = {}

            for sub_query in sub_questions:
                subgraph_state = SubgraphState(
                    messages=[HumanMessage(content=sub_query)],
                    sub_responses={},
                    rewrite_count=0,
                    original_query=sub_query
                )

                try:
                    result = subgraph.invoke(subgraph_state)
                    if result["messages"] and len(result["messages"]) > 0:
                        last_message = result["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            all_responses[sub_query] = last_message.content
                except Exception as e:
                    print(f"Error processing sub-query '{sub_query}': {str(e)}")
                    continue

            return {
                "messages": state["messages"],
                "sub_questions": sub_questions,
                "sub_responses": all_responses,
            }

        return process

    def create_main_graph(self) -> Any:
        main_graph = StateGraph(MainGraphState)

        main_graph.add_node("decompose_query", self.decompose_query)
        subgraph = self.create_subgraph()
        main_graph.add_node("process_sub_queries", self._process_sub_queries(subgraph))
        main_graph.add_node("aggregate_responses", self.aggregate_responses)

        main_graph.add_edge(START, "decompose_query")
        main_graph.add_edge("decompose_query", "process_sub_queries")
        main_graph.add_edge("process_sub_queries", "aggregate_responses")
        main_graph.add_edge("aggregate_responses", END)

        return main_graph.compile()

    def aggregate_responses(self, state: MainGraphState) -> MainGraphState:
        sub_responses = state["sub_responses"]

        summary = "Maritime Charges Calculation Breakdown:\n\n"
        total = 0.0
        individual_charges = {}

        # Identify known charge types to avoid confusion
        known_charge_types = {
            "light dues": "light dues",
            "port dues": "port dues",
            "vessel traffic service charges": "vessel traffic service charges",
            "pilotage dues": "pilotage dues",
            "running of vessel lines charges": "running of vessel lines charges",
            "berth dues": "berth dues"
        }

        for question, calculation in sub_responses.items():
            lower_question = question.lower()
            matched_charge_type = None
            for kct in known_charge_types:
                if kct in lower_question:
                    matched_charge_type = kct
                    break

            if not matched_charge_type:
                # If we cannot identify the charge type, skip this entry
                continue

            summary += f"=== {matched_charge_type.upper()} ===\n{calculation}\n\n"

            try:
                amounts = re.findall(r'R\s*([\d,]+\.?\d*)', calculation)
                if amounts:
                    final_amount = float(amounts[-1].replace(',', ''))
                    total += final_amount
                    individual_charges[matched_charge_type] = final_amount
            except Exception as e:
                print(f"Warning: Could not extract amount from calculation: {str(e)}")

        summary += f"=== TOTAL CHARGES: R {total:,.2f} ===\n"

        self.latest_result = TariffResult(
            detailed_breakdown=summary,
            total_amount=total,
            individual_charges=individual_charges
        )

        log_node_output("aggregate_responses", {"total": f"R {total:,.2f}"}, show_output=True)

        return {
            "messages": state["messages"] + [AIMessage(content=summary)],
            "sub_questions": state["sub_questions"],
            "sub_responses": sub_responses,
        }

    def calculate_tariff(self, vessel_info: str) -> TariffResult:
        graph = self.create_main_graph()
        # from IPython.display import Image, display
        # display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

        initial_state = {
            "messages": [HumanMessage(content=vessel_info)],
            "sub_questions": [],
            "sub_responses": {},
        }

        result = graph.invoke(initial_state)
        return self.latest_result
    

if __name__ == "__main__":
    pdf_path = r"./TariffDocs/Port Tariff.pdf"
    calculator = MaritimeTariffCalculator(pdf_path)

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
    51,255 / 31,192
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
    Days alongside: 3 days
    Arrival time: 15 Nov 2024 10:12
    Departure time: 22 Nov 2024 13:00

    Activity/operations: exporting iron ore
    Number of operations: 2
    """

    user_query = f"Calculate the total charges incurred by the following vessel at the port of Durban: {vessel_info}"

    print("\n=== STARTING TARIFF CALCULATION ===\n")
    result = calculator.calculate_tariff(user_query)
    print("\n=== FINAL CALCULATION RESULT ===\n")
    print(result.detailed_breakdown)