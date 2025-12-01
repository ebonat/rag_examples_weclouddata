import os
import logging
import warnings
from typing import Annotated, List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever 
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from dotenv import load_dotenv

# suppress all warnings
warnings.filterwarnings("ignore")

# Suppress ChromaDB warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.db.duckdb").setLevel(logging.ERROR)

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global variable to store retriever and last retrieved documents
retriever_instance = None
last_retrieved_docs = []

# Define the state
class State(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    question: str
    answer: str
    source_documents: List[Any]

@tool
def patient_records_search(query: str) -> str:
    """Look up patient information from medical PDF records."""
    global retriever_instance, last_retrieved_docs
    
    if retriever_instance is None:
        return "Retriever not initialized"
    
    # Get relevant documents
    docs = retriever_instance.get_relevant_documents(query)
    last_retrieved_docs = docs  # Store for later access
    
    # Format the retrieved content
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    return context

def build_hybrid_rag(pdf_path: str):
    global retriever_instance
    
    # 1. load pdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 3a. semantic retriever (ChromaDB)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3b. keyword retriever (bm25)
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 3

    # 3c. combine them into a hybrid retriever
    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.6, 0.4] # semantic prioritized, but keywords matter
    )
    
    # Store retriever globally for tool access
    retriever_instance = retriever
    
    return retriever

def create_rag_agent():
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        api_key=os.getenv('OPENAI_API_KEY'), 
        temperature=0.2
    ).bind_tools([patient_records_search])
    
    # Define agent node
    def agent(state: State):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    # Define tool node
    tool_node = ToolNode([patient_records_search])
    
    # Define routing logic
    def should_continue(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    # Build the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def run_rag_query(app, question: str):
    global last_retrieved_docs
    
    # System message for medical assistant role
    system_message = SystemMessage(content="""You are a medical assistant.
        Answer the user's question using only the patient records provided by the patient_records_search tool.
        Do NOT repeat long chunks of text.
        If the answer is not in the records, say: "Not found in the records."
        Always use the patient_records_search tool to get information before answering.""")
        
    # Create initial state
    initial_state = {
        "messages": [
            system_message,
            HumanMessage(content=question)
        ]
    }
    
    # Run the agent
    result = app.invoke(initial_state)
    
    # Extract the final answer
    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    return {
        "input": question,
        "output": answer,
        "source_documents": last_retrieved_docs
    }

if __name__ == "__main__":
    print("\n---Agentic RAG Assistant Using LangGraph---")
    pdf_path = os.getenv('PDF_PATH_FILE')
    
    # Build the retriever
    retriever = build_hybrid_rag(pdf_path)
    
    # Create the LangGraph agent
    app = create_rag_agent()
    
    # Run a query
    question = "What medications is Hassan Kim currently prescribed?"
    result = run_rag_query(app, question)
    
    # Display results
    print("Question:", result["input"])
    print("Answer:", result["output"])
    
    # Display source documents
    source_documents = result["source_documents"]
    print(f"Relevant Sources ({len(source_documents)} documents found):")
    for i, doc in enumerate(source_documents, 1):
        source_file = doc.metadata.get("source", "Unknown")
        page_num = doc.metadata.get("page", "Unknown")
        
        # Extract filename from full path
        if "\\" in source_file:
            filename = source_file.split("\\")[-1]
        elif "/" in source_file:
            filename = source_file.split("/")[-1]
        else:
            filename = source_file
            
        print(f"{i}. File: {filename}")
        print(f"   Page: {page_num}")
        print(f"   Content preview: {doc.page_content[:150]}...")
        print()