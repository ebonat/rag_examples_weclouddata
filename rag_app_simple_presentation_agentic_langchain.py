import os
import logging
import warnings
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.schema import Document
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.db.duckdb").setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()


class AgenticRAGSystem:
    """Agentic RAG system with tools for each retrieval step"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.vectorstore = None
        self.semantic_retriever = None
        self.keyword_retriever = None
        self.hybrid_retriever = None
        self.embeddings = OpenAIEmbeddings()
        self.last_retrieved_docs = []  # Store the last retrieved documents
        
    def load_pdf_tool(self, input_str: str) -> str:
        """Tool to load PDF document"""
        try:
            loader = PyPDFLoader(self.pdf_path)
            self.documents = loader.load()
            return f"Successfully loaded PDF with {len(self.documents)} pages."
        except Exception as e:
            return f"Error loading PDF: {str(e)}"
    
    def create_chunks_tool(self, input_str: str) -> str:
        """Tool to split documents into chunks"""
        try:
            if not self.documents:
                return "Error: No documents loaded. Please load PDF first."
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, 
                chunk_overlap=100
            )
            self.chunks = splitter.split_documents(self.documents)
            return f"Successfully created {len(self.chunks)} text chunks."
        except Exception as e:
            return f"Error creating chunks: {str(e)}"
    
    def create_vector_store_tool(self, input_str: str) -> str:
        """Tool to create vector store and semantic retriever"""
        try:
            if not self.chunks:
                return "Error: No chunks available. Please create chunks first."
            
            self.vectorstore = Chroma.from_documents(self.chunks, self.embeddings)
            self.semantic_retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
            )
            return "Successfully created vector store and semantic retriever."
        except Exception as e:
            return f"Error creating vector store: {str(e)}"
    
    def create_keyword_retriever_tool(self, input_str: str) -> str:
        """Tool to create BM25 keyword retriever"""
        try:
            if not self.chunks:
                return "Error: No chunks available. Please create chunks first."
            
            self.keyword_retriever = BM25Retriever.from_documents(self.chunks)
            self.keyword_retriever.k = 3
            return "Successfully created BM25 keyword retriever."
        except Exception as e:
            return f"Error creating keyword retriever: {str(e)}"
    
    def create_ensemble_retriever_tool(self, input_str: str) -> str:
        """Tool to create hybrid ensemble retriever"""
        try:
            if not self.semantic_retriever:
                return "Error: Semantic retriever not available. Please create vector store first."
            if not self.keyword_retriever:
                return "Error: Keyword retriever not available. Please create keyword retriever first."
            
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.semantic_retriever, self.keyword_retriever],
                weights=[0.6, 0.4]
            )
            return "Successfully created hybrid ensemble retriever (60% semantic, 40% keyword)."
        except Exception as e:
            return f"Error creating ensemble retriever: {str(e)}"
    
    def query_documents_tool(self, query: str) -> str:
        """Tool to query documents using the hybrid retriever"""
        try:
            if not self.hybrid_retriever:
                return "Error: Hybrid retriever not available. Please set up all retrievers first."

            # Retrieve relevant documents
            docs = self.hybrid_retriever.get_relevant_documents(query)
            
            # Store retrieved documents for later access
            
            self.last_retrieved_docs = docs

            if not docs:
                return "Not found in the records."

            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])

            # Add source information
            sources = set([doc.metadata.get("source", "Unknown") for doc in docs])
            sources_str = f"Sources: {', '.join(sources)}"

            # Return context and sources for the agent's LLM to process
            return f"""Retrieved relevant patient record information:
                {context}
                {sources_str}
                Please answer the question based on the above patient records. Do NOT repeat long chunks of text. If the answer is not in the records, say: "Not found in the records." """
        
        except Exception as e:
            return f"Error querying documents: {str(e)}"
    
    def get_source_documents(self) -> List[Document]:
        """Get the last retrieved source documents"""
        return self.last_retrieved_docs
    
    def get_tools(self) -> List[Tool]:
        """Get all available tools for the agent"""
        return [
            Tool(
                name="load_pdf",
                func=self.load_pdf_tool,
                description="Load PDF document from the specified path. Use this first before any other operations."
            ),
            Tool(
                name="create_chunks",
                func=self.create_chunks_tool,
                description="Split loaded documents into smaller chunks for processing. Must load PDF first."
            ),
            Tool(
                name="create_vector_store",
                func=self.create_vector_store_tool,
                description="Create vector store and semantic retriever using embeddings. Requires chunks to be created first."
            ),
            Tool(
                name="create_keyword_retriever",
                func=self.create_keyword_retriever_tool,
                description="Create BM25 keyword-based retriever for exact word matching. Requires chunks to be created first."
            ),
            Tool(
                name="create_ensemble_retriever",
                func=self.create_ensemble_retriever_tool,
                description="Combine semantic and keyword retrievers into a hybrid ensemble retriever. Both retrievers must be created first."
            ),
            Tool(
                name="query_documents",
                func=self.query_documents_tool,
                description="Query the documents using the hybrid retriever to answer questions. All retrievers must be set up first. Input should be the question to answer."
            ),
        ]


def create_agentic_rag_system(pdf_path: str):
    """Create an agentic RAG system with LLM-powered decision making"""
    
    # Initialize the RAG system
    rag_system = AgenticRAGSystem(pdf_path)
    
    # Get tools
    tools = rag_system.get_tools()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.2
    )
    
    # Initialize agent with OPENAI_FUNCTIONS type
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10,
        agent_kwargs={
            "system_message": """You are an intelligent assistant that helps set up and query a RAG system for medical records.
            When a user asks a question, follow these steps in order:
            1. Load the PDF document using load_pdf
            2. Create text chunks using create_chunks
            3. Create the vector store and semantic retriever using create_vector_store
            4. Create the keyword retriever using create_keyword_retriever
            5. Create the ensemble retriever using create_ensemble_retriever
            6. Query the documents using query_documents with the user's question
            Execute these steps sequentially. Each step must complete before moving to the next."""
        }
    )
    return agent, rag_system

if __name__ == "__main__":
    print("\n--- Agentic RAG Assistant Using LangChain ---\n")
    
    pdf_path = os.getenv('PDF_PATH_FILE')
    
    # Create the agentic RAG system (now returns both agent and rag_system)
    agent, rag_system = create_agentic_rag_system(pdf_path)
    
    # Example query
    question = "What medications is Hassan Kim currently prescribed?"    
    print(f"Question: {question}\n")
        
    response = agent.invoke({"input": question})
    
    print("Response:")
    print(response["output"])
    
    # Get and print source documents
    print("\n" + "="*50)
    print("SOURCE DOCUMENTS:")
    print("="*50)
    
    source_docs = rag_system.get_source_documents()
    
    if source_docs:
        for i, doc in enumerate(source_docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"Content:\n{doc.page_content[:500]}...")  # Print first 500 chars
            print("-" * 50)
    else:
        print("No source documents found.")