
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# suppress chromadb warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.db.duckdb").setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

# custom prompt 
prompt_template = PromptTemplate(
template="""
You are a medical assistant.
Answer the user's question using only the patient records provided.
Do NOT repeat long chunks of text.
If the answer is not in the records, say: "Not found in the records."
Context:
{context}
Question: {question}
Answer clearly and directly:
""",
input_variables=["context", "question"],
)

def build_hybrid_rag(pdf_path: str):
    # 1. load pdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # chunk_size = how much text (characters) goes into each chunk.
    # “How big is each puzzle piece?”
    # Larger chunks = better context, fewer pieces
    # Smaller chunks = more precise retrieval, more pieces

    # chunk_overlap = how many characters of context spill into the next chunk.
    # “How much glue keeps the puzzle pieces connected?”
    # Overlap prevents the model from losing meaning across boundaries.
    
    # For most RAG systems
    # chunk_size = 500–1000
    # chunk_overlap = 100–300

    # 2. split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Semantic retrieval = finding the most semantically similar chunks using embeddings.
    # Measuring similarity using cosine similarity, dot product, etc
    # It retrieves information based on meaning, not matching words.

    # cosine similarity
    # When text is converted into embeddings (vectors), each sentence becomes a point in high-dimensional space.
    # Do these two vectors point in nearly the same direction?
    # If yes → high similarity
    # If no → low similarity

    # 3a. semantic retriever (faiss)
    embeddings = OpenAIEmbeddings()
    # 1. using FAISS or Chroma
    # vectorstore = FAISS.from_documents(docs, embeddings)
    # 2.using Chroma
    vectorstore = Chroma.from_documents(docs, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # A keyword retriever returns documents based on literal keyword overlap between the user’s query and the documents.
    # “Find me documents that contain the same words I typed.”
    
    # 3b. keyword retriever (bm25)
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 3

    # A hybrid retriever (semantic + keyword search) will improve accuracy, especially with structured records like PDFs where names, 
    # medications, or lab values might not embed well semantically.
    # In LangChain, we can combine:
    # FAISS (vector search) → semantic similarity search
    # BM25 (keyword search) → exact word matching
    # and then merge their results.

    # 3c. combine them into a hybrid retriever
    retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.6, 0.4] # semantic prioritized, but keywords matter
    )

    # temperature controls randomness in the model's output:
    # Low values → more focused, factual, reproducible
    # High values → more creative, varied, less predictable
    # For Fact-based RAG Assistant recommended:0.0 – 0.3, Ensures precise, stable answers from retrieved context
    
    # 4. rag chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", 
            api_key=os.getenv('OPENAI_API_KEY'), 
            temperature=0.2),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa_chain

if __name__ == "__main__":
    print("\n---Classic RAG Assistant Using LangChain---")
    pdf_path = os.getenv('PDF_PATH_FILE')
    qa_chain = build_hybrid_rag(pdf_path)

    question = "What medications is Hassan Kim currently prescribed?"
    print("Question:", question)
    response = qa_chain.invoke({"query": question}) 
    print("Answer:", response["result"])
    
    source_documents = response["source_documents"]
    print(f"Relevant Sources ({len(source_documents)} documents found):")
    for doc in response["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"))
        

