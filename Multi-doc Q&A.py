from langchain_community.document_loaders import PyPDFLoader

# Load multiple pdf's
docs = PyPDFLoader("rag_implement.pdf").load() + PyPDFLoader("Lecture1_py.pdf").load()

# Splitting texts
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Converting text into vectors
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Storing vectors
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()                      # Fetching related docs from pdf

#Load models and chain
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

llm = ChatGroq(
api_key="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",
model="meta-llama/llama-4-maverick-17b-128e-instruct"
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="refine")

# Asking qureys
query = "What is python \n what is RAG"
answer = qa_chain.run(query)
print(answer)

