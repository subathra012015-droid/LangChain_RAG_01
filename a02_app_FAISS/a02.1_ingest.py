from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# STEP 1: Define project root
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# STEP 2: Load environment variables
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------
# STEP 3: Load source document
# ---------------------------------------------------------

file_path = BASE_DIR / "data" / "knowledge.txt"

text = file_path.read_text(encoding="utf-8")


# ---------------------------------------------------------
# STEP 4: Convert text into LangChain Document
# ---------------------------------------------------------

document = Document(page_content=text, metadata={"source": str(file_path)})

documents = [document]


# ---------------------------------------------------------
# STEP 5: Split into chunks
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

chunks = text_splitter.split_documents(documents)


# ---------------------------------------------------------
# STEP 6: Create embedding model
# ---------------------------------------------------------

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------
# STEP 7: Create FAISS vector store
# ---------------------------------------------------------

vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)


# ---------------------------------------------------------
# STEP 8: Define local save location
# ---------------------------------------------------------

faiss_path = BASE_DIR / "faiss_index"


# ---------------------------------------------------------
# STEP 9: Save FAISS locally
# ---------------------------------------------------------

vector_store.save_local(str(faiss_path))


# ---------------------------------------------------------
# STEP 10: Confirmation
# ---------------------------------------------------------

print("FAISS index created successfully.")
print("Chunks indexed:", len(chunks))
print("Saved at:", faiss_path)
