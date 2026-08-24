from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# =========================================================
# STEP 1: PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "knowledge.txt"

CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# STEP 2: READ SOURCE DATA
# =========================================================

print("\nReading source file...")

text = DATA_FILE.read_text(encoding="utf-8")

print(f"Characters loaded: {len(text)}")


# =========================================================
# STEP 3: CREATE LANGCHAIN DOCUMENT
# =========================================================

document = Document(page_content=text, metadata={"source": "knowledge.txt"})


# =========================================================
# STEP 4: TEXT SPLITTER
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)


# =========================================================
# STEP 5: SPLIT DOCUMENT INTO CHUNKS
# =========================================================

chunks = text_splitter.split_documents([document])

print(f"Chunks created: {len(chunks)}")


# =========================================================
# STEP 6: DISPLAY CHUNKS
# =========================================================

for index, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 70)

    print(f"CHUNK {index}")

    print("=" * 70)

    print(chunk.page_content)


# =========================================================
# STEP 7: CREATE HUGGING FACE EMBEDDING MODEL
# =========================================================

print("\nLoading Hugging Face embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# STEP 8: TEST ONE EMBEDDING
# =========================================================

sample_text = chunks[0].page_content

sample_vector = embedding_model.embed_query(sample_text)

print("\n" + "=" * 70)
print("EMBEDDING INFORMATION")
print("=" * 70)

print(f"Embedding dimensions: {len(sample_vector)}")

print("\nFirst 10 dimensions:")

for dimension, value in enumerate(sample_vector[:10], start=1):

    print(f"Dimension {dimension}: {value}")


# =========================================================
# STEP 9: CREATE CHROMA VECTOR DATABASE
# =========================================================

print("\nCreating Chroma vector database...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="abc_electronics",
    persist_directory=str(CHROMA_DIR),
)


# =========================================================
# STEP 10: VERIFY NUMBER OF STORED RECORDS
# =========================================================

stored_count = vector_store._collection.count()

print("\n" + "=" * 70)
print("CHROMA DATABASE")
print("=" * 70)

print(f"Records stored: {stored_count}")

print(f"Database location: {CHROMA_DIR}")


# =========================================================
# COMPLETE
# =========================================================

print("\nIngestion completed successfully.")
