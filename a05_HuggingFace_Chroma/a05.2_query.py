from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================================================
# STEP 1: DEFINE PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# STEP 2: CREATE THE SAME EMBEDDING MODEL
#
# IMPORTANT:
# This must be the same model used in a05.1_ingest.py
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# STEP 3: LOAD EXISTING CHROMA COLLECTION
# =========================================================

vector_store = Chroma(
    collection_name="abc_electronics",
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_DIR),
)


# =========================================================
# STEP 4: CHECK COLLECTION RECORD COUNT
# =========================================================

record_count = vector_store._collection.count()

print("\n" + "=" * 70)
print("CHROMA QUERY APPLICATION")
print("=" * 70)

print(f"\nCollection name : abc_electronics")
print(f"Records stored  : {record_count}")
print(f"Database path   : {CHROMA_DIR}")


# =========================================================
# STEP 5: ASK USER QUESTION
# =========================================================

question = input("\nEnter your question: ").strip()


# =========================================================
# STEP 6: VALIDATE QUESTION
# =========================================================

if not question:
    raise ValueError("Question cannot be empty.")


# =========================================================
# STEP 7: CREATE QUERY EMBEDDING
#
# This is just for learning/inspection.
# Chroma will also use the embedding model internally
# during similarity search.
# =========================================================

query_vector = embedding_model.embed_query(question)


print("\n" + "=" * 70)
print("QUERY EMBEDDING INFORMATION")
print("=" * 70)

print(f"\nQuestion:")
print(question)

print(f"\nQuery vector dimensions: {len(query_vector)}")

print("\nFirst 10 query-vector values:")

for dimension, value in enumerate(query_vector[:10], start=1):
    print(f"Dimension {dimension}: {value}")


# =========================================================
# STEP 8: RUN SIMILARITY SEARCH
#
# k=3 means:
# return the top 3 most similar records
# =========================================================

results = vector_store.similarity_search(query=question, k=3)


# =========================================================
# STEP 9: DISPLAY RETRIEVED DOCUMENTS
# =========================================================

print("\n" + "=" * 70)
print("SIMILARITY SEARCH RESULTS")
print("=" * 70)

print(f"\nNumber of documents retrieved: {len(results)}")


for index, document in enumerate(results, start=1):

    print("\n" + "-" * 70)

    print(f"RESULT {index}")

    print("-" * 70)

    print("\nDOCUMENT:")
    print(document.page_content)

    print("\nMETADATA:")
    print(document.metadata)


# =========================================================
# STEP 10: COMPLETE
# =========================================================

print("\nQuery completed successfully.")
