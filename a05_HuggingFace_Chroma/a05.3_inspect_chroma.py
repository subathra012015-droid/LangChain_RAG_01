from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LOAD EXISTING CHROMA COLLECTION
# =========================================================

vector_store = Chroma(
    collection_name="abc_electronics",
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_DIR),
)


# =========================================================
# COLLECTION COUNT
# =========================================================

count = vector_store._collection.count()

print("\n" + "=" * 70)
print("CHROMA COLLECTION INFORMATION")
print("=" * 70)

print(f"Collection name : abc_electronics")
print(f"Record count    : {count}")
print(f"Database path   : {CHROMA_DIR}")


# =========================================================
# GET CONTENT INCLUDING EMBEDDINGS
# =========================================================

results = vector_store._collection.get(include=["documents", "metadatas", "embeddings"])


# =========================================================
# DISPLAY EACH RECORD
# =========================================================

for index, record_id in enumerate(results["ids"], start=1):

    document = results["documents"][index - 1]
    metadata = results["metadatas"][index - 1]
    embedding = results["embeddings"][index - 1]

    print("\n" + "=" * 70)
    print(f"RECORD {index}")
    print("=" * 70)

    print(f"\nID:")
    print(record_id)

    print("\nDOCUMENT:")
    print(document)

    print("\nMETADATA:")
    print(metadata)

    print("\nVECTOR DIMENSION:")
    print(len(embedding))

    print("\nFIRST 10 VECTOR VALUES:")

    for dimension, value in enumerate(embedding[:10], start=1):
        print(f"Dimension {dimension}: {value}")
