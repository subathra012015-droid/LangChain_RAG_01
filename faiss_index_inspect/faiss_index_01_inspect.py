from pathlib import Path
import pickle

import faiss


# ---------------------------------------------------------
# STEP 1: Define project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FAISS_DIR = BASE_DIR / "faiss_index"

FAISS_FILE = FAISS_DIR / "index.faiss"
PKL_FILE = FAISS_DIR / "index.pkl"


# ---------------------------------------------------------
# STEP 2: Read index.faiss
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INDEX.FAISS INFORMATION")
print("=" * 70)

index = faiss.read_index(str(FAISS_FILE))

print("Number of vectors:", index.ntotal)
print("Vector dimension:", index.d)
print("FAISS index type:", type(index).__name__)


# ---------------------------------------------------------
# STEP 3: Read individual vectors
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VECTORS STORED IN INDEX.FAISS")
print("=" * 70)

for i in range(index.ntotal):

    vector = index.reconstruct(i)

    print(f"\nVector {i}")
    print("-" * 50)

    print("Dimension:", len(vector))

    print("First 10 values:")
    print(vector[:10])


# ---------------------------------------------------------
# STEP 4: Read index.pkl
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INDEX.PKL INFORMATION")
print("=" * 70)

with open(PKL_FILE, "rb") as file:
    data = pickle.load(file)


print("Python object type:", type(data))


# ---------------------------------------------------------
# STEP 5: Inspect pickle structure
# ---------------------------------------------------------

print("\nRaw top-level structure:")
print(data)


# ---------------------------------------------------------
# STEP 6: LangChain FAISS normally stores:
#         (docstore, index_to_docstore_id)
# ---------------------------------------------------------

docstore, index_to_docstore_id = data


print("\n" + "=" * 70)
print("FAISS INDEX → DOCUMENT ID MAPPING")
print("=" * 70)

for faiss_position, document_id in index_to_docstore_id.items():

    print(
        f"FAISS position {faiss_position}"
        f" -> Document ID {document_id}"
    )


# ---------------------------------------------------------
# STEP 7: Print readable documents
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DOCUMENTS STORED IN INDEX.PKL")
print("=" * 70)

for faiss_position, document_id in index_to_docstore_id.items():

    document = docstore.search(document_id)

    print("\n" + "-" * 70)
    print(f"FAISS POSITION: {faiss_position}")
    print(f"DOCUMENT ID: {document_id}")

    print("\nCONTENT:")
    print(document.page_content)

    print("\nMETADATA:")
    print(document.metadata)