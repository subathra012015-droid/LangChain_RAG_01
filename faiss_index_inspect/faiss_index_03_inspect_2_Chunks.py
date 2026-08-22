from pathlib import Path
import pickle

import faiss
import tiktoken

# =========================================================
# STEP 1: Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FAISS_DIR = BASE_DIR / "faiss_index"

FAISS_FILE = FAISS_DIR / "index.faiss"
PKL_FILE = FAISS_DIR / "index.pkl"


# =========================================================
# STEP 2: Load FAISS
# =========================================================

index = faiss.read_index(str(FAISS_FILE))


# =========================================================
# STEP 3: Load LangChain document mapping
# =========================================================

with open(PKL_FILE, "rb") as file:
    docstore, index_to_docstore_id = pickle.load(file)


# =========================================================
# STEP 4: Create tokenizer
# =========================================================

encoding = tiktoken.get_encoding("cl100k_base")


# =========================================================
# STEP 5: Inspect first 2 chunks
# =========================================================

for faiss_position in range(min(2, index.ntotal)):

    document_id = index_to_docstore_id[faiss_position]

    document = docstore.search(document_id)

    chunk_text = document.page_content

    vector = index.reconstruct(faiss_position)

    print("\n")
    print("=" * 80)
    print(f"CHUNK {faiss_position + 1}")
    print("=" * 80)

    # =====================================================
    # CHUNK INFORMATION
    # =====================================================

    print("\nCHUNK TEXT:")
    print(chunk_text)

    print("\nDOCUMENT ID:")
    print(document_id)

    print("\nVECTOR DIMENSION:")
    print(len(vector))

    # =====================================================
    # TOKEN INFORMATION
    # =====================================================

    print("\n")
    print("-" * 80)
    print("TOKENS AND TOKEN IDs")
    print("-" * 80)

    token_ids = encoding.encode(chunk_text)

    print(f"{'Position':<10}" f"{'Token ID':<12}" f"{'Token':<30}")

    print("-" * 55)

    for position, token_id in enumerate(token_ids):

        token_bytes = encoding.decode_single_token_bytes(token_id)

        token_text = token_bytes.decode("utf-8", errors="replace")

        print(f"{position:<10}" f"{token_id:<12}" f"{repr(token_text):<30}")

    # =====================================================
    # VECTOR INFORMATION
    # =====================================================

    print("\n")
    print("-" * 80)
    print("CHUNK EMBEDDING VECTOR")
    print("-" * 80)

    print("\nFirst 20 dimensions:\n")

    for dimension, value in enumerate(vector[:20], start=1):

        print(f"D{dimension:<5} = {float(value)}")

    print("\n...")

    print(f"\nTotal dimensions = {len(vector)}")
