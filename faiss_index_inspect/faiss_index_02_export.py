from pathlib import Path
import pickle
import csv
import json

import faiss

# =========================================================
# STEP 1: Define project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FAISS_DIR = BASE_DIR / "faiss_index"

FAISS_FILE = FAISS_DIR / "index.faiss"
PKL_FILE = FAISS_DIR / "index.pkl"

OUTPUT_FILE = BASE_DIR / "faiss_readable_data.csv"


# =========================================================
# STEP 2: Check whether files exist
# =========================================================

if not FAISS_FILE.exists():
    raise FileNotFoundError(f"FAISS file not found: {FAISS_FILE}")

if not PKL_FILE.exists():
    raise FileNotFoundError(f"PKL file not found: {PKL_FILE}")


# =========================================================
# STEP 3: Load index.faiss
# =========================================================

print("Loading index.faiss...")

index = faiss.read_index(str(FAISS_FILE))

print("FAISS index loaded.")
print("Number of vectors:", index.ntotal)
print("Vector dimension:", index.d)


# =========================================================
# STEP 4: Load index.pkl
#
# WARNING:
# Only load pickle files that you created/trust.
# =========================================================

print("\nLoading index.pkl...")

with open(PKL_FILE, "rb") as file:
    data = pickle.load(file)


# LangChain FAISS normally stores:
#
# (
#     docstore,
#     index_to_docstore_id
# )

docstore, index_to_docstore_id = data

print("PKL file loaded.")
print("Document mappings:", len(index_to_docstore_id))


# =========================================================
# STEP 5: Prepare CSV columns
# =========================================================

fieldnames = [
    "FAISS_Position",
    "Document_ID",
    "Chunk_Text",
    "Source",
    "Metadata",
    "Vector_Dimension",
    "Vector_First_10",
]


# =========================================================
# STEP 6: Create CSV
# =========================================================

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as csv_file:

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    # =====================================================
    # STEP 7: Connect FAISS vectors with documents
    # =====================================================

    for faiss_position, document_id in sorted(index_to_docstore_id.items()):

        # -------------------------------------------------
        # Get the original LangChain Document
        # -------------------------------------------------

        document = docstore.search(document_id)

        # -------------------------------------------------
        # Get the actual vector from index.faiss
        # -------------------------------------------------

        vector = index.reconstruct(faiss_position)

        # -------------------------------------------------
        # Convert first 10 values into readable numbers
        # -------------------------------------------------

        first_10_values = [float(value) for value in vector[:10]]

        # -------------------------------------------------
        # Get source from metadata
        # -------------------------------------------------

        source = document.metadata.get("source", "")

        # -------------------------------------------------
        # Convert complete metadata to readable JSON
        # -------------------------------------------------

        metadata_json = json.dumps(document.metadata, ensure_ascii=False)

        # -------------------------------------------------
        # Write one row
        # -------------------------------------------------

        writer.writerow(
            {
                "FAISS_Position": faiss_position,
                "Document_ID": document_id,
                "Chunk_Text": document.page_content,
                "Source": source,
                "Metadata": metadata_json,
                "Vector_Dimension": len(vector),
                "Vector_First_10": json.dumps(first_10_values),
            }
        )


# =========================================================
# STEP 8: Confirmation
# =========================================================

print("\n==============================================")
print("EXPORT COMPLETED")
print("==============================================")

print("Output file:")
print(OUTPUT_FILE)

print("\nRows exported:")
print(len(index_to_docstore_id))

print("\nVector dimension:")
print(index.d)
