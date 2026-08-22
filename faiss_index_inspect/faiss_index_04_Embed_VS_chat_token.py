from pathlib import Path
import pickle

import faiss
import tiktoken

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# =========================================================
# STEP 1: PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FAISS_DIR = BASE_DIR / "faiss_index"

FAISS_FILE = FAISS_DIR / "index.faiss"
PKL_FILE = FAISS_DIR / "index.pkl"


# =========================================================
# STEP 2: LOAD .ENV
# =========================================================

load_dotenv(BASE_DIR / ".env")


# =========================================================
# STEP 3: MODEL NAMES
# =========================================================

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHAT_MODEL_NAME = "gpt-4.1-mini"


# =========================================================
# STEP 4: CREATE TOKENIZER
# =========================================================

# Try to get the tokenizer appropriate for the
# embedding model.
try:
    encoding = tiktoken.encoding_for_model(
        EMBEDDING_MODEL_NAME
    )
except KeyError:
    # Fallback if model name is not recognized
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )


# =========================================================
# STEP 5: LOAD FAISS INDEX
# =========================================================

index = faiss.read_index(
    str(FAISS_FILE)
)


# =========================================================
# STEP 6: LOAD PKL DOCUMENT STORE
#
# Only load pickle files you created/trust.
# =========================================================

with open(PKL_FILE, "rb") as file:

    docstore, index_to_docstore_id = pickle.load(file)


# =========================================================
# STEP 7: CREATE MODELS
# =========================================================

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME
)

chat_model = ChatOpenAI(
    model=CHAT_MODEL_NAME,
    temperature=0
)


# =========================================================
# FUNCTION: SHOW TOKEN DETAILS
# =========================================================

def show_tokens(text, title):

    token_ids = encoding.encode(text)

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print("\nTEXT:")
    print(text)

    print("\nINPUT TOKEN COUNT:")
    print(len(token_ids))

    print("\nTOKEN DETAILS:")

    print(
        f"{'Position':<12}"
        f"{'Token ID':<15}"
        f"{'Token':<35}"
    )

    print("-" * 62)

    for position, token_id in enumerate(token_ids):

        token_bytes = encoding.decode_single_token_bytes(
            token_id
        )

        token_text = token_bytes.decode(
            "utf-8",
            errors="replace"
        )

        print(
            f"{position:<12}"
            f"{token_id:<15}"
            f"{repr(token_text):<35}"
        )

    return token_ids


# =========================================================
# STEP 8: GET FIRST TWO CHUNKS
# =========================================================

chunks = []

for faiss_position in range(
    min(2, index.ntotal)
):

    document_id = index_to_docstore_id[
        faiss_position
    ]

    document = docstore.search(
        document_id
    )

    vector = index.reconstruct(
        faiss_position
    )

    chunks.append(
        {
            "position": faiss_position,
            "document_id": document_id,
            "document": document,
            "vector": vector
        }
    )


# =========================================================
# STEP 9: INSPECT EACH CHUNK
# =========================================================

for chunk_number, chunk in enumerate(
    chunks,
    start=1
):

    text = chunk["document"].page_content

    vector = chunk["vector"]


    # -----------------------------------------------------
    # TOKEN INFORMATION
    # -----------------------------------------------------

    token_ids = show_tokens(
        text,
        f"CHUNK {chunk_number} - EMBEDDING INPUT"
    )


    # -----------------------------------------------------
    # VECTOR INFORMATION
    # -----------------------------------------------------

    print("\nEMBEDDING MODEL:")
    print(EMBEDDING_MODEL_NAME)

    print("\nINPUT TOKENS:")
    print(len(token_ids))

    print("\nOUTPUT TEXT TOKENS:")
    print("Not applicable")

    print("\nOUTPUT VECTOR DIMENSION:")
    print(len(vector))

    print("\nFIRST 10 ACTUAL VECTOR VALUES:")

    for dimension, value in enumerate(
        vector[:10],
        start=1
    ):

        print(
            f"D{dimension:<4} = {float(value)}"
        )


# =========================================================
# STEP 10: ASK USER QUESTION
# =========================================================

question = input(
    "\nEnter a question: "
).strip()


# =========================================================
# STEP 11: TOKENIZE QUESTION FOR EMBEDDING
# =========================================================

question_token_ids = show_tokens(
    question,
    "QUESTION - EMBEDDING INPUT"
)


# =========================================================
# STEP 12: CREATE ACTUAL QUESTION EMBEDDING
# =========================================================

question_vector = embedding_model.embed_query(
    question
)

print("\n" + "=" * 90)
print("QUESTION EMBEDDING RESULT")
print("=" * 90)

print("\nEmbedding model:")
print(EMBEDDING_MODEL_NAME)

print("\nInput token count:")
print(len(question_token_ids))

print("\nOutput text tokens:")
print("Not applicable")

print("\nOutput vector dimension:")
print(len(question_vector))

print("\nFirst 10 actual query-vector values:")

for dimension, value in enumerate(
    question_vector[:10],
    start=1
):

    print(
        f"D{dimension:<4} = {float(value)}"
    )


# =========================================================
# STEP 13: BUILD CONTEXT USING TWO CHUNKS
# =========================================================

context = "\n\n".join(
    chunk["document"].page_content
    for chunk in chunks
)


# =========================================================
# STEP 14: CREATE CHAT MESSAGES
# =========================================================

system_message = SystemMessage(
    content=(
        "Answer the user's question using only "
        "the provided context. "
        "If the answer is not available, say: "
        "'I don't know based on the provided information.'"
    )
)


human_message = HumanMessage(
    content=f"""
Context:

{context}

Question:

{question}
"""
)


# =========================================================
# STEP 15: DISPLAY WHAT CHAT MODEL RECEIVES
# =========================================================

print("\n" + "=" * 90)
print("CHAT MODEL INPUT")
print("=" * 90)

print("\nSYSTEM MESSAGE:")
print(system_message.content)

print("\nHUMAN MESSAGE:")
print(human_message.content)


# =========================================================
# STEP 16: CALL CHAT MODEL
# =========================================================

response = chat_model.invoke(
    [
        system_message,
        human_message
    ]
)


# =========================================================
# STEP 17: DISPLAY ANSWER
# =========================================================

print("\n" + "=" * 90)
print("CHAT MODEL OUTPUT")
print("=" * 90)

print("\nANSWER:")

print(response.content)


# =========================================================
# STEP 18: DISPLAY ACTUAL CHAT TOKEN USAGE
# =========================================================

print("\n" + "=" * 90)
print("CHAT MODEL TOKEN USAGE")
print("=" * 90)


# LangChain AIMessage commonly exposes normalized usage
# information in usage_metadata.

usage = response.usage_metadata

if usage:

    print("\nInput tokens:")
    print(
        usage.get(
            "input_tokens",
            "Not available"
        )
    )

    print("\nOutput tokens:")
    print(
        usage.get(
            "output_tokens",
            "Not available"
        )
    )

    print("\nTotal tokens:")
    print(
        usage.get(
            "total_tokens",
            "Not available"
        )
    )

else:

    print(
        "\nToken usage was not available "
        "in response.usage_metadata."
    )

    print("\nRaw response metadata:")

    print(
        response.response_metadata
    )


# =========================================================
# STEP 19: FINAL COMPARISON
# =========================================================

print("\n" + "=" * 90)
print("FINAL COMPARISON")
print("=" * 90)

print("\nEMBEDDING MODEL")
print("------------------------------")

print(
    "Model                :",
    EMBEDDING_MODEL_NAME
)

print(
    "Question input tokens:",
    len(question_token_ids)
)

print(
    "Output text tokens   :",
    "None"
)

print(
    "Output dimensions    :",
    len(question_vector)
)


print("\nCHAT MODEL")
print("------------------------------")

print(
    "Model                :",
    CHAT_MODEL_NAME
)

if usage:

    print(
        "Input tokens         :",
        usage.get(
            "input_tokens",
            "Not available"
        )
    )

    print(
        "Output tokens        :",
        usage.get(
            "output_tokens",
            "Not available"
        )
    )

    print(
        "Total tokens         :",
        usage.get(
            "total_tokens",
            "Not available"
        )
    )

print(
    "Output type           :",
    "Natural-language text"
)