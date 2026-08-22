from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)


# =========================================================
# STEP 1: DEFINE PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================================================
# STEP 2: LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv(BASE_DIR / ".env")


# =========================================================
# STEP 3: CREATE EMBEDDING MODEL
#
# IMPORTANT:
# Must match the model used while creating the FAISS index.
# =========================================================

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# =========================================================
# STEP 4: LOAD EXISTING FAISS VECTOR STORE
# =========================================================

faiss_path = BASE_DIR / "faiss_index"

vector_store = FAISS.load_local(
    str(faiss_path),
    embedding_model,
    allow_dangerous_deserialization=True
)


# =========================================================
# STEP 5: CONVERT FAISS INTO A RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


# =========================================================
# STEP 6: FUNCTION TO FORMAT RETRIEVED DOCUMENTS
# =========================================================

def format_documents(documents):

    formatted_text = "\n\n".join(
        document.page_content
        for document in documents
    )

    return formatted_text


# =========================================================
# STEP 7: CREATE CHAT PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful ABC Electronics assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say exactly:

I don't know based on the provided information.
"""
        ),
        (
            "human",
            """
Context:

{context}


Question:

{question}
"""
        )
    ]
)


# =========================================================
# STEP 8: CREATE CHAT MODEL
# =========================================================

chat_model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# =========================================================
# STEP 9: CREATE OUTPUT PARSER
# =========================================================

output_parser = StrOutputParser()


# =========================================================
# STEP 10: CREATE DOCUMENT FORMATTER RUNNABLE
# =========================================================

document_formatter = RunnableLambda(
    format_documents
)


# =========================================================
# STEP 11: BUILD LANGCHAIN RAG PIPELINE
# =========================================================

rag_chain = (

    {
        "context":
            retriever
            | document_formatter,

        "question":
            RunnablePassthrough()
    }

    | prompt       | chat_model      | output_parser
)


# =========================================================
# STEP 12: ASK USER QUESTION
# =========================================================

question = input(
    "\nEnter your question: "
).strip()


# =========================================================
# STEP 13: EXECUTE COMPLETE RAG PIPELINE
# =========================================================

answer = rag_chain.invoke(
    question
)


# =========================================================
# STEP 14: DISPLAY FINAL ANSWER
# =========================================================

print("\n" + "=" * 70)

print("FINAL ANSWER")

print("=" * 70)

print(answer)