from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# =========================================================
# STEP 1: PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================================================
# STEP 2: LOAD .ENV
# =========================================================

load_dotenv(BASE_DIR / ".env")


# =========================================================
# STEP 3: EMBEDDING MODEL
# Must match ingestion model
# =========================================================

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# =========================================================
# STEP 4: LOAD SAVED FAISS
# =========================================================

faiss_path = BASE_DIR / "faiss_index"

vector_store = FAISS.load_local(
    str(faiss_path), embedding_model, allow_dangerous_deserialization=True
)


# =========================================================
# STEP 5: CREATE RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# =========================================================
# STEP 6: CHAT MODEL
# =========================================================

chat_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


# =========================================================
# STEP 7: OUTPUT PARSER
# =========================================================

output_parser = StrOutputParser()


# =========================================================
# STEP 8: IN-MEMORY CONVERSATION HISTORY
# =========================================================

chat_history = []


# =========================================================
# FUNCTION 1:
# FORMAT RETRIEVED DOCUMENTS
# =========================================================


def format_documents(documents):

    return "\n\n".join(document.page_content for document in documents)


# =========================================================
# FUNCTION 2:
# FORMAT CHAT HISTORY
# =========================================================


def format_history():

    if not chat_history:
        return "No previous conversation."

    lines = []

    for message in chat_history:

        lines.append(f"{message['role']}: {message['content']}")

    return "\n".join(lines)


# =========================================================
# STEP 9:
# QUESTION REWRITING PROMPT
# =========================================================

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You rewrite follow-up questions into clear standalone questions.

Use conversation history only to understand references
such as:
it,
this,
that,
they,
them,
the product,
the laptop,
or similar references.

Do not answer the question.

Return only the rewritten standalone question.
""",
        ),
        (
            "human",
            """
Conversation history:

{history}


Current question:

{question}
""",
        ),
    ]
)


# =========================================================
# STEP 10:
# CREATE QUESTION REWRITING PIPELINE
# =========================================================

rewrite_chain = rewrite_prompt | chat_model | output_parser


# =========================================================
# STEP 11:
# FINAL RAG PROMPT
# =========================================================

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful ABC Electronics assistant.

Answer factual questions using only the retrieved context.

Conversation history may help you understand the discussion,
but company facts must come from the retrieved context.

If the answer cannot be found in the retrieved context,
say exactly:

I don't know based on the provided information.
""",
        ),
        (
            "human",
            """
Conversation history:

{history}


Retrieved context:

{context}


Original question:

{question}
""",
        ),
    ]
)


# =========================================================
# STEP 12:
# RAG ANSWER PIPELINE
# =========================================================

rag_answer_chain = rag_prompt | chat_model | output_parser


# =========================================================
# STEP 13:
# MAIN CONVERSATIONAL LOOP
# =========================================================

print("\n" + "=" * 70)

print("ABC Electronics Conversational RAG Assistant")

print("=" * 70)

print("\nCommands:")
print("history -> show conversation")
print("clear   -> clear conversation")
print("exit    -> close program")


while True:

    # -----------------------------------------------------
    # GET QUESTION
    # -----------------------------------------------------

    question = input("\nYou: ").strip()

    if not question:

        print("Please enter a question.")

        continue

    # -----------------------------------------------------
    # EXIT
    # -----------------------------------------------------

    if question.lower() == "exit":

        print("\nAssistant: Goodbye.")

        break

    # -----------------------------------------------------
    # SHOW HISTORY
    # -----------------------------------------------------

    if question.lower() == "history":

        print("\n--- CONVERSATION HISTORY ---")

        print(format_history())

        continue

    # -----------------------------------------------------
    # CLEAR HISTORY
    # -----------------------------------------------------

    if question.lower() == "clear":

        chat_history.clear()

        print("\nAssistant: Conversation history cleared.")

        continue

    # =====================================================
    # STEP A:
    # GET CURRENT HISTORY
    # =====================================================

    history = format_history()

    # =====================================================
    # STEP B:
    # CREATE STANDALONE QUESTION
    # =====================================================

    standalone_question = rewrite_chain.invoke(
        {"history": history, "question": question}
    )

    print("\n--- STANDALONE QUESTION ---")

    print(standalone_question)

    # =====================================================
    # STEP C:
    # RETRIEVE DOCUMENTS
    # =====================================================

    retrieved_documents = retriever.invoke(standalone_question)

    # =====================================================
    # STEP D:
    # FORMAT DOCUMENTS
    # =====================================================

    context = format_documents(retrieved_documents)

    print("\n--- RETRIEVED CONTEXT ---")

    print(context)

    # =====================================================
    # STEP E:
    # GENERATE ANSWER
    # =====================================================

    answer = rag_answer_chain.invoke(
        {"history": history, "context": context, "question": question}
    )

    # =====================================================
    # STEP F:
    # DISPLAY ANSWER
    # =====================================================

    print("\nAssistant:")

    print(answer)

    # =====================================================
    # STEP G:
    # SAVE QUESTION TO MEMORY
    # =====================================================

    chat_history.append({"role": "User", "content": question})

    # =====================================================
    # STEP H:
    # SAVE ANSWER TO MEMORY
    # =====================================================

    chat_history.append({"role": "Assistant", "content": answer})
