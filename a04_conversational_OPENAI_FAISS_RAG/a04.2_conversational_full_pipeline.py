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
# STEP 2: LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv(BASE_DIR / ".env")


# =========================================================
# STEP 3: EMBEDDING MODEL
# =========================================================

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# =========================================================
# STEP 4: LOAD FAISS
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
# STEP 8: IN-MEMORY HISTORY
# =========================================================

chat_history = []


# =========================================================
# FUNCTION: FORMAT CHAT HISTORY
# =========================================================


def format_history():

    if not chat_history:
        return "No previous conversation."

    return "\n".join(
        f"{message['role']}: {message['content']}" for message in chat_history
    )


# =========================================================
# FUNCTION: FORMAT RETRIEVED DOCUMENTS
# =========================================================


def format_documents(documents):

    return "\n\n".join(document.page_content for document in documents)


# =========================================================
# STEP 9: QUESTION REWRITE PROMPT
# =========================================================

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Rewrite the user's question as a standalone question.

Use conversation history only to resolve references such as:
it, this, that, they, them, he, she, the product, or similar references.

Do not answer the question.

Return only the standalone question.
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
# STEP 10: QUESTION REWRITE CHAIN
# =========================================================

rewrite_chain = rewrite_prompt | chat_model | output_parser


# =========================================================
# STEP 11: RAG ANSWER PROMPT
# =========================================================

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful ABC Electronics assistant.

Answer factual questions using only the retrieved context.

Conversation history may help understand the discussion,
but factual company information must come from retrieved context.

If the answer cannot be found in the retrieved context, say exactly:

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

Standalone question used for retrieval:

{standalone_question}
""",
        ),
    ]
)


# =========================================================
# STEP 12: RAG ANSWER CHAIN
# =========================================================

rag_answer_chain = rag_prompt | chat_model | output_parser


# =========================================================
# STEP 13: HELPER
# CREATE INPUT FOR REWRITE CHAIN
# =========================================================


def create_rewrite_input(data):

    return {"history": data["history"], "question": data["question"]}


# =========================================================
# STEP 14: HELPER
# RETRIEVE USING STANDALONE QUESTION
# =========================================================


def retrieve_context(data):

    documents = retriever.invoke(data["standalone_question"])

    context = format_documents(documents)

    return {**data, "context": context}


# =========================================================
# STEP 15: CREATE STANDALONE QUESTION
# =========================================================


def add_standalone_question(data):

    standalone_question = rewrite_chain.invoke(
        {"history": data["history"], "question": data["question"]}
    )

    return {**data, "standalone_question": standalone_question}


# =========================================================
# STEP 16: BUILD FULL CONVERSATIONAL PIPELINE
# =========================================================

conversation_rag_chain = (
    RunnableLambda(add_standalone_question)
    | RunnableLambda(retrieve_context)
    | rag_answer_chain
)


# =========================================================
# STEP 17: MAIN LOOP
# =========================================================

print("\n" + "=" * 70)
print("ABC Electronics Conversational RAG Pipeline")
print("=" * 70)

print("\nCommands:")
print("history -> show conversation")
print("clear   -> clear conversation")
print("exit    -> close program")


while True:

    question = input("\nYou: ").strip()

    if not question:
        print("Please enter a question.")
        continue

    if question.lower() == "exit":

        print("\nAssistant: Goodbye.")
        break

    if question.lower() == "history":

        print("\n--- CONVERSATION HISTORY ---")
        print(format_history())

        continue

    if question.lower() == "clear":

        chat_history.clear()

        print("\nAssistant: Conversation history cleared.")

        continue

    # -----------------------------------------------------
    # BUILD INPUT
    # -----------------------------------------------------

    pipeline_input = {"history": format_history(), "question": question}

    # -----------------------------------------------------
    # RUN COMPLETE CONVERSATIONAL RAG PIPELINE
    # -----------------------------------------------------

    answer = conversation_rag_chain.invoke(pipeline_input)

    # -----------------------------------------------------
    # DISPLAY ANSWER
    # -----------------------------------------------------

    print("\nAssistant:")
    print(answer)

    # -----------------------------------------------------
    # SAVE QUESTION
    # -----------------------------------------------------

    chat_history.append({"role": "User", "content": question})

    # -----------------------------------------------------
    # SAVE ANSWER
    # -----------------------------------------------------

    chat_history.append({"role": "Assistant", "content": answer})
