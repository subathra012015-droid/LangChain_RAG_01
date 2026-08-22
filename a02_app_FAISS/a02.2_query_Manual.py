from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# STEP 1: Define project root
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# STEP 2: Load environment variables
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------
# STEP 3: Create the embedding model
#
# IMPORTANT:
# This must be the same embedding model used during ingestion.
# ---------------------------------------------------------

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------
# STEP 4: Define FAISS index path
# ---------------------------------------------------------

faiss_path = BASE_DIR / "faiss_index"


# ---------------------------------------------------------
# STEP 5: Load the existing FAISS vector store
# ---------------------------------------------------------

vector_store = FAISS.load_local(
    str(faiss_path), embedding_model, allow_dangerous_deserialization=True
)


# ---------------------------------------------------------
# STEP 6: Create the OpenAI chat model
# ---------------------------------------------------------

chat_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


# ---------------------------------------------------------
# STEP 7: Create the system instruction
# ---------------------------------------------------------

system_message = SystemMessage(
    content=(
        "You are a helpful assistant. "
        "Answer the user's question using only the provided context. "
        "If the answer cannot be found in the context, say exactly: "
        "'I don't know based on the provided information.'"
    )
)


# ---------------------------------------------------------
# STEP 8: Start conversational loop
# ---------------------------------------------------------

print("\n==============================================")
print("ABC Electronics RAG Assistant")
print("==============================================")

print("\nType 'exit' to stop the program.")


while True:

    # -----------------------------------------------------
    # Ask the user for a question
    # -----------------------------------------------------

    question = input("\nEnter your question: ").strip()

    # -----------------------------------------------------
    # Check whether user wants to exit
    # -----------------------------------------------------

    if question.lower() == "exit":
        print("\nGoodbye.")
        break

    # -----------------------------------------------------
    # Ignore empty questions
    # -----------------------------------------------------

    if not question:
        print("Please enter a question.")
        continue

    # -----------------------------------------------------
    # Retrieve relevant chunks from FAISS
    # -----------------------------------------------------

    results = vector_store.similarity_search(question, k=3)

    # -----------------------------------------------------
    # Combine retrieved chunks into one context
    # -----------------------------------------------------

    context = "\n\n".join(result.page_content for result in results)

    # -----------------------------------------------------
    # Display retrieved information
    # -----------------------------------------------------

    print("\n--- RETRIEVED CONTEXT ---")

    print(context)

    # -----------------------------------------------------
    # Create user message
    # -----------------------------------------------------

    user_message = HumanMessage(content=f"""
Context:

{context}

Question:

{question}
""")

    # -----------------------------------------------------
    # Send context + question to the chat model
    # -----------------------------------------------------

    response = chat_model.invoke([system_message, user_message])

    # -----------------------------------------------------
    # Print final answer
    # -----------------------------------------------------

    print("\n--- FINAL ANSWER ---")

    print(response.content)
