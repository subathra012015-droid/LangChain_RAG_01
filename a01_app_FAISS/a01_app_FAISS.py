from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# STEP 1: Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# STEP 2: Load the knowledge file
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "knowledge.txt"

text = file_path.read_text(encoding="utf-8")


# ---------------------------------------------------------
# STEP 3: Convert text into a LangChain Document
# ---------------------------------------------------------

document = Document(page_content=text, metadata={"source": str(file_path)})

documents = [document]


# ---------------------------------------------------------
# STEP 4: Split the document into chunks
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

chunks = text_splitter.split_documents(documents)


# ---------------------------------------------------------
# STEP 5: Create the embedding model
# ---------------------------------------------------------

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------
# STEP 6: Create FAISS vector store
# ---------------------------------------------------------

vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)


# ---------------------------------------------------------
# STEP 7: Create the chat model
# ---------------------------------------------------------

chat_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


# ---------------------------------------------------------
# STEP 8: Ask the user a question
# ---------------------------------------------------------

print("\n--- SAMPLE QUESTION ---")
print("1. What is the purpose of this knowledge file?")
print("2. What are the key points mentioned in the knowledge file?")
print("3. What is the coverage period?")
print("4. Negartive Question : Warren")
print("5. Semantic Question : Can I send an unopened item back after 20 days?")
print("6. Unknown Question : Who is the CEO of ABC Electronics?")

question = input("\nEnter your question: ")


# ---------------------------------------------------------
# STEP 9: Retrieve relevant chunks from FAISS
# ---------------------------------------------------------

results = vector_store.similarity_search(question, k=3)


# ---------------------------------------------------------
# STEP 10: Combine retrieved chunks into one context string
# ---------------------------------------------------------

context_parts = []

for result in results:
    context_parts.append(result.page_content)

context = "\n\n".join(context_parts)


# ---------------------------------------------------------
# STEP 11: Display retrieved context
# ---------------------------------------------------------

print("\n--- RETRIEVED CONTEXT ---")

print(context)


# ---------------------------------------------------------
# STEP 12: Create system instruction
# ---------------------------------------------------------

system_message = SystemMessage(
    content=(
        "You are a helpful assistant. "
        "Answer the user's question using only the provided context. "
        "If the answer cannot be found in the context, say: "
        "'I don't know based on the provided information.'"
    )
)


# ---------------------------------------------------------
# STEP 13: Create user message containing context + question
# ---------------------------------------------------------

user_message = HumanMessage(content=f"""
Context:

{context}

Question:

{question}
""")


# ---------------------------------------------------------
# STEP 14: Send messages to ChatOpenAI
# ---------------------------------------------------------

response = chat_model.invoke([system_message, user_message])

print("\n--- RAW RESPONSE ---")
print("\n--- USER PROMPT  :: ", user_message.content)
print("\n--- SYSTEM PROMPT :: ", system_message.content)

# ---------------------------------------------------------
# STEP 15: Print final answer
# ---------------------------------------------------------

print("\n--- FINAL ANSWER ---")

print(response.content)
