# a01_app_FAISS -> Basic RAG => User Prompt , System Prompt

User Prompt -> Question
   ↓
Embedding -> (LLM = ChatGPT - OpenAI) -> API Calls are Costly
   ↓
Vector Store / DB (FAISS : similarity search)
   ↓
Relevant chunks
   ↓
Build context
   ↓
Context + Question
   ↓
System Prompt -> (LLM = ChatGPT - OpenAI) -> API Calls are Costly
   ↓
Grounded answer
