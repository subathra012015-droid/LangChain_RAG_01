# -------------------------------------------------------
# LangChain RAG Application - Backend Dockerfile
# -------------------------------------------------------

# 1. Use Python 3.11 as the base image
FROM python:3.11-slim

# 2. Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# 3. Display Python output immediately
ENV PYTHONUNBUFFERED=1

# 4. Set working directory inside container
WORKDIR /app

# 5. Copy requirements first
COPY requirements.txt .

# 6. Upgrade pip
RUN pip install --upgrade pip

# 7. Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy project files into container
COPY . .

# 9. FastAPI port
EXPOSE 8000

# 10. Start FastAPI application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]