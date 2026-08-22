# ==========================================================
# LangChain_RAG_01
# Dockerfile for a01_app_FAISS
# ==========================================================

# 1. Use Python 3.11
FROM python:3.11-slim


# 2. Python configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# 3. Working directory inside Docker container
WORKDIR /app


# 4. Copy requirements first
COPY requirements.txt .


# 5. Upgrade pip
RUN python -m pip install --upgrade pip


# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# 7. Copy project files into Docker image
COPY . .


# 8. Start a01 FAISS application
CMD ["python", "a01_app_FAISS/a01_app_FAISS.py"]