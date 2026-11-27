FROM python:3.12-slim

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies first for Docker layer caching
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and raw datasheet PDFs
COPY --chown=user . .

# Pre-build ChromaDB + BM25 index during image build so runtime startup is fast
RUN GROQ_API_KEY=build-placeholder python -m src.rag.ingestion

EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]