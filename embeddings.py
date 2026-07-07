import os
import hashlib
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from config import (
    DATA_FOLDER,
    VECTOR_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO)


class EmbeddingManager:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    ####################################################################
    # Load all supported documents
    ####################################################################

    def load_documents(self):

        documents = []

        for file in os.listdir(DATA_FOLDER):

            path = os.path.join(DATA_FOLDER, file)

            if file.endswith(".pdf"):

                loader = PyPDFLoader(path)

            elif file.endswith(".html"):

                loader = UnstructuredHTMLLoader(path)

            elif file.endswith(".md"):

                loader = UnstructuredMarkdownLoader(path)

            else:
                continue

            documents.extend(loader.load())

        logging.info(f"Loaded {len(documents)} pages.")

        return documents

    ####################################################################
    # Split into chunks
    ####################################################################

    def split_documents(self, documents):

        chunks = self.splitter.split_documents(documents)

        logging.info(f"Generated {len(chunks)} chunks.")

        return chunks

    ####################################################################
    # Generate hash for idempotent ingestion
    ####################################################################

    def generate_chunk_hash(self, text):

        return hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

    ####################################################################
    # Add metadata
    ####################################################################

    def prepare_chunks(self, chunks):

        processed = []

        seen = set()

        for chunk in chunks:

            chunk_hash = self.generate_chunk_hash(
                chunk.page_content
            )

            if chunk_hash in seen:
                continue

            seen.add(chunk_hash)

            chunk.metadata["chunk_id"] = chunk_hash

            processed.append(chunk)

        logging.info(
            f"{len(processed)} unique chunks after duplicate removal."
        )

        return processed

    ####################################################################
    # Build FAISS index
    ####################################################################

    def build_vector_store(self):

        docs = self.load_documents()

        chunks = self.split_documents(docs)

        chunks = self.prepare_chunks(chunks)

        vector_db = FAISS.from_documents(
            chunks,
            self.embedding_model
        )

        vector_db.save_local(VECTOR_DB_PATH)

        logging.info(
            f"Vector database saved to '{VECTOR_DB_PATH}'."
        )

    ####################################################################
    # Load Existing Index
    ####################################################################

    def load_vector_store(self):

        return FAISS.load_local(
            VECTOR_DB_PATH,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )


####################################################################
# Run
####################################################################

if __name__ == "__main__":

    manager = EmbeddingManager()

    manager.build_vector_store()

    print("Vector database created successfully.")