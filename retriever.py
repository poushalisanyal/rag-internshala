import time
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from embeddings import EmbeddingManager

from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    TOP_K,
)

logging.basicConfig(level=logging.INFO)


class Retriever:

    def __init__(self):

        # Load FAISS Vector Database
        self.vector_db = EmbeddingManager().load_vector_store()

        # Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

        self.prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant.

Answer the question using ONLY the information provided in the context.

Provide a clear and slightly detailed explanation.
Explain important concepts, definitions, and key points.
Keep the answer concise but informative (around 1-3 paragraphs).

If the answer is not present in the context, reply exactly:

"I couldn't find relevant information in the provided documents."

Do not make up facts.

Context:
{context}

Question:
{question}

Answer:
""",
)


    ###########################################################
    # Retrieve Documents
    ###########################################################

    def retrieve(self, question, k=TOP_K, metadata_filter=None):

        start_time = time.time()

        if metadata_filter:

            docs = self.vector_db.similarity_search(
                query=question,
                k=k,
                filter=metadata_filter,
            )

        else:

            docs = self.vector_db.similarity_search(
                query=question,
                k=k,
            )


        # -----------------------------------
        # Remove duplicate chunks
        # -----------------------------------

        unique_docs = []
        seen = set()

        for doc in docs:

            content = doc.page_content.strip()

            if content not in seen:
                unique_docs.append(doc)
                seen.add(content)

        docs = unique_docs


        latency = time.time() - start_time

        logging.info(f"Retrieval Latency: {latency:.3f} sec")
        logging.info(f"Retrieved Chunks: {len(docs)}")


        return docs, latency



    ###########################################################
    # Build Context
    ###########################################################

    def build_context(self, docs):

        return "\n\n".join(
            doc.page_content for doc in docs
        )



    ###########################################################
    # Ask Question
    ###########################################################

    def ask(self, question, k=TOP_K):

        docs, latency = self.retrieve(question, k)


        if len(docs) == 0:

            return {
                "answer": "I couldn't find relevant information in the provided documents.",
                "sources": [],
                "latency": latency,
                "token_usage": "Not Available",
            }


        context = self.build_context(docs)


        prompt = self.prompt.format(
            context=context,
            question=question,
        )


        response = self.llm.invoke(prompt)


        metadata = getattr(response, "response_metadata", {})

        token_usage = metadata.get(
            "token_usage",
            "Not Available"
        )


        logging.info(f"Token Usage: {token_usage}")


        sources = []


        for doc in docs:

            sources.append(
                {
                    "source": doc.metadata.get(
                        "source",
                        "Unknown"
                    ),
                    "chunk_id": doc.metadata.get(
                        "chunk_id"
                    ),
                }
            )


        return {

            "question": question,

            "answer": response.content,

            "sources": sources,

            "latency": latency,

            "token_usage": token_usage,

        }