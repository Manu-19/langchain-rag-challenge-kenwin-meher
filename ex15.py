# ─────────────────────────────────────────────────────────────
# TASK 15 — RAG with Source Attribution
# ─────────────────────────────────────────────────────────────
"""
TASK 15: RAG with Source Attribution
---------------------------------------
Extend the RAG pipeline to also return the source documents
used to generate the answer.  Return a dict:
  {
    "answer" : str,
    "sources": [{"content": str, "score": float}, ...]
  }

HINT:
  Use RunnableParallel to run retrieval and generation
  in parallel, or retrieve docs first and pass them to both
  the formatter and the chain:

  from langchain_core.runnables import RunnableParallel, RunnablePassthrough

  retrieval_chain = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
  )
  # Then use the context in both the answer chain and as sources.
"""


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def rag_with_sources(documents: list, question: str) -> dict:
    """Returns the answer AND the source documents used."""
    docs = [Document(page_content=d) for d in documents]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    connection_string = "postgresql+psycopg://postgres:Pass%40123@localhost:5432/vectordb"

    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_sources_collection",
        connection_string=connection_string
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retrieval_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    answer_chain = (
        {
            "context": lambda x: format_docs(x["context"]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = retrieval_chain.invoke(question)
    answer = answer_chain.invoke(result)

    sources = [{"content": doc.page_content, "score": 0.0} for doc in result["context"]]

    return {
        "answer": answer,
        "sources": sources
    }


RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

result = rag_with_sources(RAG_DOCUMENTS, "What is RAG and why is it useful?")

print("Answer:", result["answer"])
print("Sources:")
for i, src in enumerate(result["sources"], 1):
    print(f"{i}.", src["content"])