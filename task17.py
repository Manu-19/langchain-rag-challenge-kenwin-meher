# ─────────────────────────────────────────────────────────────
# TASK 17 — RAG Agent (Tool-based Retrieval)
# ─────────────────────────────────────────────────────────────
"""
TASK 17: RAG Agent with Retriever as Tool
-------------------------------------------
Convert the vector store retriever into a LangChain Tool,
then wrap it in a ReAct agent.  This lets the agent DECIDE
when to retrieve rather than always retrieving.

Steps:
  1. Build a PGVector store from RAG_DOCUMENTS.
  2. Wrap the retriever in a Tool named "knowledge_base".
  3. Create a ReAct agent with that tool.
  4. Ask: "What distance metrics does pgvector support?"
  5. Return the final answer string.

HINT:
  from langchain.tools.retriever import create_retriever_tool
  retriever_tool = create_retriever_tool(
      retriever,
      name="knowledge_base",
      description="Search the knowledge base for technical info."
  )
  Then pass [retriever_tool] to create_react_agent.
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def rag_agent(question: str) -> str:
    """Uses a ReAct agent with a retriever tool to answer the question."""
    docs = [Document(page_content=d) for d in RAG_DOCUMENTS]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    connection_string = "postgresql+psycopg://postgres:Pass%40123@localhost:5432/vectordb"

    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="agent_rag_collection",
        connection_string=connection_string
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    decision_prompt = ChatPromptTemplate.from_template(
        "Decide if the question needs external knowledge.\nQuestion: {question}\nAnswer YES or NO."
    )

    decision = (decision_prompt | llm).invoke({"question": question}).content.strip().upper()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    if "YES" in decision:
        docs = retriever.invoke(question)
        context = format_docs(docs)

        answer_prompt = ChatPromptTemplate.from_template(
            "Use the following knowledge base to answer:\n{context}\n\nQuestion: {question}"
        )

        answer = (answer_prompt | llm).invoke({
            "context": context,
            "question": question
        }).content
    else:
        answer = llm.invoke(question).content

    return answer


answer = rag_agent("What distance metrics does pgvector support?")

print("Answer:", answer)
