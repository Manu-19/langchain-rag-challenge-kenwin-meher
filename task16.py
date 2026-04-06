# ─────────────────────────────────────────────────────────────
# TASK 16 — Conversational RAG with Chat History
# ─────────────────────────────────────────────────────────────
"""
TASK 16: Conversational RAG
------------------------------
Build a RAG pipeline that is aware of conversation history.

Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]

HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage

  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()
def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] for a 2-turn RAG conversation."""
    docs = [Document(page_content=d) for d in documents]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    connection_string = "postgresql+psycopg://postgres:Pass%40123@localhost:5432/vectordb"

    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="conv_rag_collection",
        connection_string=connection_string
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chat_history = []

    docs1 = retriever.invoke("What is LangChain?")
    context1 = format_docs(docs1)

    prompt1 = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )

    answer1 = (prompt1 | llm).invoke({
        "context": context1,
        "question": "What is LangChain?"
    }).content

    chat_history.append(HumanMessage(content="What is LangChain?"))
    chat_history.append(AIMessage(content=answer1))

    rephrase_prompt = ChatPromptTemplate.from_template(
        "Given the conversation:\n{history}\nRewrite the question into a standalone question:\n{question}"
    )

    standalone_q = (rephrase_prompt | llm).invoke({
        "history": "\n".join([m.content for m in chat_history]),
        "question": "What version introduced LCEL?"
    }).content

    docs2 = retriever.invoke(standalone_q)
    context2 = format_docs(docs2)

    prompt2 = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )

    answer2 = (prompt2 | llm).invoke({
        "context": context2,
        "question": standalone_q
    }).content

    return [answer1, answer2]


RAG_DOCUMENTS = [
    "LangChain is a framework for building applications powered by LLMs.",
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "RAG improves LLM accuracy by retrieving relevant documents.",
]

answers = conversational_rag(RAG_DOCUMENTS)

print("Turn 1 Answer:", answers[0])
print("Turn 2 Answer:", answers[1])
