"""
TASK 18: LangSmith Tracing
-----------------------------
Instrument a simple LCEL chain so every invocation is
traced in LangSmith.  Your function should:
  1. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT.
  2. Build the same basic LCEL chain from Task 1.
  3. Add run_name and tags to the invocation config.
  4. Return the response AND the run_id of the trace.

Expected return:
  {"answer": str, "run_id": str}

HINT:
  from langchain_core.tracers.context import collect_runs

  with collect_runs() as cb:
      result = chain.invoke(
          {"topic": topic},
          config={"run_name": "task18_trace", "tags": ["challenge"]}
      )
  run_id = str(cb.traced_runs[0].id)
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.context import collect_runs
from dotenv import load_dotenv

load_dotenv()

def traced_chain(topic: str) -> dict:
    """Runs a chain with LangSmith tracing. Returns answer and run_id."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "task18_project"

    prompt = ChatPromptTemplate.from_template(
        "Explain the following topic in 2-3 lines:\n{topic}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | llm | StrOutputParser()

    with collect_runs() as cb:
        result = chain.invoke(
            {"topic": topic},
            config={"run_name": "task18_trace", "tags": ["challenge"]}
        )

    run_id = str(cb.traced_runs[0].id)

    return {
        "answer": result,
        "run_id": run_id
    }


output = traced_chain("LangChain")

print("Answer:", output["answer"])
print("Run ID:", output["run_id"])

   
    
