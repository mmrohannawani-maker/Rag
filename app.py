# =========================================================
# IMPORTS
# =========================================================
# Core Python utilities and typing
from typing import TypedDict, Any
import os
import numpy as np
import pandas as pd

# Database connectivity
from sqlalchemy import create_engine

# LangGraph for workflow orchestration
from langgraph.graph import StateGraph

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# RAG components: embeddings + vector search
from sentence_transformers import SentenceTransformer
import faiss


# =========================================================
# GEMINI API KEY
# =========================================================
# ⚠️ ADD YOUR OWN GEMINI API KEY HERE
os.environ["GOOGLE_API_KEY"] = "AIzaSyBt_Vn-JhnDmq3TPVbeZxrOitr9weeCQTM"


# =========================================================
# DATABASE CONNECTION (POSTGRESQL)
# =========================================================
# Connection details for PostgreSQL
DB_USER = "postgres"
DB_PASSWORD = "Rohan18"
DB_NAME = "Sales"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# =========================================================
# INITIALIZE GEMINI LLM
# =========================================================
# Temperature 0 ensures stable and repeatable outputs
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0
)


# =========================================================
# RAG DOCUMENT LOADING (FROM FILES)
# =========================================================
# This function loads all text files from the docs folder
# 👉 Add / edit documents only inside the docs/ directory
def load_docs_from_folder(path="docs"):
    documents = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents


# =========================================================
# RAG SETUP (EMBEDDINGS + FAISS)
# =========================================================
print("🔹 Initializing RAG")

# Load documents from docs folder
rag_documents = load_docs_from_folder()

# Convert text to embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(rag_documents)

# Create FAISS index
dimension = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(doc_embeddings))

print(f"✅ Loaded {len(rag_documents)} documents into RAG")


# =========================================================
# LANGGRAPH STATE
# =========================================================
# This state object flows through the entire graph
class GraphState(TypedDict):
    question: str
    route: str               # sql / docs / both
    rag_context: str
    sql_query: str
    sql_result: Any
    final_answer: str


# =========================================================
# NODE 1: DECIDE ROUTE (SQL / DOCS / BOTH)
# =========================================================
def decide_route(state: GraphState):
    print("🤔 Decide")

    prompt = f"""
Decide how the question should be answered.

Options:
- sql : needs database query
- docs : can be answered from documents
- both : needs database + documents

Question:
{state["question"]}

Return ONLY one word: sql, docs, or both
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"route": response.content.strip().lower()}


# =========================================================
# NODE 2: RETRIEVE DOCUMENT CONTEXT (RAG)
# =========================================================
def retrieve_docs(state: GraphState):
    print("📚 Docs")

    query_embedding = embedding_model.encode([state["question"]])
    _, indices = faiss_index.search(np.array(query_embedding), k=3)

    retrieved_docs = [rag_documents[i] for i in indices[0]]
    return {"rag_context": "\n---\n".join(retrieved_docs)}


# =========================================================
# NODE 3: GENERATE SQL
# =========================================================
def generate_sql(state: GraphState):
    print("🧠 SQL")

    schema = """
Table: sales
Columns:
sale_id, order_date, customer_id, customer_name,
region, product_id, product_name, category,
quantity, unit_price, discount, total_amount

Table: customers
Columns:
customer_id, customer_name, email,
phone, created_at

Table: products
Columns:
product_id, product_name, category,
base_price, created_at

Table: regions
Columns:
region_id, region_name, country

RELATIONSHIPS:
- sales.customer_id → customers.customer_id
- sales.product_id → products.product_id
- sales.region → regions.region_name
"""

    prompt = f"""
You are a PostgreSQL SQL expert whose output is executed directly against a real database.

YOUR TASK:
Generate ONE valid PostgreSQL SELECT query that answers the user question.

ABSOLUTE OUTPUT RULES (MUST FOLLOW):
1. Output MUST start with SELECT.
2. Output MUST contain ONLY SQL (no explanations, no markdown, no comments).
3. Use ONLY SELECT queries.
4. DO NOT use CTEs (NO WITH keyword).
5. DO NOT invent columns or tables.
6. DO NOT assume columns that are not explicitly listed in the schema.
7. The query MUST run successfully on PostgreSQL.

CRITICAL COLUMN-SAFETY RULES:
- Use ONLY column names EXACTLY as defined in the schema below.
- Column names are CASE-SENSITIVE as written.
- If a column is not listed, it DOES NOT EXIST.
- Always qualify columns using table aliases (e.g., s.total_amount).
- NEVER reference a column from the wrong table.
- NEVER guess foreign keys — use ONLY the defined relationships.

SQL STYLE STANDARD (FOLLOW STRICTLY):
- Always use table aliases:
  sales → s
  customers → c
  products → p
  regions → r
- Always use explicit JOINs (no implicit joins).
- Use consistent alias names everywhere.
- Aggregate functions must match GROUP BY columns.
- For “top / highest / most per group”:
  → Use a subquery with ROW_NUMBER().
  → Filter using WHERE rn = 1 in the outer query.
- Avoid unnecessary columns in SELECT.


DATABASE SCHEMA:
{schema}

HOW TO SOLVE COMPLEX QUESTIONS (FOLLOW INTERNALLY):
1. Identify required tables and joins.
2. Aggregate values where needed.
3. If the question asks for “highest / top / most per group”:
   - Use a subquery with a window function.
   - Filter results using the window function in an outer SELECT.
4. Ensure the final result answers ALL parts of the question.

MANDATORY INTERNAL VALIDATION (DO BEFORE OUTPUT):
- Check every column exists in its table.
- Check every JOIN uses a valid relationship.
- Check GROUP BY includes all non-aggregated SELECT columns.
- Check no forbidden keywords are used.
- If unsure, simplify the query instead of guessing.

CRITICAL ALIAS RULE:
- Any column referenced in an outer query MUST exist in the inner query SELECT list.
- Semantic names like "top_category" or "most_contributing_category" MUST be explicitly created using AS.
- NEVER reference a column alias that was not explicitly defined.

most important rule:only go through the schema given

FINAL SQL (START WITH SELECT, NO ERRORS):

    User Question:
    {state['question']}

    Return ONLY SQL.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"sql_query": response.content.strip()}


# =========================================================
# NODE 4: EXECUTE SQL
# =========================================================
def execute_sql(state: GraphState):
    print("⚙️ Run")

    with engine.connect() as conn:
        df = pd.read_sql(state["sql_query"], conn)

    return {"sql_result": df}


# =========================================================
# NODE 5: COMBINE SQL + DOCS INTO FINAL ANSWER
# =========================================================
def combine_evidence(state: GraphState):
    print("🔗 Combine")

    sql_text = ""
    if state.get("sql_result") is not None:
        sql_text = state["sql_result"].head(10).to_string()

    prompt = f"""
Answer the question using the available information.

DOCUMENT CONTEXT:
{state.get("rag_context", "")}

SQL RESULT:
{sql_text}

Question:
{state["question"]}

Answer clearly in simple English.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content.strip()}


# =========================================================
# ROUTING LOGIC
# =========================================================
def route_logic(state: GraphState):
    if state["route"] == "sql":
        return "generate_sql"
    if state["route"] == "docs":
        return "retrieve_docs"
    return "retrieve_docs"   # both starts with docs


# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================
workflow = StateGraph(GraphState)

workflow.add_node("decide_route", decide_route)
workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("combine_evidence", combine_evidence)

workflow.set_entry_point("decide_route")

workflow.add_conditional_edges(
    "decide_route",
    route_logic,
    {
        "generate_sql": "generate_sql",
        "retrieve_docs": "retrieve_docs"
    }
)

workflow.add_edge("retrieve_docs", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "combine_evidence")
workflow.add_edge("retrieve_docs", "combine_evidence")

workflow.set_finish_point("combine_evidence")

app = workflow.compile()


# =========================================================
# RUN CHATBOT
# =========================================================
if __name__ == "__main__":
    print("\n🤖 SQL + RAG Chatbot Ready (type 'exit')\n")

    while True:
        user_input = input("🧑 You: ")

        if user_input.lower() == "exit":
            break

        result = app.invoke({"question": user_input})

        print("\n✅ Final Answer:")
        print(result["final_answer"])
        print("\n" + "=" * 60 + "\n")
