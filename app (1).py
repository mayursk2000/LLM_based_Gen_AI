import os
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.tools import Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ── API Keys ─────────────────────────────────────────────────────────────────
# When running via `streamlit run app.py`, read keys from environment or
# Streamlit secrets (.streamlit/secrets.toml).
os.environ.setdefault("OPENAI_API_KEY",  st.secrets.get("OPENAI_API_KEY", ""))
os.environ.setdefault("SERPAPI_API_KEY", st.secrets.get("SERPAPI_API_KEY", ""))

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = OpenAI(temperature=0)

# ── Tools ────────────────────────────────────────────────────────────────────
search_tool = load_tools(["serpapi"], llm=llm)[0]

def compare_items(query: str) -> str:
    try:
        parts = [p.strip() for p in query.split(",")]
        if len(parts) < 3:
            return "Error: please provide at least two items and a category."
        items, category = parts[:-1], parts[-1]
        template = """Compare the following {category} in detail.
Items: {items}
Provide strengths, weaknesses, and a recommendation.
Comparison:"""
        chain = LLMChain(llm=llm, prompt=PromptTemplate(
            input_variables=["items", "category"], template=template))
        return chain.invoke({"items": ", ".join(items), "category": category})["text"].strip()
    except Exception as e:
        return f"Error: {e}"

compare_tool = Tool(
    name="Compare",
    func=compare_items,
    description="Compare items in a category. Input: 'item1, item2, category'",
)

def analyze_results(query: str) -> str:
    template = """Analyze the following text. Highlight key takeaways and insights.
Text: {text}
Concise Analysis:"""
    chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["text"], template=template))
    return chain.invoke({"text": query})["text"].strip()

analyze_tool = Tool(
    name="Analyze",
    func=analyze_results,
    description="Summarize and extract insights from text.",
)

# ── Agent ────────────────────────────────────────────────────────────────────
agent = initialize_agent(
    tools=[search_tool, compare_tool, analyze_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("ReAct Agent")
st.write("Ask a complex question and let the agent reason through it step by step.")

query = st.text_input("Enter your query:", placeholder="e.g. Compare the top 3 smartphones of 2024")

if st.button("Submit"):
    if query:
        with st.spinner("Thinking..."):
            try:
                output = agent.invoke({"input": query})
                result = output["output"]
                steps = output.get("intermediate_steps", [])
            except Exception as e:
                result = f"Error: {e}"
                steps = []

        # Display final answer
        st.subheader("Answer")
        st.write(result)

        # Display reasoning trace
        if steps:
            with st.expander("Step-by-step reasoning", expanded=False):
                for idx, (action, observation) in enumerate(steps, 1):
                    st.markdown(f"**Step {idx}**")
                    st.markdown(f"- **Thought/Action:** {action.tool} [{action.tool_input}]")
                    st.markdown(f"- **Observation:** {observation[:500]}")
                    st.divider()
    else:
        st.warning("Please enter a query before submitting.")
