import re

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from typing import List, Union, TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph

from tools.pubmed import search_articles, fetch_summary
from dotenv import load_dotenv
load_dotenv()

import requests

import requests
from typing import List, Dict, Any


# class PubMedTool(AtomicTool):
#     def __init__(self, api_key):
#         super().__init__()
#         self.api_key = api_key
#         self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
#
#     def search_articles(self, query: str) -> List[str]:
#         params = {
#             "db": "pubmed",
#             "term": query,
#             "retmode": "json",
#             "retmax": 5,
#             "api_key": self.api_key
#         }
#         response = requests.get(f"{self.base_url}esearch.fcgi", params=params)
#         data = response.json()
#         return data["esearchresult"]["idlist"]
#
#     def fetch_summary(self, pmid: str) -> Dict[str, Any]:
#         params = {
#             "db": "pubmed",
#             "id": pmid,
#             "retmode": "json",
#             "api_key": self.api_key
#         }
#         response = requests.get(f"{self.base_url}esummary.fcgi", params=params)
#         data = response.json()
#         return data["result"][pmid]
#
#     def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         query = input_data.get("query", "")
#         return self.react_agent(query)
#
#     def react_agent(self, query: str) -> Dict[str, Any]:
#         thought_process = []
#
#         # Think
#         thought_process.append(f"Thought: I need to search for articles about '{query}' in PubMed.")
#
#         # Act
#         thought_process.append("Action: Searching PubMed for articles.")
#         pmids = self.search_articles(query)
#
#         # Observe
#         thought_process.append(f"Observation: Found {len(pmids)} articles.")
#
#         results = []
#         for pmid in pmids:
#             # Think
#             thought_process.append(f"Thought: I should fetch the summary for article with PMID {pmid}.")
#
#             # Act
#             thought_process.append(f"Action: Fetching summary for PMID {pmid}.")
#             summary = self.fetch_summary(pmid)
#
#             # Observe
#             thought_process.append(f"Observation: Retrieved summary for article titled '{summary.get('title')}'.")
#
#             results.append({
#                 "title": summary.get("title"),
#                 "abstract": summary.get("abstract", "No abstract available."),
#                 "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#             })
#
#         # Final Thought
#         thought_process.append(
#             f"Thought: I have collected information on {len(results)} articles related to '{query}'.")
#
#         return {
#             "results": results,
#             "thought_process": thought_process
#         }
#
#
# # Example usage
# pubmed_tool = PubMedTool(api_key="YOUR_API_KEY_HERE")
# query = "show me the latest studies on hemophilia"
# response = pubmed_tool.execute({"query": query})
#
# print("Results:")
# for result in response['results']:
#     print(f"Title: {result['title']}")
#     print(f"Abstract: {result['abstract'][:100]}...")  # Truncated for brevity
#     print(f"Link: {result['link']}")
#     print()
#
# print("\nThought Process:")
# for thought in response['thought_process']:
#     print(thought)



pubmed_research_system = """
You are an AI medical research assistant. Given a research question, you will follow a structured approach to find relevant articles on PubMed.

**Reasoning Phase** (<reasoning> tag)
<reasoning>
I will **always** include this section before writing a query. Here, I will:
- Explain what information you need and why.
- Describe your expected outcome.
- Identify potential challenges.
- Justify your approach.
</reasoning>

**Analysis Phase** (<analysis> tag)
<analysis>
- Define PubMed search terms.
- Specify filters (e.g., publication date, study type).
- Outline steps to retrieve and summarize articles.
</analysis>


**Verification Phase** (<final_check> tag)
<final_check>
- Verify that the reasoning, analysis, and query align with the initial question.
- Confirm that all articles are summarized and links are provided.
</final_check>

**Final Output** (<final_output> tag)
<final_output>
- Return final output and finish processing
</final_output>

Important Rules:
1. Verify each phase before proceeding.
2. You may only call pubmed api a MAXIMUM of 3 times.
"""



pubmed_research_system = """
You are an AI medical research assistant. Given a research question, you will follow a structured approach to find relevant articles on PubMed.

"""

# **Query Phase** (<query> tag)
# <query>
# - Execute PubMed search using the PubMedTool.
# - Summarize each article.
# - Provide article links.
# </query>

pubmed_research_prompt = ChatPromptTemplate.from_messages([
    ("system", pubmed_research_system),
    MessagesPlaceholder(variable_name="messages")
])

pubmed_toolkit = [search_articles, fetch_summary]

query_gen_model = pubmed_research_prompt | ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
).bind_tools(
    tools=pubmed_toolkit
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def query_gen_node(state: State):
    return {"messages": [query_gen_model.invoke(state["messages"])]}


checkpointer = MemorySaver()

graph_builder.add_node("query_gen", query_gen_node)
query_gen_tools_node = ToolNode(tools=pubmed_toolkit)
graph_builder.add_node("query_gen_tools", query_gen_tools_node)

graph_builder.add_conditional_edges(
    "query_gen",
    tools_condition,
    {"tools": "query_gen_tools", END: END},
)

graph_builder.add_edge("query_gen_tools", "query_gen")
graph_builder.set_entry_point("query_gen")
graph = graph_builder.compile(checkpointer=checkpointer)


# def format_section(title: str, content: str) -> str:
#     if not content:
#         return ""
#     return f"\n{content}\n"
#
#
# def extract_section(text: str, section: str) -> str:
#     pattern = f"<{section}>(.*?)</{section}>"
#     match = re.search(pattern, text, re.DOTALL)
#     return match.group(1).strip() if match else ""
#
#
# def process_event(event: Dict[str, Any]) -> Optional[str]:
#     if 'query_gen' in event:
#         messages = event['query_gen']['messages']
#         for message in messages:
#             content = message.content if hasattr(message, 'content') else ""
#
#             reasoning = extract_section(content, "reasoning")
#             if reasoning:
#                 print(format_section("", reasoning))
#
#             analysis = extract_section(content, "analysis")
#             if analysis:
#                 print(format_section("", analysis))
#
#             query = extract_section(content, "query")
#             if query:
#                 print(format_section("", query))
#
#             final_check = extract_section(content, "final_check")
#             if final_check:
#                 print(format_section("", final_check))
#
#             if hasattr(message, 'tool_calls'):
#                 for tool_call in message.tool_calls:
#                     tool_name = tool_call['name']
#                     # if tool_name == 'sql_db_query':
#                     #     return tool_call['args']['query']
#
#             final_output = extract_section(content, "final_output")
#             if final_output:
#                 response = f"\nHere's the analysis: \n"
#                 for i, result in enumerate(final_output):
#                     response += f"\nArticle {i + 1}:\n"
#                     response += f"Title: {result['title']}\n"
#                     response += f"Abstract: {result['abstract']}\n"
#                     response += f"Link: {result['link']}\n"
#                     response += "\n---\n"
#                 return format_section("", query)
#     return None

# def run_query(query_text: str):
#     print(f"\nAnalyzing your question: {query_text}")
#     final_output = None
#
#     for event in graph.stream({"messages": [("user", query_text)]},
#                               config={"configurable": {"thread_id": 12}}):
#         output = process_event(event)
#         if output:
#             final_output = final_output
#
#     if final_output:
#         print(
#             "\nBased on my analysis, here are the articles and details:")
#         print(f"\n{final_output}")
#         return final_output

def run_query(query_text: str) -> str:
    for event in graph.stream({"messages": [("user", query_text)]},
                                  config={"configurable": {"thread_id": 12}}):
        if 'query_gen' in event:
            messages = event['query_gen']['messages']
            for message in messages:
                print(message)

def interactive_agent():
    print("\nWelcome to the Medial Research Assistant.")

    while True:
        try:
            query = input("\nWhat would you like to know? ")
            if query.lower() in ['exit', 'quit']:
                print("\nThank you for using the Medical Research Assistant!")
                break

            run_query(query)

        except KeyboardInterrupt:
            print("\nThank you for using the Medical Research Assistant!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    interactive_agent()
