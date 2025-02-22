from typing import TypedDict, Annotated
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph
from tools.pubmed import search_articles, fetch_article_abstract
from dotenv import load_dotenv
from typing import List, Dict, Any
load_dotenv()


class PubMedAgent:

    def __init__(self, system_prompt: str, chat_model, tools: List[BaseTool]):
        self.system_prompt = system_prompt
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.chat_model = chat_model
        self.chain = self.prompt_template | self.chat_model
        self.tools = tools
        self.graph = self._build_graph()

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def query_gen_node(self, state: State):
        return {"messages": [self.chain.invoke(state["messages"])]}

    def _build_graph(self):
        graph_builder = StateGraph(self.State)
        checkpointer = MemorySaver()

        graph_builder.add_node("query_gen", self.query_gen_node)
        query_gen_tools_node = ToolNode(tools=self.tools)
        graph_builder.add_node("query_gen_tools", query_gen_tools_node)

        graph_builder.add_conditional_edges(
            "query_gen",
            tools_condition,
            {"tools": "query_gen_tools", END: END},
        )

        graph_builder.add_edge("query_gen_tools", "query_gen")
        graph_builder.set_entry_point("query_gen")
        graph = graph_builder.compile(checkpointer=checkpointer)
        return graph

    def process_event(self, event: Dict[str, Any]):
        if 'query_gen' in event:
            messages = event['query_gen']['messages']
            for message in messages:
                print(message.content)
        elif 'query_gen_tools' in event:
            messages = event['query_gen_tools']['messages']
            for message in messages:
                print(message.content)

    def run_query(self, query_text: str) -> str:
        for event in self.graph.stream({"messages": [("user", query_text)]},
                                      config={"configurable": {"thread_id": 12}}):
            self.process_event(event=event)

    def interactive_agent(self):
        print("\nWelcome to the Medial Research Assistant.")

        while True:
            try:
                query = input("\nWhat would you like to know? ")
                if query.lower() in ['exit', 'quit']:
                    print("\nThank you for using the Medical Research Assistant!")
                    break

                self.run_query(query)

            except KeyboardInterrupt:
                print("\nThank you for using the Medical Research Assistant!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("Please try again with a different query.")


if __name__ == "__main__":
    pubmed_research_system = """
    You are an AI medical research assistant. Given a research question, you will follow a structured approach to find relevant articles on PubMed.

    To accomplish your tasks, follow this workflow:

    1.  **[Thought]:** 
    - Start by outlining your reasoning process. What are the key steps needed to address the query? What information do you currently lack? This section is for your internal planning.

    2.  **[Analysis]:** 
    - Based on your thought process, determine which tool(s) are most appropriate to gather the necessary information. Clearly state why you've chosen these tool(s).

    3.  **[Action]:** 
    - Specify the tool to use and the corresponding input. This must be a tool name, followed by valid JSON containing the input parameters for the tool.

    4.  **[Verification]:** 
    - After receiving the tool's output, carefully assess its relevance and accuracy. Does the output address the information you sought in the 'Analysis' step? If not, adjust your approach and select a different tool or modify the input.

    5.  **[Final Output]:** 
    - Once you have gathered all the necessary information and verified its accuracy, provide a well-structured and comprehensive response.
    """

    pubmed_toolkit = [search_articles, fetch_article_abstract]
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0
    ).bind_tools(
        tools=pubmed_toolkit
    )

    pubmed_agent = PubMedAgent(system_prompt=pubmed_research_system, llm=llm, tools=pubmed_toolkit)
    pubmed_agent.interactive_agent()
