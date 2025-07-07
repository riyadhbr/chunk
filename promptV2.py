import os
import json
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI
from sklearn.preprocessing import normalize
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StaticRAGSystem:
    collection_all = "uuid_all"
    memory_enabled = True
    max_memory_msgs = 5
    chat_history = []
    react_graph = None
    lang = "العربية"
    doc_name_pdf="products"
    doc_name_txt="general informations"
    doc_name_Tabular="2_Table_des_prix_JUPITER"
    user_prompt = (
        "You're a sales assistant at jupyter. "
        "Your business is to provide information about jupyter and about the products it offers."
    )

    qdrant = AsyncQdrantClient(url="http://localhost:6333")
    openai_client_key = os.environ.get("OPENAI_API_KEY") or ""
    openai_client = AsyncOpenAI(api_key=openai_client_key)
    llm = ChatOpenAI(
        openai_api_key=openai_client_key,
        model_name="gpt-4o-mini-2024-07-18",
        temperature=0
    )

    @staticmethod
    async def fetch_data(limit: int = 100) -> Dict[str, Any]:
        summary_result = await StaticRAGSystem.qdrant.scroll(
            collection_name=StaticRAGSystem.collection_all,
            limit=limit,
            with_payload=True,
            scroll_filter=Filter(
                must=[FieldCondition(key="title", match=MatchValue(value="Summary"))],
                should=[
                    FieldCondition(key="doc_name", match=MatchValue(value=StaticRAGSystem.doc_name_pdf)),
                    FieldCondition(key="doc_name", match=MatchValue(value=StaticRAGSystem.doc_name_txt)),
                ]
            )
        )
        summaries = [point.payload for point in summary_result[0] if point.payload]

        product_result = await StaticRAGSystem.qdrant.scroll(
            collection_name=StaticRAGSystem.collection_all,
            limit=limit,
            with_payload=True,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_name", match=MatchValue(value=StaticRAGSystem.doc_name_Tabular))]
            )
        )
        payloads = [p.payload for p in product_result[0] if p.payload]
        all_keys = sorted({k for payload in payloads for k in payload.keys() if k != "doc_name"})

        return {"summaries": summaries, "product_keys": all_keys}

    @staticmethod
    async def create_system_message() -> SystemMessage:
        data = await StaticRAGSystem.fetch_data()
        summaries_text = ", ".join([
            (s.get("text") or s.get("content") or s.get("summary") or "")
            .replace("Topics:", "").strip()
            for s in data["summaries"]
        ])
        product_keys_text = ", ".join(data["product_keys"])

        return SystemMessage(
            content=(
                f"{StaticRAGSystem.user_prompt} Answer in {StaticRAGSystem.lang}, using the most common everyday expressions and colloquial phrases. "
                "Always keep the tone friendly, clear, and simple. "
                "In addition, you are a useful assistant tasked with using the following tools:\n"
                f"{{search_similar}} to search for the following internal summaries: {summaries_text}.\n"
                f"{{search_product_details}} To search for product or service information, such as {product_keys_text}.\n"
                "If you feel your answer is not convincing or complete, ask the user: ‘Would you like to speak with a human support agent for more help?’\n"
                "If the user replies with 'yes' or something similar, or if the user explicitly requests human support (e.g., 'human support', 'talk to a person'), "
                "call{{trigger_human_support}}."
            )
        )

    @staticmethod
    async def generate_embedding(text: str) -> List[float]:
        response = await StaticRAGSystem.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return normalize(np.array([response.data[0].embedding]))[0].tolist()

     

    @staticmethod
    async def search_similar(text: str) -> List[Dict[str, str]]:
        "search_similar"
        embedding = await StaticRAGSystem.generate_embedding(text)
        result = await StaticRAGSystem.qdrant.search(
            collection_name=StaticRAGSystem.collection_all,
            query_vector=embedding,
            limit=5,
            query_filter=Filter(
                should=[
                FieldCondition(key="doc_name", match=MatchValue(value=StaticRAGSystem.doc_name_pdf)),
                FieldCondition(key="doc_name", match=MatchValue(value=StaticRAGSystem.doc_name_txt))
            ]        )
        )
        return  [{'text': r.payload.get('text'), 'title': r.payload.get('title')} for r in result]

    @staticmethod
    async def search_product_details(text: str) -> List[Dict[str, Any]]:
        "search_product_details"
        embedding = await StaticRAGSystem.generate_embedding(text)
        result = await StaticRAGSystem.qdrant.search(
            collection_name=StaticRAGSystem.collection_all,
            query_vector=embedding,
            limit=5,
            query_filter=Filter(
            must=[FieldCondition(key="doc_name", match=MatchValue(value=StaticRAGSystem.doc_name_Tabular))]
        )
        )
        return [{k: v for k, v in hit.payload.items() if k != "doc_name"} for hit in result]

    @staticmethod
    def route_after_reasoner(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            first_call = last.tool_calls[0]
            if isinstance(first_call, dict) and first_call.get("name") == "trigger_human_support":
                return END
            return "tools"
        return END


    @staticmethod
    @tool
    async def trigger_human_support() -> str:
        "trigger_human_support"
        return 

    @staticmethod
    async def reasoner(state: MessagesState) -> Dict[str, Any]:
        tools = [
            StaticRAGSystem.search_similar,
            StaticRAGSystem.search_product_details,
            StaticRAGSystem.trigger_human_support
        ]
        llm_with_tools = StaticRAGSystem.llm.bind_tools(tools)
        result = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [result]}

    @staticmethod
    async def _create_react_graph_async() -> Any:
        builder = StateGraph(MessagesState)
        builder.add_node("reasoner", StaticRAGSystem.reasoner)
        builder.add_node("tools", ToolNode([
            StaticRAGSystem.search_similar,
            StaticRAGSystem.search_product_details,
            StaticRAGSystem.trigger_human_support
        ]))
        builder.add_edge(START, "reasoner")
        builder.add_edge("tools", "reasoner")
        builder.add_conditional_edges("reasoner",StaticRAGSystem.route_after_reasoner, {"tools": "tools", END: END})
        return builder.compile()

    @staticmethod
    async def init_graph():
        if StaticRAGSystem.react_graph is None:
            StaticRAGSystem.react_graph = await StaticRAGSystem._create_react_graph_async()
        if not hasattr(StaticRAGSystem, "sys_msg"):
            StaticRAGSystem.sys_msg = await StaticRAGSystem.create_system_message()

    @staticmethod
    async def converse(user_input: str) -> str:
        messages = [StaticRAGSystem.sys_msg]
        if StaticRAGSystem.memory_enabled and StaticRAGSystem.chat_history:
            messages.extend(StaticRAGSystem.chat_history)
        messages.append(HumanMessage(content=user_input))

        state = {'messages': messages}
        await StaticRAGSystem.init_graph()

        async for event in StaticRAGSystem.react_graph.astream(state, {"recursion_limit": 4}):
            if "reasoner" in event and event["reasoner"].get("messages"):
                messages.extend(event["reasoner"]["messages"])

        reply = messages[-1].content
        if StaticRAGSystem.memory_enabled:
            StaticRAGSystem.chat_history.append(HumanMessage(content=user_input))
            StaticRAGSystem.chat_history.append(messages[-1])
            if len(StaticRAGSystem.chat_history) > StaticRAGSystem.max_memory_msgs * 2:
                StaticRAGSystem.chat_history = StaticRAGSystem.chat_history[-(StaticRAGSystem.max_memory_msgs * 2):]

        return reply


if __name__ == "__main__":
    async def main():
        await StaticRAGSystem.init_graph()
        while True:
            user_input = input("You: ")
            response = await StaticRAGSystem.converse(user_input)
            logger.info(f"Response: {response}")
            

    asyncio.run(main())
