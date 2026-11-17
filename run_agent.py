from config.settings import (
    CHROMA_PERSIST_DIR,
    KNOWLEDGE_FILE,
)
from rag import init_knowledge
from graph import graph
from langchain_core.messages import HumanMessage

# 初始化RAG知识库（全局只加载一次）
init_knowledge(KNOWLEDGE_FILE, CHROMA_PERSIST_DIR)

if __name__ == "__main__":
    print("✅ Agent启动成功，输入 exit 退出")
    config = {"configurable": {"thread_id": "1"}}
    while True:
        query = input("\n请输入你的问题：")
        if query.strip().lower() in ["exit", "quit"]:
            print("程序退出")
            break
        inputs = {"messages": [HumanMessage(content=query)]}
        final_state = None
        for chunk in graph.stream(inputs, config=config):
            for node_name, value in chunk.items():
                final_state = value
        if final_state and "messages" in final_state:
            final_msg = final_state["messages"][-1]
            print("\n【最终回答】")
            print(final_msg.content)
