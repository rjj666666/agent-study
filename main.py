from graph.agent_graph import build_graph
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    graph = build_graph()
    print("RAG Agent启动，输入q退出")
    while True:
        user_input = input("\n请输入问题：")
        if user_input == "q":
            break
        res = graph.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        print("\n回答：", res["messages"][-1].content)
