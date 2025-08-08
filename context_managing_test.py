import random
from typing import Annotated, TypedDict, List, Tuple, Union
import operator
from pydantic import BaseModel, Field
import asyncio
import json
from pathlib import Path
import random

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 환경 변수 로드
load_dotenv()

# llm 모델 생성
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0, streaming=False)

class GraphState(TypedDict):
    graph: str
    arrive: bool
    target_location: str

class Move(BaseModel):
    graph: str = Field(description="그래프를 표현하는 json형식의 텍스트")
    arrive: bool = Field(description="목표 위치에 도달했는지에 대한 정보")

graph_prompt = ChatPromptTemplate.from_template("""
당신은 그래프의 경로를 찾아주는 경로 탐색의 전문가입니다.
당신에게는 노드와 엣지에 대한 정보와 현재 위치를 가지는 json문자열이 제공됩니다.
목표 장소가 지정되면, 현재 위치로부터 목표 장소까지 이동시켜야 하며, 노드는 한 번에 인접한 노드로만 이동할 수 있습니다.
출력 형식은 입력된 json그래프의 형태를 유지해야만 하고, current_location만을 편집해야 합니다.
###
목표 장소: {target_location}
###
```json
{graph}
```
""")

graph_builder = StateGraph(GraphState)

graph_builder.add_node("이동", graph_prompt|llm.with_structured_output(Move))

graph_builder.add_edge(START, "이동")
graph_builder.add_conditional_edges("이동", lambda state: "이동" if not state["arrive"] else END, ["이동", END])

graph = graph_builder.compile()

# try:
#     png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
#     with open("test.png", "wb") as f:
#         f.write(png_bytes)
# except Exception:
#     pass

async def main():
    config = {"recursion_limit": 20}
    inputs = {"graph": """{
  "nodes": [
    { "id": 1,  "name": "Statue of Liberty" },
    { "id": 2,  "name": "Eiffel Tower" },
    { "id": 3,  "name": "Great Wall of China" },
    { "id": 4,  "name": "Sydney Opera House" },
    { "id": 5,  "name": "Machu Picchu" },
    { "id": 6,  "name": "Taj Mahal" },
    { "id": 7,  "name": "Christ the Redeemer" },
    { "id": 8,  "name": "Pyramids of Giza" },
    { "id": 9,  "name": "Mount Fuji" },
    { "id": 10, "name": "Niagara Falls" }
  ],
  "edges": [
    { "source": 1, "target": 4 },
    { "source": 1, "target": 7 },
    { "source": 2, "target": 3 },
    { "source": 2, "target": 5 },
    { "source": 2, "target": 9 },
    { "source": 3, "target": 6 },
    { "source": 3, "target": 8 },
    { "source": 4, "target": 5 },
    { "source": 4, "target": 10 },
    { "source": 5, "target": 8 },
    { "source": 6, "target": 7 },
    { "source": 7, "target": 9 },
    { "source": 8, "target": 10 }
  ],
  "current_location": "Taj Mahal"
}""", "target_location": "Machu Picchu"}

    async for event in graph.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print("=========================")
                print(v)
                print("=========================")

if __name__ == "__main__":
    asyncio.run(main())