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

# 전체 상태 그래프에서 공유하는 상태
class GameState(TypedDict):
    # 플레이어 상태
    health: int             # 체력
    sanity: int             # 정신력
    type: str               # 플레이어 타입
    money: int              # 소지금
    current_location: str   # 현재 위치
    mood: str               # 현재 기분
    items: str              # 소지품
    player_lore: str        # 플레이어 로어
    causeOfDeath: str       # 플레이어 사망 원인
    choices: str            # 플레이어의 선택지

    # 장소 상태
    location_info: str

    # 요괴 상태
    monster_info: str

def generate_init_info():
    with open("location_info.json", "r", encoding="utf-8") as f:
        location_info_dict = json.load(f)

    with open("monster.json", "r", encoding="utf-8") as f:
        monster_info_dict = json.load(f)

    random.shuffle(monster_info_dict["요괴"])

    init_player_location = random.choice(monster_info_dict['요괴'])['name']

    for i in range(len(monster_info_dict["요괴"])):
        location_info_dict["graph"]['nodes'][i]['거주 요괴'] = monster_info_dict['요괴'][i]['name']

    location_info_json = json.dumps(
        location_info_dict,
        ensure_ascii=False,
        indent=2
    )

    return {"init_player_location": init_player_location, "init_location_info": location_info_json}

print(generate_init_info()['init_location_info'])