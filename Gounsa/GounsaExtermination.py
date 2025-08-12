from typing import TypedDict
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

# 환경 변수 로드
load_dotenv()

# llm 모델 생성
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=.4, streaming=False)

