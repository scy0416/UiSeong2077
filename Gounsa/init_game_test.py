from typing import List, TypedDict, Optional
import firebase_admin
from firebase_admin import credentials, firestore
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
import json
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0, streaming=False, cache=False)
llm_reasoning = ChatOpenAI(model="gpt-5-mini-2025-08-07", streaming=False, cache=False)

cred = credentials.Certificate("../uiseong2077-firebase-key.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

class History(BaseModel):
    type: str = Field(description="텍스트의 타입으로 나레이션이면 narration, 대사이면 dialogue입니다.")
    speaker: str = Field(description="type이 dialogue일 때, 대사를 말하는 주체입니다. narration일 때는 빈 문자열입니다.")
    text: str = Field(description="나레이션 또는 대사의 텍스트입니다.")

class Histories(BaseModel):
    generated_history: List[History] = Field(description="만들어지는 적절한 내용들")

class Supervising(BaseModel):
    edited_history: List[History] = Field(description="감독한 진행 내용의 최종 결과")
    supervising_done: bool = Field(description="감독이 끝나고서 추가 감독이 필요한지에 대한 여부입니다.")

class UserData(BaseModel):
    health: int = Field(description="플레이어의 체력입니다.")
    sanity: int = Field(description="플레이어의 정신력입니다.")
    purification: int = Field(description="고운사의 요괴 정화도 수준입니다.")
    current_location: str = Field(description="플레이어의 현재 위치입니다.")
    current_mood: str = Field(description="플레이어의 현재 기분입니다.")
    items: str = Field(description="아이템 목록으로 아이템은 컴마(,)로 구분됩니다.")
    choices: List[str] = Field(default_factory=list, description="사용자의 선택지입니다.")
    history: List[History] = Field(default_factory=list, description="게임의 진행 기록입니다.")
    map: str = Field(description="맵의 상태에 대한 텍스트로 json형식입니다.")
    monster: str = Field(description="요괴의 상태에 대한 텍스트로 json형식입니다.")
    player_lore: str = Field(description="플레이어의 기억입니다.")
    master_lore: str = Field(description="게임 전반에 걸친 기억입니다.")
    #supervising_done: bool = Field(description="감독이 끝나고서 추가 감독이 필요한지에 대한 여부입니다.")

class Choices(BaseModel):
    choices: List[str] = Field(description="사용자가 할 수 있는 행동들로 각각의 선택지는 성공했을 때의 이점과 실패했을 때의 벌점에 대한 내용을 포함합니다.")

class ChangedUserData(BaseModel):
    health: int = Field(description="플레이어의 체력입니다.")
    sanity: int = Field(description="플레이어의 정신력입니다.")
    purification: int = Field(description="고운사의 요괴 정화도 수준입니다.")
    current_location: str = Field(description="플레이어의 현재 위치입니다.")
    current_mood: str = Field(description="플레이어의 현재 기분입니다.")
    items: str = Field(description="아이템 목록으로 아이템은 컴마(,)로 구분됩니다.")
    map: str = Field(description="맵의 상태에 대한 텍스트로 json형식입니다.")
    monster: str = Field(description="요괴의 상태에 대한 텍스트로 json형식입니다.")
    player_lore: str = Field(description="플레이어의 기억입니다.")
    master_lore: str = Field(description="게임 전반에 걸친 기억입니다.")

class ChangedWorldData(BaseModel):
    #map: str = Field(description="맵의 상태에 대한 텍스트로 json형식입니다.")
    plan: str = Field(description="map에 변경을 가할 계획입니다.")
    map: str = Field(description="맵의 상태에 대한 텍스트로 json형식입니다. 전달 받은 모든 내용을 그대로 작성하세요.")
    #monster: str = Field(description="요괴의 상태에 대한 텍스트로 json형식입니다.")

class UserGraphData(UserData):
    total_history: List[History] = Field(description="전체 스토리")
    current_history: Optional[History] = None

class Map(BaseModel):
    map: str = Field(description="맵을 표현하는 json형태의 텍스트입니다.")

progress_graph_builder = StateGraph(UserData)
choices_generate_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("""
    당신은 텍스트 어드벤처 게임의 선택지 생성 담당자입니다.
    현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
    이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
    요괴들에 대한 정보는 json형식으로 제공됩니다.
    요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
    사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
    플레이어의 기억(player_lore)들이 있습니다.
    세계에 대한 정보로는 진행기록(history), 마스터 기억(master_lore)가 있습니다.
    
    당신은 이 정보들을 바탕으로 사용자가 선택할 수 있는 선택지를 생성해야 합니다.
    사용자는 이동을 할 수 있고, 현재 위치에 요괴가 있으면 퇴치를 시도할 수도 있으며
    퇴치방법을 모르는 경우에는 가지고 있는 아이템들을 활용해서 시도할 수도 있습니다.
    각 선택지는 [행동|성공 시 이점|실패시 위험]으로 구성되어야 합니다.
    문맥이 파괴되지 않도록 조심해서 만들어주세요.
    """),
    SystemMessagePromptTemplate("""
    ###맵
    ```json
    {map}
    ```
    ###요괴
    ```json
    {monster}
    ```
    ###사용자 정보
    health: {health}
    sanity: {sanity}
    current_location: {current_location}
    current_mood: {current_mood}
    items: {items}
    player_lore: {player_lore}
    ###세계 정보
    master_lore: {master_lore}
    """),
    MessagesPlaceholder("history")
])
choices_generate = choices_generate_prompt_template | llm.with_structured_output(UserData)
progress_generate_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("""
    당신은 텍스트 어드벤처 게임의 진행 담당자입니다.
    현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
    이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
    요괴들에 대한 정보는 json형식으로 제공됩니다.
    요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
    사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
    플레이어의 기억(player_lore)들이 있습니다.
    세계에 대한 정보로는 진행기록(history), 마스터 기억(master_lore)가 있습니다.
    
    당신은 이 정보들을 바탕으로 게임을 진행해야 합니다.
    마지막의 사용자의 선택지를 참고하여 그 선택지에 대한 적절한 진행을 해주세요.
    게임 진행이 아니라 튜토리얼 상태라면 히스토리를 초기화하고 게임의 처음상태부터 진행해주세요
    
    ###history 생성 지침 
    세계에 대한 나레이션을 진행하고자 한다면 narration을
    플레이어나 요괴의 대사가 진행되는 경우에는 dialogue를 만들어주세요.
    문맥이 파괴되지 않도록 조심해서 만들어주세요.
    ###사용자 정보 편집 지침
    사용자의 선택, 이야기의 흐름에 따라서 편집해야 하는 경우에 편집해주세요.
    ###맵 정보 편집 지침
    맵의 특정 노드에서 사건이 발생했고, 이를 기억해야 하는 경우에는 lore에 기록하세요.
    노드에서 요괴가 퇴치되었다면 monster를 공백으로 바꾸세요.
    ###요괴 정보 편집 지침
    요괴가 퇴치되었거나 하는 경우에는 해당 몬스터의 정보를 없애세요.
    """),
    SystemMessagePromptTemplate("""
    ###맵
    ```json
    {map}
    ```
    ###요괴
    ```json
    {monster}
    ```
    ###사용자 정보
    health: {health}
    sanity: {sanity}
    current_location: {current_location}
    current_mood: {current_mood}
    items: {items}
    player_lore: {player_lore}
    ###세계 정보
    master_lore: {master_lore}
    """),
    MessagesPlaceholder("history")
])
progress_generate = progress_generate_prompt_template | llm.with_structured_output(UserData)
supervisor_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("""
    당신은 텍스트 어드벤처 게임의 문맥 감독입니다.
    현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
    이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
    요괴들에 대한 정보는 json형식으로 제공됩니다.
    요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
    사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
    플레이어의 기억(player_lore)들이 있습니다.
    세계에 대한 정보로는 진행기록(history), 마스터 기억(master_lore)가 있습니다.
    
    당신은 이 정보들을 바탕으로 게임의 문맥을 감독해야 합니다.
    문맥이 맞지 않다고 판단되는 부분은 편집을 통해서 적절게 만들어주세요.
    단, 너무 오래된 기록은 건드리지 마세요.
    감독이 끝난 후에 추가적인 감독이 필요하지 않다면 supervising_done을 true로,
    추가적인 감독이 필요하다면 false로 설정해주세요.
    """),
    HumanMessage("""
    ###맵
    ```json
    {map}
    ```
    ###요괴
    ```json
    {monster}
    ```
    ###사용자 정보
    health: {health}
    sanity: {sanity}
    current_location: {current_location}
    current_mood: {current_mood}
    items: {items}
    player_lore: {player_lore}
    ###세계 정보
    master_lore: {master_lore}
    """),
    MessagesPlaceholder("history")
])
supervisor = supervisor_prompt_template | llm.with_structured_output(UserData)

progress_graph_builder.add_node("내용 진행", progress_generate)
progress_graph_builder.add_node("감독", supervisor)
progress_graph_builder.add_node("선택지 생성", choices_generate)
progress_graph_builder.add_edge(START, "내용 진행")
progress_graph_builder.add_edge("내용 진행", "감독")
progress_graph_builder.add_conditional_edges("감독", lambda state: "선택지 생성" if state["supervising_done"] else "감독", ["감독", "선택지 생성"])
progress_graph_builder.add_edge("선택지 생성", END)
graph = progress_graph_builder.compile()
try:
    png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
except Exception:
    pass

def init_game(user_id, tutorial):
    with open("map.json", "r", encoding="utf-8") as f:
        map = json.load(f)

    with open("monster.json", "r", encoding="utf-8") as f:
        monster = json.load(f)

    map_str = json.dumps(map, ensure_ascii=False, indent=2)
    monster_str = json.dumps(monster, ensure_ascii=False, indent=2)
    prompt_template = PromptTemplate.from_template("""
    당신은 맵 그래프에 요괴를 배치하는 에이전트입니다.
    맵 그래프는 json으로 노드와 엣지로 구성되어 있으며,
    배치할 수 있는 요괴의 목록도 json으로 제공됩니다.
    노드에 요괴를 배치할 때에는 monster에 배치할 요괴의 이름을 삽입하세요.
    각 요괴는 중복으로 배치될 수 없으며, 모든 요괴를 한 번만 사용하세요.
    node6(대웅보전)에는 무조건 구미호를 배치해주세요.
    이외의 요괴들은 여러 노드에 분포시켜주세요.
    배치한 결과도 같은 json이어야 합니다.
    제공된 맵의 기본 구조를 파괴해서는 안됩니다.
    ### 맵
    ```json
    {map}
    ```
    ### 요괴
    ```json
    {monster}
    ```
    ###
    """)

    initialized_map = (prompt_template | llm_reasoning.with_structured_output(Map)).invoke({"map": map_str, "monster": monster_str})
    #print(initialized_map.map)
    #print(type(initialized_map))

    user_data = UserData(
        health=5,
        sanity=5,
        purification=50,
        current_location='일주문',
        current_mood='당황스러움',
        items='',
        choices=['다음으로'],
        history=[History(type='narration', speaker='', text='고운사를 요괴들이 장악했습니다.'),
                 History(type='narration', speaker='', text='선택지를 골라가며 자기만의 방식으로 요괴들을 퇴치해가보세요!')],
        map=initialized_map.map,
        monster=monster_str,
        player_lore='',
        master_lore='',
    )
    doc_ref = db.collection("users").add(user_data.model_dump(), user_id)

class Ctx(TypedDict):
    total_history: List[dict]
    current_history: History

def progress(user_id, select):
    doc_ref = db.collection("users").document(user_id).get()

    user_data = doc_ref.to_dict()
    user_data['total_history'] = []
    user_data['current_history'] = None
    graph_data = UserData(**user_data)

    #print(graph_data)
    progress_graph_builder = StateGraph(UserData, context_schema=Ctx)
    def 진행(state: UserData, runtime: Runtime[Ctx]):
        progress_generate_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage("""
            당신은 텍스트 어드벤처 게임의 진행 담당자입니다.
            현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
            이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
            요괴들에 대한 정보는 json형식으로 제공됩니다.
            요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
            사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
            플레이어의 기억(player_lore)들이 있습니다.
            세계에 대한 정보로는 진행기록(history), 마스터 기억(master_lore)가 있습니다.

            당신은 이 정보들을 바탕으로 게임을 진행해야 합니다.
            마지막의 사용자의 선택지를 참고하여 그 선택지에 대한 적절한 진행을 해주세요.
            게임 진행이 아니라 튜토리얼 상태라면 히스토리를 초기화하고 게임의 처음상태부터 진행해주세요

            ###history 생성 지침 
            세계에 대한 나레이션을 진행하고자 한다면 narration을
            플레이어나 요괴의 대사가 진행되는 경우에는 dialogue를 만들어주세요.
            문맥이 파괴되지 않도록 조심해서 만들어주세요.
            ###사용자 정보 편집 지침
            사용자의 선택, 이야기의 흐름에 따라서 편집해야 하는 경우에 편집해주세요.
            ###맵 정보 편집 지침
            맵의 특정 노드에서 사건이 발생했고, 이를 기억해야 하는 경우에는 lore에 기록하세요.
            노드에서 요괴가 퇴치되었다면 monster를 공백으로 바꾸세요.
            ###요괴 정보 편집 지침
            요괴가 퇴치되었거나 하는 경우에는 해당 몬스터의 정보를 없애세요.
            """),
            HumanMessage("""
            ###맵
            ```json
            {map}
            ```
            ###요괴
            ```json
            {monster}
            ```
            ###사용자 정보
            health: {health}
            sanity: {sanity}
            current_location: {current_location}
            current_mood: {current_mood}
            items: {items}
            player_lore: {player_lore}
            사용자의 선택 행동: {player_selection}
            ###세계 정보
            master_lore: {master_lore}
            """),
            MessagesPlaceholder("history")
        ])
        #print(state.history)
        history = []
        for h in state.history:
            #history.append({"role": f"[{h.type}]{h.speaker}", "content": h.text})
            history.append({"role": "human" if h.speaker == "player" else "assistant", "content": h.text})
        #print(history)

        output = (progress_generate_prompt_template | llm.with_structured_output(Histories)).invoke({
            'map': state.map,
            'monster': state.monster,
            'health': state.health,
            'sanity': state.sanity,
            'current_location': state.current_location,
            'current_mood': state.current_mood,
            'items': state.items,
            'player_lore': state.player_lore,
            'player_selection': state.choices[select],
            'master_lore': state.master_lore,
            'history': history
        })

        runtime.context['current_history'] = output.generated_history

    def 감독(state: UserData, runtime: Runtime[Ctx]):
        #print(runtime.context['current_history'])
        supervisor_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage("""
            당신은 텍스트 어드벤처 게임의 문맥 감독입니다.
            현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
            이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
            요괴들에 대한 정보는 json형식으로 제공됩니다.
            요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
            사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
            플레이어의 기억(player_lore)들이 있습니다.
            세계에 대한 정보로는 진행기록(history), 마스터 기억(master_lore)가 있습니다.

            당신은 이 정보들을 바탕으로 현재의 진행 사항을 감독해야 합니다.
            문맥이 맞지 않다고 판단되는 부분은 편집을 통해서 적절하게 만들어주세요.
            편집이 되었건 되지 않았건 제공된 모든 진행 내용은 edited_history로 다시 전달해야합니다.
            문맥 상에 문제가 없다면 편집을 하지 않고 그대로 edited_history로 전달하세요.
            감독이 끝난 후에 추가적인 감독이 필요하지 않다면 supervising_done을 True로,
            추가적인 감독이 필요하다면 False로 설정해주세요.
            """),
            SystemMessagePromptTemplate("""
            ###맵
            ```json
            {map}
            ```
            ###요괴
            ```json
            {monster}
            ```
            ###사용자 정보
            health: {health}
            sanity: {sanity}
            current_location: {current_location}
            current_mood: {current_mood}
            items: {items}
            player_lore: {player_lore}
            ###세계 정보
            master_lore: {master_lore}
            ###과거 진행 기록
            """),
            MessagesPlaceholder("total_history"),
            SystemMessage("""
            ###현재 진행기록
            """),
            MessagesPlaceholder("current_history")
        ])
        runtime.context['supervising_done'] = False

        history = []
        for h in state.history:
            # history.append({"role": f"[{h.type}]{h.speaker}", "content": h.text})
            history.append({"role": "human" if h.speaker == "player" else "assistant", "content": h.text})
        current_history = []
        for h in runtime.context['current_history']:
            current_history.append({"role": "human" if h.speaker == "player" else "assistant", "content": h.text})

        output = (supervisor_prompt_template | llm.with_structured_output(Supervising)).invoke({
            'map': state.map,
            'monster': state.monster,
            'health': state.health,
            'sanity': state.sanity,
            'current_location': state.current_location,
            'current_mood': state.current_mood,
            'items': state.items,
            'player_lore': state.player_lore,
            'player_selection': state.choices[select],
            'master_lore': state.master_lore,
            'total_history': history,
            'current_history': current_history
        })
        #print(output)

        runtime.context['supervising_done'] = output.supervising_done
        # if output.supervising_done:
        #     return {"history": state.history + output.edited_history}
        # else:
        #     return

    def 선택지_분기(state:UserData, runtime: Runtime[Ctx]):
        if runtime.context['supervising_done']:
            return "진행 적용"
        else:
            return "진행"

    def 진행사항_적용(state: UserData, runtime: Runtime[Ctx]):
        #print(state)
        #print(runtime.context['current_history'])
        apply_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage("""
                    당신은 텍스트 어드벤처 게임의 문맥 적용 전문가입니다. map/monster의 JSON을 재작성하지 마세요.
                    현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
                    * 맵은 nodes(id, name, monster, items, lore), edges(source, target)로 구성됩니다.
                    * edges는 source, target으로 구성됩니다.
                    이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
                    * monster는 monsters로 구성되며, 각 요괴는 name, 퇴치법, lore로 구성됩니다. 
                    요괴들에 대한 정보는 json형식으로 제공됩니다.
                    요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
                    사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
                    플레이어의 기억(player_lore)들이 있습니다.
                    세계에 대한 정보로는 진행기록(total_history), 마스터 기억(master_lore)가 있습니다.

                    추가적으로 현재 진행사항에 대한 정보가 제공되며, 이 기록을 바탕으로 사용자의 상태들과 맵, 요괴들의 상태들을 변경해야 합니다.
                    이전까지의 상태에서 현재 진행사항을 바탕으로 문맥에 맞게 상태를 변경해야 합니다.
                    
                    진행사항을 적용해야하는 정보들은 다음과 같습니다.
                    * 플레이어의 체력(health): 체력이 닳으면 -1, 회복하면 +1, 변경이 없으면 그대로 전달
                    * 플레이어의 정신력(sanity): 정신력이 닳으면 -1, 회복하면 +1, 변경이 없으면 그대로 전달
                    * 고운사의 요괴 정화도 수준(purification): 요괴를 퇴치한 경우 증가하고, 그 이외에는 변경 없이 그대로 전달하세요.
                    * 플레이어의 현재 위치(current_location): 현재 진행사항에서 플레이어가 이동하는 경우에 변경, 이동 이외의 행동을 하는 경우에는 그대로 전달. 사용자는 맵에 있는 노드 이외의 장소에 있을 수 없습니다.
                        * 일주문은 사찰의 입구를 의미합니다.
                    * 플레이어의 현재 기분(current_mood): 진행사항을 참고해서 플레이어가 가질 기분을 적어주세요. 한 단어로 적어야 합니다.
                    * 아이템 목록(items): 현재 진행사항에서 플레이어가 특정 아이템을 얻은 것이 아니라면 변경 없이 그대로 전달
                    * 맵(map): 현재 진행사항을 바탕으로 기존 맵 정보를 편집하세요.
                        * 절대로 제공되는 기본 구조를 파괴해서는 안됩니다.
                        * 절대로 기존의 id, name필드 데이터는 변경해서는 안됩니다.
                        * 새로운 노드의 추가, 기존 노드의 삭제는 안됩니다.
                        * 각 노드의 순서를 변경하고, 연결 관계를 재정의해서는 안됩니다.
                        * 변경 사항이 적용되는 맵은 이전 맵과 진행상황을 고려해서 변경되어야 하며, 이 때, 문맥이 맞아야 합니다.
                        * 노드에 있던 요괴를 갑자기 없애거나 만들어내지 마세요.
                        * 변경사항이 없다면 그대로 전달하세요.
                    * 요괴(monster): 현재 진행사항에서 요괴에 대한 변경할 사항이 있다면 편집하고, 이외에는 그대로 전달하세요.
                        * [주의] 절대로 제공되는 monster의 기본구조를 파괴해서는 안됩니다.
                    * 플레이어의 기억(player_lore): 플레이어가 기억할 내용이 있다면 그 내용을 추가하고, 없다면 그대로 전달하세요.
                        * 무조건 제공되는 내용에서 추가되어야 하며, 갑자기 큰 변화가 있어서는 안됩니다.
                    * 게임 전반에 걸친 기억(master_lore): 게임 전반에서 기억해야 할 내용이 있다면 그 내용을 추가하고, 없다면 그대로 전달하세요.
                        * 무조건 제공되는 내용에서 추가되어야 하며, 갑자기 큰 변화가 있어서는 안됩니다.
                    
                    이 정보들은 제공되는 정보를 바탕으로 변경사항을 적용해야 하고, 완전히 새로 만들어져서는 안됩니다.
                    """),
            SystemMessagePromptTemplate.from_template("""
                    ###맵
                    ```json
                    {map}
                    ```
                    ###요괴
                    ```json
                    {monster}
                    ```
                    ###사용자 정보
                    health: {health}
                    sanity: {sanity}
                    current_location: {current_location}
                    current_mood: {current_mood}
                    items: {items}
                    player_lore: {player_lore}
                    ###세계 정보
                    master_lore: {master_lore}
                    ###지금까지의 진행사항
                    """),
            MessagesPlaceholder("total_history"),
            SystemMessage("""
                        ###현재 진행사항
                        """),
            MessagesPlaceholder("current_history"),
            #HumanMessage("###진행사항 적용 결과:")
        ])

        history = []
        for h in state.history:
            # history.append({"role": f"[{h.type}]{h.speaker}", "content": h.text})
            history.append({"role": "human" if h.speaker == "player" else "assistant", "content": h.text})
        current_history = []
        for h in runtime.context['current_history']:
            current_history.append({"role": "human" if h.speaker == "player" else "assistant", "content": h.text})

        output = (apply_prompt_template | llm_reasoning.with_structured_output(ChangedUserData)).invoke({
            "map": state.map,
            "monster": state.monster,
            "health": state.health,
            "sanity": state.sanity,
            "current_location": state.current_location,
            "current_mood": state.current_mood,
            "items": state.items,
            "player_lore": state.player_lore,
            "master_lore": state.master_lore,
            "total_history": history,
            "current_history": current_history
        })

        output_dict = output.model_dump()
        output_dict['history'] = state.history
        output_dict['history'].append(History(type="narration", speaker="", text=f"{state.choices[select]} 선택"))

        #print(output)
        return output_dict

    def 선택지_생성(state: UserData, runtime: Runtime[Ctx]):
        #print("넘어옴")
        choices_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage("""
                            당신은 텍스트 어드벤처 게임의 선택지 생성 전문가입니다.
                            현재 맵에 대한 정보는 노드와 엣지로 구성된 json형식으로 제공됩니다.
                            * 맵은 nodes(id, name, monster, items, lore), edges(source, target)로 구성됩니다.
                            * edges는 source, target으로 구성됩니다.
                            이 맵에는 배치된 요괴에 대한 정보와 노드마다 사건을 기록하는 lore로 구성됩니다.
                            * monster는 monsters로 구성되며, 각 요괴는 name, 퇴치법, lore로 구성됩니다. 
                            요괴들에 대한 정보는 json형식으로 제공됩니다.
                            요괴들의 정보에도 사건을 기록하는 lore가 있으며, 사용자가 탐색을 통해서 알아낸 퇴치법에 대한 정보도 있습니다.
                            사용자의 정보는 체력(health), 정신력(sanity), 현재 위치(current_location), 현재 기분(current_mood), 아이템 목록(items),
                            플레이어의 기억(player_lore)들이 있습니다.
                            세계에 대한 정보로는 진행기록(total_history), 마스터 기억(master_lore)가 있습니다.

                            추가적으로 현재 진행사항에 대한 정보가 제공되며, 이 기록들을 바탕으로 사용자가 할 수 있는 행위 선택지를 2~5까지 만들어야 합니다.

                            진행사항을 적용해야하는 정보들은 다음과 같습니다.
                            * 플레이어의 체력(health)
                            * 플레이어의 정신력(sanity)
                            * 고운사의 요괴 정화도 수준(purification)
                            * 플레이어의 현재 위치(current_location)
                            * 플레이어의 현재 기분(current_mood)
                            * 아이템 목록(items)
                            * 맵(map)
                            * 요괴(monster)
                            * 플레이어의 기억(player_lore)
                            * 게임 전반에 걸친 기억(master_lore)
                            """),
            SystemMessagePromptTemplate.from_template("""
                            ###맵
                            ```json
                            {map}
                            ```
                            ###요괴
                            ```json
                            {monster}
                            ```
                            ###사용자 정보
                            health: {health}
                            sanity: {sanity}
                            current_location: {current_location}
                            current_mood: {current_mood}
                            items: {items}
                            player_lore: {player_lore}
                            ###세계 정보
                            master_lore: {master_lore}
                            ###지금까지의 진행사항
                            """),
            MessagesPlaceholder("total_history"),
            SystemMessage("""
                                ###현재 진행사항
                                """),
            MessagesPlaceholder("current_history"),
            # HumanMessage("###진행사항 적용 결과:")
        ])

        history = []
        for h in state.history:
            # history.append({"role": f"[{h.type}]{h.speaker}", "content": h.text})
            history.append({"role": "human" if h.speaker == "player" else "assistant", "content": h.text})
        current_history = []
        for h in runtime.context['current_history']:
            current_history.append({"role": "human" if h.speaker == "player" else "assistant", "content": h.text})

        output = (choices_prompt_template | llm_reasoning.with_structured_output(Choices)).invoke({
            "map": state.map,
            "monster": state.monster,
            "health": state.health,
            "sanity": state.sanity,
            "current_location": state.current_location,
            "current_mood": state.current_mood,
            "items": state.items,
            "player_lore": state.player_lore,
            "master_lore": state.master_lore,
            "total_history": history,
            "current_history": current_history
        })

        #state.history = state.history + runtime.context['current_history']
        #print(runtime.context['current_history'])
        #print(state.history)

        #print(output)

        output_dict = output.model_dump()
        output_dict['history'] = state.history + runtime.context['current_history']

        return output_dict

    def 서버에적용(state: UserData):
        #print(state)
        doc_ref = db.collection("users").document(user_id)
        doc_ref.update(state.model_dump())

    progress_graph_builder.add_node("진행", 진행)
    progress_graph_builder.add_node("감독", 감독)
    progress_graph_builder.add_node("진행 적용", 진행사항_적용)
    progress_graph_builder.add_node("선택지 생성", 선택지_생성)
    progress_graph_builder.add_node("서버에 적용", 서버에적용)

    progress_graph_builder.add_edge(START, "진행")
    progress_graph_builder.add_edge("진행", "감독")
    progress_graph_builder.add_conditional_edges("감독", 선택지_분기, ["진행 적용", "진행"])
    progress_graph_builder.add_edge("진행 적용", "선택지 생성")
    progress_graph_builder.add_edge("선택지 생성", "서버에 적용")
    progress_graph_builder.add_edge("서버에 적용", END)

    progress_graph = progress_graph_builder.compile()
    # output = graph.invoke(user_data)
    #output = progress_graph.invoke(user_data, context={"select": select})
    output = progress_graph.invoke(graph_data, context={"select": select})
    # print(output)

    try:
        png_bytes = progress_graph.get_graph(xray=True).draw_mermaid_png()
        with open("progress_graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass

#init_game("tttt", True)
progress("tttt", 0)