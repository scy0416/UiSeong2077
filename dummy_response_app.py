import os

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트 도메인으로 교체
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_CLIENT_ID = os.getenv("UISEONG_GOOGLE_CLIENT_ID")
GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

def verify_google_id_token(credential: str) -> dict:
    try:
        idinfo = id_token.verify_oauth2_token(credential, requests.Request(), GOOGLE_CLIENT_ID)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google ID token: {e}")

    if idinfo.get("iss") not in GOOGLE_ISSUERS:
        raise HTTPException(status_code=401, detail="Wrong Issuer")

    return idinfo

def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[0]
    return verify_google_id_token(token)

@app.get("/newgame")
#def newgame(tutorial: bool, user: dict = Depends(get_current_user)):
def newgame(tutorial: bool):
    #return Response(200)
    return RedirectResponse("http://localhost:3000/game")

@app.get("/loadgame")
#def loadgame(user: dict = Depends(get_current_user)):
def loadgame():
    return RedirectResponse("http://localhost:3000/game")

@app.get("/loadinfo")
#def loadinfo(user: dict = Depends(get_current_user)):
def loadinfo():
    return {
        "health": 5,
        "sanity": 5,
        "purification": 50,
        "current_location": "일주문",
        "current_mood": "불안함",
        "items": "소금, 거울",
        "job": "일반인",
        "choices": ["[선택지1] 아무튼 선택지", "[선택지2] 아무튼 선택지", "[선택지3] 아무튼 선택지"],
        "history": [
            {"type": "narration", "text": "현재 상황이 이러이러하다."},
            {"type": "dialogue", "speaker": "달걀귀신", "text": "안녕"},
            {"type": "choice", "text": "[선택지1] 아무튼 선택지"}
        ],
        "username": "의성2077"
    }

@app.post("/selectchoice")
#def selectchoice(user: dict = Depends(get_current_user)):
def selectchoice():
    return Response(200)

