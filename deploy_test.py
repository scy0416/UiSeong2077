import os

from fastapi import FastAPI, HTTPException
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

class TokenIn(BaseModel):
    credential: str

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/identify")
def identify(body: TokenIn):
    try:
        idinfo = id_token.verify_oauth2_token(
            body.credential,
            requests.Request(),
            GOOGLE_CLIENT_ID
        )
        if idinfo["iss"] not in ("accounts.google.com", "https://accounts.google.com"):
            raise HTTPException(status_code=401, detail="Wrong issuer")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google ID token")

    return {
        "google_id": idinfo["sub"],
        "email": idinfo.get("email"),
        "name": idinfo.get("name"),
        "picture": idinfo.get("picture")
    }