import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse, JSONResponse
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer

load_dotenv()

# 환경 변수 로드
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLITNE_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

app = FastAPI()

# 서버의 JWT 인증을 위한 설정
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# google oauth2 관련 엔드포인트
@app.get("/login/google")
async def login_google():
    """
    google 로그인 페이지로 리디렉션합니다.
    """
    auth_uri = (
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
    )
    return RedirectResponse(url=auth_uri)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    """
    google 로그인 후 리디렉션되는 콜백 처리
    인증 코드를 액세스 토큰으로 교환하고, 사용자 정보로 자체 JWT를 생성한다.
    """
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not found")

    # 액세스 토큰 요청
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLITNE_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code"
            }
        )

    token_data = token_response.json()
    if "access_token" not in token_data:
        raise HTTPException(status_code=400, detail="Could not get access token")

    access_token = token_data["access_token"]

    # 사용자 정보 요청
    async with httpx.AsyncClient() as client:
        user_info_response = await client.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )

    user_info = user_info_response.json()

    jwt_payload = {
        "sub": user_info["email"],
        "name": user_info.get("name"),
        "exp": datetime.utcnow() + timedelta(hours=1),
    }

    app_token = jwt.encode(jwt_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    response = JSONResponse(content={"message": "Authentication successful"})

    response.set_cookie(
        key="access_token",
        value=app_token,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 24
    )

    return response