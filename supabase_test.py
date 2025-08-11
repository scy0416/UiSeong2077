import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 환경변수를 설정하세요.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI(title="FastAPI x Supabase test")

class TodoIn(BaseModel):
    title: str
    done: bool = False

class TodoOut(TodoIn):
    id: int

@app.get("/health")
def health():
    try:
        res = supabase.table("todos").select("id").limit(1).execute()
        return {"ok": True, "rows_seen": len(res.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/todos", response_model=List[TodoOut])
def list_todos():
    try:
        res = supabase.table("todos").select("*").order("id").execute()
        return res.data
    except Exception as e:
        raise HTTPException(500, f"select failed: {e}")

@app.post("/todos", response_model=TodoOut, status_code=201)
def create_todo(todo: TodoIn):
    try:
        res = supabase.table("todos").insert(todo.model_dump()).execute()
        return res.data[0]
    except Exception as e:
        raise HTTPException(500, f"insert failed: {e}")

@app.put("/todos/{todo_id}", response_model=TodoOut)
def update_todo(todo_id: int = Path(..., ge=1), payload: TodoIn = None):
    try:
        res = supabase.table("todos").update(payload.model_dump()).eq("id", todo_id).execute()
        if not res.data:
            raise HTTPException(404, "not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"update failed: {e}")

@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int = Path(..., ge=1)):
    try:
        res = supabase.table("todos").delete().eq("id", todo_id).execute()
        if res.count == 0 and not res.data:
            # supabase-py는 count가 없을 수도 있어요. data 비어있으면 없는 것으로 처리.
            pass
        return
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")