import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("uiseong2077-firebase-key.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = FastAPI()

class User(BaseModel):
    name: str
    email: str
    age: int | None = None

@app.post("/users/", status_code=201)
def create_user(user: User):
    doc_ref = db.collection("users").add(user.dict())
    return {"id": doc_ref[1].id}

@app.get("/users/{user_id}")
def get_user(user_id: str):
    try:
        doc = db.collection("users").document(user_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        return doc.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/")
def get_all_users():
    users_ref = db.collection("users")
    docs = users_ref.stream()

    users = []
    for doc in docs:
        user_data = doc.to_dict()
        user_data['id'] = doc.id
        users.append(user_data)

    return users

@app.put("/users/{user_id}")
def update_user(user_id: str, user: User):
    try:
        db.collection("users").document(user_id).set(user.dict())
        return {"message": f"User {user_id} updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    try:
        db.collection("users").document(user_id).delete()
        return {"message": f"User {user_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)