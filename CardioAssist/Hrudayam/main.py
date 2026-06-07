from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working!"}

@app.get("/data")
def get_data():
    return {"name": "Abhi", "role": "Student"}