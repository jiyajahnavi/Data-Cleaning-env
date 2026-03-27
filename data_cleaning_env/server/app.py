from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os

from data_cleaning_env.models import Action, Observation, ResetConfig
from data_cleaning_env.server.environment import DataCleaningEnv

app = FastAPI(title="DataCleaningEnv Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DataCleaningEnv()

@app.get("/")
def get_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    return FileResponse(ui_path)

@app.post("/reset")
def reset_environment(config: Optional[ResetConfig] = Body(default=None)):
    task_id = config.task_id if config else "easy"
    obs = env.reset(task_id=task_id)
    return obs

@app.post("/step")
def take_step(action: Action):
    result = env.step(action)
    return result

@app.get("/state")
def get_state():
    return env.state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("data_cleaning_env.server.app:app", host="127.0.0.1", port=8000, reload=True)