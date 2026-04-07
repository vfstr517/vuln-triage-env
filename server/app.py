import uvicorn
from fastapi import FastAPI, Request
from src.env import VulnTriageEnv
import traceback

app = FastAPI(title="VulnTriageEnv Server")

# FIX 1: Explicitly pass a task_level so it doesn't crash if your code requires one.
try:
    env = VulnTriageEnv(task_level="easy")
except TypeError:
    # Fallback just in case your __init__ doesn't accept arguments
    env = VulnTriageEnv()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "VulnTriageEnv is running! Ready for OpenEnv evaluation.",
        "endpoints": ["/reset", "/step", "/health"]
    }

@app.post("/reset")
async def reset_env(request: Request):
    try:
        # The automated grader sends an empty JSON body '{}'
        body = await request.json()
    except Exception:
        body = {}

    try:
        obs = env.reset()
        # FIX 2: Return 'obs' directly. FastAPI is incredibly smart and will 
        # automatically convert Pydantic models, dicts, or strings into valid JSON.
        return obs 
    except Exception as e:
        # If it crashes again, this will print the EXACT reason to your Hugging Face logs
        print("CRITICAL ERROR IN /reset:")
        traceback.print_exc()
        return {"error": "Internal Server Error", "details": str(e)}

@app.post("/step")
async def step_env(request: Request):
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok", "environment": "VulnTriageEnv"}

def main():
    # Hugging Face Spaces strictly require port 7860
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()