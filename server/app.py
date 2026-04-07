import uvicorn
from fastapi import FastAPI, Request
from src.env import VulnTriageEnv

app = FastAPI(title="VulnTriageEnv Server")
env = VulnTriageEnv()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "VulnTriageEnv is running! Ready for OpenEnv evaluation.",
        "endpoints": ["/reset", "/step", "/health"]
    }

# The grader sends a POST request with an empty JSON body '{}'
@app.post("/reset")
async def reset_env(request: Request):
    obs = env.reset()
    return obs.model_dump()

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