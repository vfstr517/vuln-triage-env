import uvicorn
from fastapi import FastAPI
from src.env import VulnTriageEnv

# Minimal server wrapper required by OpenEnv multi-mode deployment
app = FastAPI(title="VulnTriageEnv Server")

@app.get("/health")
def health_check():
    return {"status": "ok", "environment": "VulnTriageEnv"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()