import os
import json
from openai import OpenAI
from src.env import VulnTriageEnv
from src.models import TriageAction

# Initialize the client with Grok's base URL and look for a GROK API Key
client = OpenAI(
    api_key=os.environ.get("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)

def run_baseline():
    env = VulnTriageEnv(task_level="easy")
    obs = env.reset()
    done = False
    
    print("--- Starting Task 1: The Noise Filter ---")
    
    while not done:
        print(f"\nCurrent Alerts: {[a.alert_id for a in obs.open_alerts]}")
        
        # Ask Grok what to do
        response = client.chat.completions.create(
            model="grok-2-latest", # <--- Update this to the Grok model you want to use
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": f"You are a DevSecOps agent. Review the observation and output a JSON action matching this schema: {TriageAction.model_json_schema()}"},
                {"role": "user", "content": f"Observation: {obs.model_dump_json()}"}
            ]
        )
        
        # Parse the LLM's action
        raw_action = json.loads(response.choices[0].message.content)
        action = TriageAction(**raw_action)
        print(f"Agent chose: {action.action_type} for {action.alert_id}")
        
        # Take a step in the environment
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward.value} | Reasoning: {reward.reasoning}")

    final_state = env.state()
    print(f"\nTask Complete! Final Score: {final_state['internal_score']:.2f} / 1.00")

if __name__ == "__main__":
    run_baseline()