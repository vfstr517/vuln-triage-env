import os
import json
from openai import OpenAI
from src.env import VulnTriageEnv
from src.models import TriageAction

# 1. Update this to use Groq's URL and look for your new secret
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def run_baseline():
    tasks = ["easy", "medium", "hard"]
    
    for level in tasks:
        print(f"\n{'='*40}\nStarting Task: {level.upper()}\n{'='*40}")
        env = VulnTriageEnv(task_level=level)
        obs = env.reset()
        done = False
        
        while not done:
            # 2. Update the model name here
            response = client.chat.completions.create(
                model="llama3-70b-8192", # Groq's best model for this
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": f"You are a DevSecOps agent. Output a JSON action matching this schema: {TriageAction.model_json_schema()}"},
                    {"role": "user", "content": f"Observation: {obs.model_dump_json()}"}
                ]
            )
            
            raw_action = json.loads(response.choices[0].message.content)
            action = TriageAction(**raw_action)
            print(f"-> Agent chose: {action.action_type} for {action.alert_id}")
            
            obs, reward, done, info = env.step(action)
            print(f"   Reward: {reward.value} | Reason: {reward.reasoning}")

        print(f"\n>>> Task {level.upper()} Complete! Final Score: {env.state()['internal_score']:.2f} / 1.00 <<<")

if __name__ == "__main__":
    run_baseline()