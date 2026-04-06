import os
import json
from openai import OpenAI
from src.env import VulnTriageEnv
from src.models import TriageAction

# Ensure you have OPENAI_API_KEY set in your environment variables
client = OpenAI()

def run_baseline():
    env = VulnTriageEnv(task_level="easy")
    obs = env.reset()
    done = False
    
    print("--- Starting Task 1: The Noise Filter ---")
    
    while not done:
        print(f"\nCurrent Alerts: {[a.alert_id for a in obs.open_alerts]}")
        
        # Ask the LLM what to do
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-4-turbo
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