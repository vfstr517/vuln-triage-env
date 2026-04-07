import os
import json
import sys
from openai import OpenAI
from src.env import VulnTriageEnv
from src.models import TriageAction

# 1. Required Environment Variables exactly as specified by checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "VulnTriageEnv"

if not HF_TOKEN:
    print("CRITICAL ERROR: HF_TOKEN is missing from the environment variables!")
    sys.exit(1)

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    for level in tasks:
        env = VulnTriageEnv(task_level=level)
        obs = env.reset()
        done = False
        step_num = 0
        rewards_history = []
        
        # MANDATORY [START] LOG
        print(f"[START] task={level} env={BENCHMARK} model={MODEL_NAME}")
        
        while not done:
            step_num += 1
            error_msg = "null"
            action_str = "none"
            reward_val = 0.00
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    response_format={ "type": "json_object" },
                    messages=[
                        {"role": "system", "content": f"You are a DevSecOps agent. Output a JSON action matching this schema: {TriageAction.model_json_schema()}"},
                        {"role": "user", "content": f"Observation: {obs.model_dump_json()}"}
                    ]
                )
                
                raw_action = json.loads(response.choices[0].message.content)
                action = TriageAction(**raw_action)
                
                # Format action string to avoid breaking their parser
                action_str = f"{action.action_type}({action.alert_id})"
                
                obs, reward, done, info = env.step(action)
                reward_val = reward.value
                
            except Exception as e:
                # Catch LLM or JSON errors and report them exactly as required
                error_msg = str(e).replace(' ', '_') 
                done = True
                
            # Keep track of rewards for the END log
            rewards_history.append(f"{reward_val:.2f}")
            
            # MANDATORY [STEP] LOG
            done_str = "true" if done else "false"
            print(f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={done_str} error={error_msg}")

        # MANDATORY [END] LOG
        final_score = env.state()['internal_score']
        success_str = "true" if final_score > 0.0 else "false" 
        rewards_joined = ",".join(rewards_history)
        
        print(f"[END] success={success_str} steps={step_num} score={final_score:.2f} rewards={rewards_joined}")

if __name__ == "__main__":
    run_inference()