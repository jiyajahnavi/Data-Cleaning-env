import os
import requests
import json
from data_cleaning_env.models import Action, Observation

class DataCleaningEnvClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def reset(self, task_id="easy") -> Observation:
        response = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        response.raise_for_status()
        return Observation(**response.json())

    def step(self, action: Action):
        response = requests.post(f"{self.base_url}/step", json=action.model_dump())
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    from google import genai
    
    # Read GEMINI_API_KEY from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run the baseline!")
        exit(1)
        
    client = genai.Client(api_key=api_key)
    
    env = DataCleaningEnvClient()

    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        print(f"\n--- Starting Task: {task.upper()} ---")
        obs = env.reset(task_id=task)
        done = False
        step = 0
        total_reward = 0.0
        
        while not done and step < 10:
            import time
            time.sleep(4)  # ⏳ Wait a few seconds to avoid hitting the free tier rate limit
            
            prompt = f"You are an agent cleaning dataset. Current dataset quality is {obs.quality_score:.2f} (1.0 is perfect).\n\n"
            prompt += "Action space:\n"
            prompt += "- remove_nulls\n- fill_missing\n- remove_duplicates\n- lowercase\n- remove_punctuation\n- remove_stopwords\n- normalize_text\n- no_op\n\n"
            prompt += f"Cleaned Data State:\n{json.dumps(obs.cleaned_data, indent=2)}\n\n"
            prompt += "Select exactly one action name from the Action space list to apply to the dataset. Reply with just the action name and nothing else."
            
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt
            )
            action_name = response.text.strip().replace("`", "").strip()
            
            print(f"Agent chose: {action_name}")
            
            # create action object
            action_dict = {
                "remove_nulls": action_name == "remove_nulls",
                "fill_missing": action_name == "fill_missing",
                "remove_duplicates": action_name == "remove_duplicates",
                "lowercase": action_name == "lowercase",
                "remove_punctuation": action_name == "remove_punctuation",
                "remove_stopwords": action_name == "remove_stopwords",
                "normalize_text": action_name == "normalize_text",
                "no_op": action_name == "no_op"
            }
            
            act = Action(**action_dict)
            res = env.step(act)
            obs = Observation(**res["observation"])
            reward = res["reward"]
            done = res["done"]
            total_reward += reward
            step += 1
            
            print(f"Step {step}: Reward = {reward:.2f}, New Quality = {obs.quality_score:.2f}")

        print(f"Finished {task}. Final Quality Score: {obs.quality_score:.2f}. Total Reward: {total_reward:.2f}")
