import requests
from data_cleaning_env.models import Action, Observation

class DataCleaningEnv:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def reset(self) -> Observation:
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return Observation(**response.json())

    def step(self, action: Action):
        response = requests.post(f"{self.base_url}/step", json=action.model_dump())
        response.raise_for_status()
        return response.json()

    def state(self):
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    client = DataCleaningEnv()
    obs = client.reset()
    print("Initial Quality:", obs.quality_score)
    result = client.step(Action(lowercase=True))
    print("Reward:", result["reward"], "Done:", result["done"])
    print("State:", client.state())
