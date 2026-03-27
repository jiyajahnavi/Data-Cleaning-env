import csv
import string
import copy
import os
from typing import List, Dict, Any, Optional

from data_cleaning_env.models import Action, Observation

class DataCleaningEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.raw_data: List[Dict[str, Any]] = []
        self.cleaned_data: List[Dict[str, Any]] = []
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.last_action: Optional[Action] = None
        self.initial_issues = 0
        self._load_data()

    def _load_data(self):
        data_path = f"data/task_{self.task_id}.csv"
        # fallback to sample_dataset.csv if not found
        if not os.path.exists(data_path):
            data_path = "data/sample_dataset.csv"

        try:
            with open(data_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.raw_data = [dict(row) for row in reader]
        except Exception:
            self.raw_data = [
                {"id": "1", "text": "HELLO WORLD!!!", "value": ""},
                {"id": "2", "text": "This is a TEST.", "value": "10"},
            ]
        self.cleaned_data = copy.deepcopy(self.raw_data)
        self.initial_issues = self._count_issues()

    def _count_issues(self) -> int:
        if not self.cleaned_data:
            return 0
        issues = 0
        
        # Nulls
        issues += sum(1 for row in self.cleaned_data for v in row.values() if not v)
        
        # Duplicates
        unique_rows = {tuple(sorted(row.items())) for row in self.cleaned_data}
        issues += len(self.cleaned_data) - len(unique_rows)
        
        # Text issues
        stopwords = {"a", "an", "the", "this", "that", "is", "are", "here", "in"}
        for row in self.cleaned_data:
            text = row.get("text", "")
            if text:
                if any(c.isupper() for c in text): issues += 1
                if any(c in string.punctuation for c in text): issues += 1
                words = text.split()
                if any(w.lower() in stopwords for w in words): issues += 1
                if "  " in text or text != text.strip(): issues += 1
                
        return issues
        
    def _compute_quality(self) -> float:
        current_issues = self._count_issues()
        if self.initial_issues == 0:
            return 1.0
        score = 1.0 - (current_issues / self.initial_issues)
        return max(0.0, score)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id:
            self.task_id = task_id
        self._load_data()
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.last_action = None

        return Observation(
            raw_data=self.raw_data,
            cleaned_data=self.cleaned_data,
            quality_score=self._compute_quality(),
            step_count=self.step_count,
        )

    def step(self, action: Action):
        self.step_count += 1
        self.last_action = action

        old_quality = self._compute_quality()
        changed = False

        if action.remove_nulls:
            old_len = len(self.cleaned_data)
            self.cleaned_data = [row for row in self.cleaned_data if all(v for v in row.values())]
            if len(self.cleaned_data) < old_len: changed = True

        if action.fill_missing:
            for row in self.cleaned_data:
                for k, v in row.items():
                    if not v:
                        row[k] = "UNKNOWN"
                        changed = True

        if action.remove_duplicates:
            seen = set()
            new_data = []
            for row in self.cleaned_data:
                t = tuple(sorted(row.items()))
                if t not in seen:
                    seen.add(t)
                    new_data.append(row)
            if len(new_data) != len(self.cleaned_data): changed = True
            self.cleaned_data = new_data

        if action.lowercase:
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    new_text = row["text"].lower()
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        if action.remove_punctuation:
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    new_text = row["text"].translate(str.maketrans("", "", string.punctuation))
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        if action.remove_stopwords:
            stopwords = {"a", "an", "the", "this", "that", "is", "are", "here", "in"}
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    words = row["text"].split()
                    new_words = [w for w in words if w.lower() not in stopwords]
                    new_text = " ".join(new_words)
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        if action.normalize_text:
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    new_text = " ".join(row["text"].split()).strip()
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        if action.no_op:
            pass

        new_quality = self._compute_quality()
        reward = 0.0

        if new_quality > old_quality:
            reward = new_quality - old_quality  # Meaningful reward function scaling with progress
        elif changed and new_quality <= old_quality:
            reward = -0.1 # Penalize undesirable or redundant edits
        elif not changed and not action.no_op:
            reward = -0.1
            
        done = new_quality >= 1.0 or self.step_count >= 10

        if new_quality >= 1.0 and old_quality < 1.0:
            reward += 1.0 # task fully complete bonus

        self.cumulative_reward += reward

        obs = Observation(
            raw_data=self.raw_data,
            cleaned_data=self.cleaned_data,
            quality_score=new_quality,
            step_count=self.step_count,
        )
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
        }

    def state(self):
        return {
            "step_count": self.step_count,
            "last_action": self.last_action.model_dump() if self.last_action else None,
            "cumulative_reward": self.cumulative_reward,
            "quality_score": self._compute_quality(),
        }