import csv
import string
import copy
from typing import List, Dict, Any, Optional

from data_cleaning_env.models import Action, Observation


class DataCleaningEnv:
    def __init__(self, data_path: str = "data/sample_dataset.csv"):
        self.data_path = data_path
        self.raw_data: List[Dict[str, Any]] = []
        self.cleaned_data: List[Dict[str, Any]] = []
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.last_action: Optional[Action] = None
        self._load_data()

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    def _load_data(self):
        try:
            with open(self.data_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.raw_data = [dict(row) for row in reader]
        except Exception:
            # fallback sample
            self.raw_data = [
                {"id": "1", "text": "HELLO WORLD!!!", "value": ""},
                {"id": "2", "text": "This is a TEST.", "value": "10"},
            ]

        self.cleaned_data = copy.deepcopy(self.raw_data)

    # -------------------------------
    # QUALITY SCORE
    # -------------------------------
    def _compute_quality(self) -> float:
        if not self.cleaned_data:
            return 0.0

        null_count = sum(
            1 for row in self.cleaned_data for v in row.values() if not v
        )

        dup_count = len(self.cleaned_data) - len(
            {tuple(sorted(row.items())) for row in self.cleaned_data}
        )

        score = 100.0 - (null_count * 5.0) - (dup_count * 10.0)
        return max(0.0, score)

    # -------------------------------
    # RESET
    # -------------------------------
    def reset(self, data: Optional[List[Dict[str, Any]]] = None) -> Observation:
        if data:
            self.raw_data = data
            self.cleaned_data = copy.deepcopy(data)
        else:
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

    # -------------------------------
    # STEP
    # -------------------------------
    def step(self, action: Action):
        self.step_count += 1
        self.last_action = action

        old_quality = self._compute_quality()
        changed = False

        # ---- REMOVE NULLS ----
        if action.remove_nulls:
            old_len = len(self.cleaned_data)
            self.cleaned_data = [
                row for row in self.cleaned_data if all(v for v in row.values())
            ]
            if len(self.cleaned_data) < old_len:
                changed = True

        # ---- FILL MISSING ----
        if action.fill_missing:
            for row in self.cleaned_data:
                for k, v in row.items():
                    if not v:
                        row[k] = "UNKNOWN"
                        changed = True

        # ---- REMOVE DUPLICATES ----
        if action.remove_duplicates:
            seen = set()
            new_data = []
            for row in self.cleaned_data:
                t = tuple(sorted(row.items()))
                if t not in seen:
                    seen.add(t)
                    new_data.append(row)
            if len(new_data) != len(self.cleaned_data):
                changed = True
            self.cleaned_data = new_data

        # ---- LOWERCASE ----
        if action.lowercase:
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    new_text = row["text"].lower()
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        # ---- REMOVE PUNCTUATION ----
        if action.remove_punctuation:
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    new_text = row["text"].translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        # ---- REMOVE STOPWORDS ----
        if action.remove_stopwords:
            stopwords = {"is", "a", "the", "this", "are", "here"}
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    words = row["text"].split()
                    new_words = [w for w in words if w.lower() not in stopwords]
                    new_text = " ".join(new_words)
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        # ---- NORMALIZE TEXT ----
        if action.normalize_text:
            for row in self.cleaned_data:
                if "text" in row and row["text"]:
                    new_text = " ".join(row["text"].split())
                    if new_text != row["text"]:
                        row["text"] = new_text
                        changed = True

        # ---- NO-OP ----
        if action.no_op:
            pass

        # -------------------------------
        # REWARD
        # -------------------------------
        new_quality = self._compute_quality()
        reward = 0.0

        if new_quality > old_quality:
            reward = 1.0
        elif changed and new_quality == old_quality:
            reward = 0.5
        elif not changed and not action.no_op:
            reward = -1.0

        if new_quality == 100.0 and old_quality < 100.0:
            reward += 5.0

        self.cumulative_reward += reward

        done = new_quality == 100.0 or self.step_count >= 20

        observation = Observation(
            raw_data=self.raw_data,
            cleaned_data=self.cleaned_data,
            quality_score=new_quality,
            step_count=self.step_count,
        )

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
        }

    # -------------------------------
    # STATE
    # -------------------------------
    def state(self):
        return {
            "step_count": self.step_count,
            "last_action": self.last_action.model_dump()
            if self.last_action
            else None,
            "cumulative_reward": self.cumulative_reward,
            "quality_score": self._compute_quality(),
        }