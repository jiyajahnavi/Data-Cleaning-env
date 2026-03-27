from pydantic import BaseModel
from typing import List, Dict, Any

class Action(BaseModel):
    remove_nulls: bool = False
    fill_missing: bool = False
    remove_duplicates: bool = False
    lowercase: bool = False
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    normalize_text: bool = False
    no_op: bool = False

class Observation(BaseModel):
    raw_data: List[Dict[str, Any]]
    cleaned_data: List[Dict[str, Any]]
    quality_score: float
    step_count: int
