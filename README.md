# DataCleaningEnv

A real-world task environment for evaluating autonomous agents on interactive data cleaning tasks. This environment complies fully with the OpenEnv specification.

## Description & Motivation
In the real world, datasets are rarely clean. Data Scientists and ML Engineers spend a vast amount of their time standardizing, cleaning, and formatting unstructured or messy tabular data. This environment simulates this iterative process. An agent interacts with the `DataCleaningEnv`, observing the current state of a noisy dataset, and issuing cleaning operations one step at a time, receiving rewards for improving the dataset's overall quality score.

## Spaces

### Observation Space
The observation space is represented by a JSON payload (Pydantic `Observation` model) containing:
- `raw_data`: The initial, uncleaned dataset.
- `cleaned_data`: The current state of the dataset after applying actions.
- `quality_score`: A float from `0.0` (completely unstructured) to `1.0` (perfectly clean).
- `step_count`: Current step in the episode.

### Action Space
The action space consists of 8 discrete operations in the `Action` model:
- `remove_nulls`: Drops rows with missing values.
- `fill_missing`: Fills missing values with a placeholder.
- `remove_duplicates`: Retains only unique rows.
- `lowercase`: Converts text to lowercase.
- `remove_punctuation`: Strips punctuation from text.
- `remove_stopwords`: Removes common stopwords (e.g., 'the', 'a', 'is').
- `normalize_text`: Standardizes whitespace.
- `no_op`: Takes no action.

## Tasks and Difficulty
The environment defines 3 programmatic tasks with a deterministic grader (`1.0` signifies completion):

1. **Easy (`easy`)**: Minor inconsistencies. The agent simply needs to remove uppercase text and missing values.
2. **Medium (`medium`)**: A moderately messy dataset containing duplicates, punctuation, and missing values.
3. **Hard (`hard`)**: Extremely unstructured. Requires chaining operations perfectly to normalize spaces, eliminate stopwords, remove duplicates, and lowercase text.

## Setup Instructions

### Local Execution (FastAPI + Uvicorn)
1. Install dependencies:
   ```bash
   pip install -e .
   ```
2. Run the environment server:
   ```bash
   uvicorn data_cleaning_env.server.app:app --reload --host 0.0.0.0 --port 8000
   ```
   
### Containerized (Docker)
The environment comes with a working Dockerfile.
1. Build the image:
   ```bash
   docker build -t datacleaningenv .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 datacleaningenv
   ```

### Hugging Face Spaces
This environment can be deployed directly to Hugging Face Spaces by creating a new generic Docker Space, tagging it with `openenv`, and pushing this repository! The provided Dockerfile is fully compatible.

## Baseline Results
The repository includes `baseline.py`, which uses the Gemini API (e.g., `gemini-1.5-flash`) to perform the data cleaning task iteratively. 

Run the baseline:
```bash
export GEMINI_API_KEY="AIza..."
python baseline.py
```
**Expected Scores**:
- `easy`: Quality 1.0 
- `medium`: Quality 1.0 
- `hard`: Quality 1.0 
