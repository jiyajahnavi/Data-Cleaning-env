# DataCleaningEnv

A real-world task environment for evaluating autonomous agents on interactive **data cleaning tasks**. Fully compliant with the OpenEnv specification, this environment simulates iterative cleaning of messy tabular data.

---

## 📖 Description & Motivation
Datasets in the real world are rarely clean. Data Scientists and ML Engineers spend a significant portion of their time standardizing, cleaning, and formatting unstructured or messy data.  

`DataCleaningEnv` provides a simulated environment where an agent observes a noisy dataset, takes actions to clean it step by step, and receives rewards based on the improvement of the dataset's overall quality.

---

## 🧩 Observation & Action Spaces

### Observation Space
The environment returns a JSON payload (Pydantic `Observation` model) containing:
- `raw_data`: The initial, unclean dataset.
- `cleaned_data`: Current state of the dataset after applied actions.
- `quality_score`: Float between `0.0` (completely unstructured) and `1.0` (perfectly clean).
- `step_count`: Current step number.

### Action Space
The agent can perform **8 discrete actions** (`Action` model):
- `remove_nulls` – Drop rows with missing values.
- `fill_missing` – Fill missing values with placeholders.
- `remove_duplicates` – Keep only unique rows.
- `lowercase` – Convert text to lowercase.
- `remove_punctuation` – Remove punctuation characters.
- `remove_stopwords` – Remove common stopwords (`the`, `a`, `is`, etc.).
- `normalize_text` – Standardize whitespace and formatting.
- `no_op` – Take no action.

---

## 🎯 Tasks and Difficulty
The environment defines **3 tasks**, each with a deterministic grader (`1.0` = fully solved):

1. **Easy (`easy`)** – Minor inconsistencies; mainly requires lowercasing and removing nulls.
2. **Medium (`medium`)** – Moderately messy with duplicates, punctuation, and missing values.
3. **Hard (`hard`)** – Extremely unstructured; requires chaining multiple operations to clean text, remove duplicates, normalize spaces, and remove stopwords.

---

## ⚙️ Setup Instructions

### Python Environment
- Requires **Python 3.11+**.
- Recommended: Create a virtual environment:

```powershell
python -m venv env
.\env\Scripts\activate
````

* Install dependencies:

```powershell
pip install -e .
```

---

### Run Locally (FastAPI + Uvicorn)

```powershell
uvicorn data_cleaning_env.server.app:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

### Containerized Setup (Docker)

**Note (Windows users):** Ensure Docker Desktop is running with WSL 2 backend enabled.

1. Build the image:

```powershell
docker build -t datacleaningenv .
```

2. Run the container:

```powershell
docker run -p 8000:8000 datacleaningenv
```

---

### Deploy to Hugging Face Spaces

* Create a new **Generic Docker Space**.
* Push your repository. The provided Dockerfile is fully compatible.

---

## 🏆 Baseline Results

The repository includes `baseline.py`, which uses the Gemini API (`gemini-1.5-flash`) to perform iterative data cleaning.

**Run the baseline (Windows PowerShell):**

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
python baseline.py
```

**Expected Quality Scores:**

* Easy: 1.0
* Medium: 1.0
* Hard: 1.0

---

## 📝 License

This project is released under the **MIT License**.

```
