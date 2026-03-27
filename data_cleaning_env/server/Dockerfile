FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY data_cleaning_env /app/data_cleaning_env
COPY data /app/data

RUN pip install --no-cache-dir fastapi uvicorn pydantic requests

EXPOSE 8000

ENV PYTHONPATH=/app
CMD ["uvicorn", "data_cleaning_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
