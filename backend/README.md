# Resume Analyzer — Backend

FastAPI backend for the AI-powered Resume Analyzer & Job Matcher.

## Local Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs will be available at: http://localhost:8000/docs

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Status check |
| POST | `/analyze/text` | Analyze resume + JD as plain text (JSON) |
| POST | `/analyze/pdf` | Upload PDF resume + .txt JD |
| POST | `/extract-text` | Extract text from PDF |

## Example: /analyze/text

```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Experienced Python developer with FastAPI, React, AWS...",
    "job_description": "Looking for a Python backend engineer with FastAPI, Docker, AWS..."
  }'
```

## Response Shape

```json
{
  "match_score": 72.4,
  "resume_skills": ["python", "fastapi", "react", "aws"],
  "jd_skills": ["python", "fastapi", "docker", "aws"],
  "matched_skills": ["python", "fastapi", "aws"],
  "missing_skills": ["docker"],
  "top_jd_keywords": ["backend", "engineer", "cloud", "deployment"],
  "missing_keywords": ["deployment", "cloud"],
  "suggestions": ["Add Docker to your skills section.", "..."],
  "resume_word_count": 120,
  "jd_word_count": 95
}
```

## AWS EC2 Deployment

```bash
# On your EC2 instance (Ubuntu 22.04):
sudo apt update && sudo apt install python3-pip -y
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# For production, use:
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

Make sure port 8000 is open in your EC2 Security Group inbound rules.
