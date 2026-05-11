from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pdfplumber
import re
import io
import math
from collections import Counter

app = FastAPI(title="Resume Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Bundled stopwords (no NLTK download needed) ───────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it",
    "its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other","some",
    "such","no","nor","not","only","own","same","so","than","too","very","s",
    "t","can","will","just","don","should","now","d","ll","m","o","re","ve",
    "y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
    "use","used","using","work","worked","working","also","within","across",
    "well","new","one","two","three","get","make","set","using","based","via",
    "per","etc","key","lead","led","help","build","built","develop","developed",
    "support","ensure","provide","create","manage","strong","ability","good",
    "experience","years","year","including","include","required","preferred",
    "knowledge","understanding","familiar","familiarity","hands","proven",
    "excellent","effective","written","verbal","communication","skills","skill",
    "team","teams","cross","functional","multiple","various","high","large",
    "level","levels","looking","seeking","join","role","position","candidate",
    "responsible","responsibilities","opportunity","opportunities","must","able",
    "plus","bonus","nice","have","will","would","should","could","may","might"
}

# ─── Tech skill keyword list for extraction ─────────────────────────────────
TECH_SKILLS = [
    "python","java","javascript","typescript","html","css","react","angular","vue",
    "nodejs","node.js","fastapi","flask","django","spring","express","nextjs","next.js",
    "mysql","postgresql","mongodb","redis","sqlite","firebase","dynamodb",
    "aws","azure","gcp","docker","kubernetes","terraform","ci/cd","github","gitlab",
    "git","linux","bash","rest","api","graphql","microservices","kafka","rabbitmq",
    "machine learning","deep learning","nlp","tensorflow","pytorch","scikit-learn",
    "pandas","numpy","opencv","bert","transformers","llm","generative ai",
    "figma","ui/ux","agile","scrum","jira","data structures","algorithms",
    "oop","solid","tdd","jwt","oauth","sql","nosql","pandas","spark","hadoop",
    "elasticsearch","selenium","jest","pytest","postman","swagger","openapi",
    "c","c++","c#","go","golang","rust","php","ruby","kotlin","swift","dart","flutter",
    "sass","tailwind","bootstrap","material ui","mui","redux","graphql","websocket",
    "multithreading","concurrency","system design","devops","mlops","data engineering",
    "computer vision","reinforcement learning","etl","data pipeline","airflow",
    "tableau","powerbi","excel","r","matlab"
]


# ─── Utility Functions ───────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def clean_text(text: str) -> str:
    """Lowercase, remove special characters, normalize whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r'\b[a-z][a-z0-9\+\#\.]*\b', text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def extract_skills(text: str) -> list[str]:
    """Extract recognized tech skills from text."""
    text_lower = text.lower()
    found = []
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(dict.fromkeys(found))  # preserve order, deduplicate


def compute_tfidf_vectors(doc1_tokens: list[str], doc2_tokens: list[str]):
    """Compute TF-IDF vectors for two documents and return cosine similarity.
    Uses smooth IDF: log(1 + N/df) + 1 so words shared across both docs
    still contribute a weight of 1.0 rather than 0.
    """
    vocab = list(set(doc1_tokens + doc2_tokens))
    if not vocab:
        return 0.0, {}, {}

    set1, set2 = set(doc1_tokens), set(doc2_tokens)
    N = 2  # total documents

    def tf(tokens):
        count = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {w: count[w] / total for w in count}

    def smooth_idf(word):
        """Smooth IDF: log(1 + N/df) + 1 — never zero for any word."""
        in_doc1 = 1 if word in set1 else 0
        in_doc2 = 1 if word in set2 else 0
        df = in_doc1 + in_doc2
        return math.log(1 + N / df) + 1 if df > 0 else 0

    def tfidf_vector(tf_map):
        return {word: tf_map.get(word, 0) * smooth_idf(word) for word in vocab}

    vec1 = tfidf_vector(tf(doc1_tokens))
    vec2 = tfidf_vector(tf(doc2_tokens))

    # Cosine similarity
    dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in vocab)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    similarity = dot / (mag1 * mag2) if mag1 and mag2 else 0.0
    return round(similarity * 100, 1), vec1, vec2


def find_top_jd_keywords(jd_tokens: list[str], top_n: int = 20) -> list[str]:
    """Return the most frequent meaningful keywords from a JD."""
    freq = Counter(jd_tokens)
    # Filter to words with length >= 4 for relevance
    filtered = {w: c for w, c in freq.items() if len(w) >= 4}
    return [w for w, _ in sorted(filtered.items(), key=lambda x: -x[1])[:top_n]]


def generate_suggestions(missing_skills: list[str], missing_keywords: list[str]) -> list[str]:
    """Generate actionable resume improvement suggestions."""
    suggestions = []

    if missing_skills:
        top_missing = missing_skills[:4]
        suggestions.append(
            f"Add these in-demand skills to your skills section: "
            f"{', '.join(s.title() for s in top_missing)}."
        )
        for skill in top_missing[:2]:
            suggestions.append(
                f"Consider adding a project or bullet point that demonstrates your use of {skill.title()}."
            )

    if missing_keywords:
        suggestions.append(
            f"Incorporate these JD keywords naturally into your bullet points: "
            f"{', '.join(missing_keywords[:6])}."
        )

    suggestions.append(
        "Quantify your achievements — add metrics like percentages, team size, or number of users where possible."
    )
    suggestions.append(
        "Mirror the exact terminology from the job description (e.g. if JD says 'REST API', use that exact phrase)."
    )

    return suggestions[:6]


# ─── Request / Response Models ───────────────────────────────────────────────

class AnalyzeTextRequest(BaseModel):
    resume_text: str
    job_description: str


class AnalysisResponse(BaseModel):
    match_score: float
    resume_skills: list[str]
    jd_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    top_jd_keywords: list[str]
    missing_keywords: list[str]
    suggestions: list[str]
    resume_word_count: int
    jd_word_count: int


# ─── Core Analysis Logic ─────────────────────────────────────────────────────

def run_analysis(resume_text: str, job_description: str) -> AnalysisResponse:
    if not resume_text.strip() or not job_description.strip():
        raise HTTPException(status_code=400, detail="Resume and job description cannot be empty.")

    # Tokenize
    resume_tokens = tokenize(clean_text(resume_text))
    jd_tokens = tokenize(clean_text(job_description))

    if not resume_tokens or not jd_tokens:
        raise HTTPException(status_code=400, detail="Could not extract meaningful text.")

    # TF-IDF similarity score
    match_score, _, _ = compute_tfidf_vectors(resume_tokens, jd_tokens)

    # Skill extraction
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    matched_skills = sorted(set(resume_skills) & set(jd_skills))
    missing_skills = sorted(set(jd_skills) - set(resume_skills))

    # Keyword gap analysis
    top_jd_keywords = find_top_jd_keywords(jd_tokens, top_n=20)
    resume_keyword_set = set(resume_tokens)
    missing_keywords = [kw for kw in top_jd_keywords if kw not in resume_keyword_set][:10]

    # Suggestions
    suggestions = generate_suggestions(missing_skills, missing_keywords)

    return AnalysisResponse(
        match_score=match_score,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        top_jd_keywords=top_jd_keywords,
        missing_keywords=missing_keywords,
        suggestions=suggestions,
        resume_word_count=len(resume_tokens),
        jd_word_count=len(jd_tokens),
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Resume Analyzer API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze/pdf", response_model=AnalysisResponse)
async def analyze_pdf(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: UploadFile = File(..., description="Job description as a .txt file"),
):
    """Analyze a PDF resume against a job description text file."""
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")

    resume_bytes = await resume.read()
    jd_bytes = await job_description.read()

    try:
        resume_text = extract_text_from_pdf(resume_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")

    jd_text = jd_bytes.decode("utf-8", errors="ignore")
    return run_analysis(resume_text, jd_text)


@app.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(payload: AnalyzeTextRequest):
    """Analyze resume and job description provided as plain text (JSON body)."""
    return run_analysis(payload.resume_text, payload.job_description)


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """Extract and return raw text from a PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    file_bytes = await file.read()
    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")
    return {"filename": file.filename, "text": text, "word_count": len(text.split())}
