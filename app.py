"""
AI Resume ATS Checker
---------------------
Upload a resume (PDF or DOCX) and get:
  * an ATS score (blend of rule-based checks + AI evaluation via Groq)
  * strengths, weaknesses, missing keywords
  * prioritized, actionable improvements
  * optional job-description matching
"""

import io
import json
import os
import re

import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
MAX_RESUME_CHARS = 12000   # keep prompt size safe for the model
MAX_JD_CHARS = 5000
AI_WEIGHT = 0.7            # final score = 70% AI + 30% rule-based
RULE_WEIGHT = 0.3

ACTION_VERBS = {
    "led", "managed", "developed", "built", "designed", "created", "implemented",
    "improved", "increased", "reduced", "launched", "delivered", "optimized",
    "automated", "analyzed", "coordinated", "achieved", "established", "drove",
    "streamlined", "mentored", "architected", "deployed", "engineered", "owned",
}

SECTION_KEYWORDS = {
    "Summary / Objective": ["summary", "objective", "profile", "about me"],
    "Experience": ["experience", "employment", "work history", "internship"],
    "Education": ["education", "academic", "qualification"],
    "Skills": ["skills", "technologies", "technical skills", "competencies"],
    "Projects": ["projects", "project work"],
}


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password-protected. Please upload an unlocked copy.")
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    # Resumes often put content in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())
    return "\n".join(parts).strip()


def extract_resume_text(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")


# --------------------------------------------------------------------------- #
# Rule-based checks (deterministic, no API needed)
# --------------------------------------------------------------------------- #
def count_metrics(text: str) -> int:
    """Count numeric achievements (%, $, 2M+, 500+, team sizes) while ignoring years, phones and URLs."""
    cleaned = re.sub(r"https?://\S+|\S+@\S+|\b\S+\.(?:com|org|io|net)\S*", " ", text)  # urls / emails
    cleaned = re.sub(r"\+?\d[\d\s().-]{8,}\d", " ", cleaned)                              # phone numbers
    cleaned = re.sub(r"\b(?:19|20)\d{2}\b", " ", cleaned)                                   # years
    return len(re.findall(r"\$\s?\d[\d,.]*|\d[\d,.]*\s?(?:%|[kKmMbB]\b|\+|x\b)|\b\d[\d,.]*", cleaned))


def run_rule_based_checks(text: str) -> dict:
    """Return a list of pass/fail checks and a 0-100 score."""
    lower = text.lower()
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    checks = []

    def add(label, passed, points, tip):
        checks.append({"label": label, "passed": bool(passed), "points": points, "tip": tip})

    # Contact info
    has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text) is not None
    has_phone = re.search(r"(\+?\d[\d\s().-]{8,}\d)", text) is not None
    has_link = re.search(r"(linkedin\.com|github\.com|portfolio|https?://)", lower) is not None
    add("Email address found", has_email, 10, "Add a professional email address at the top.")
    add("Phone number found", has_phone, 5, "Add a phone number in the header.")
    add("LinkedIn / GitHub / portfolio link", has_link, 5, "Add a LinkedIn or GitHub URL.")

    # Sections
    for label, keys in SECTION_KEYWORDS.items():
        found = any(re.search(rf"(^|\n)\s*{re.escape(k)}\b", lower) for k in keys)
        points = 10 if label in ("Experience", "Education", "Skills") else 4
        add(f"'{label}' section", found, points,
            f"Add a clearly labeled '{label.split(' /')[0]}' heading (ATS parsers look for standard headings).")

    # Length
    good_length = 300 <= word_count <= 1000
    add(f"Length is reasonable ({word_count} words)", good_length, 10,
        "Aim for roughly 400-800 words (1-2 pages). Too short lacks detail; too long gets skimmed.")

    # Quantified achievements
    metrics = count_metrics(text)
    add(f"Quantified achievements ({metrics} found)", metrics >= 3, 10,
        "Add numbers to your bullets: percentages, revenue, users, time saved.")

    # Action verbs
    verb_hits = sum(1 for w in words if w.lower() in ACTION_VERBS)
    add(f"Strong action verbs ({verb_hits} found)", verb_hits >= 5, 6,
        "Start bullets with verbs like 'Built', 'Led', 'Optimized', 'Reduced'.")

    # Bullets
    bullet_lines = len(re.findall(r"(^|\n)\s*[-•*▪●◦–]\s+", text))
    add("Uses bullet points", bullet_lines >= 5, 5,
        "Use bullet points for achievements instead of long paragraphs.")

    # Date patterns
    has_dates = re.search(r"(19|20)\d{2}", text) is not None
    add("Dates present", has_dates, 5, "Include dates (e.g. 'Jan 2022 - Mar 2024') for each role.")

    total = sum(c["points"] for c in checks)
    earned = sum(c["points"] for c in checks if c["passed"])
    score = round(100 * earned / total) if total else 0
    return {"score": score, "checks": checks, "word_count": word_count}


# --------------------------------------------------------------------------- #
# Groq / LLM
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and professional resume coach.
You evaluate resumes strictly and honestly. Do not inflate scores.

Return ONLY a valid JSON object with exactly this structure:
{
  "overall_score": <integer 0-100>,
  "section_scores": {
    "formatting_and_structure": <integer 0-100>,
    "keywords_and_skills": <integer 0-100>,
    "impact_and_achievements": <integer 0-100>,
    "clarity_and_grammar": <integer 0-100>,
    "job_match": <integer 0-100 or null if no job description was provided>
  },
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<string>", ...],
  "weaknesses": ["<string>", ...],
  "missing_keywords": ["<string>", ...],
  "improvements": [
    {"priority": "High" | "Medium" | "Low", "issue": "<what is wrong>", "fix": "<specific, actionable fix>"}
  ],
  "bullet_rewrites": [
    {"original": "<a weak bullet from the resume>", "improved": "<stronger rewrite with metrics/action verb>"}
  ]
}

Rules:
- Base everything ONLY on the resume text provided. Never invent experience, companies, or numbers
  in the resume. In "bullet_rewrites", use placeholders like [X%] where a metric is unknown.
- Give 3-6 strengths, 3-6 weaknesses, 5-8 improvements, and 2-4 bullet rewrites.
- If a job description is provided, missing_keywords must be keywords from the job description
  that are absent from the resume. Otherwise, list commonly expected keywords for the candidate's field.
- Output JSON only. No markdown, no commentary."""


def build_user_prompt(resume_text: str, job_description: str) -> str:
    prompt = f"RESUME TEXT:\n\"\"\"\n{resume_text[:MAX_RESUME_CHARS]}\n\"\"\"\n"
    if job_description.strip():
        prompt += f"\nTARGET JOB DESCRIPTION:\n\"\"\"\n{job_description.strip()[:MAX_JD_CHARS]}\n\"\"\"\n"
    else:
        prompt += "\nNo job description provided. Set section_scores.job_match to null.\n"
    return prompt


def parse_json_response(raw: str) -> dict:
    """Parse model output into a dict, tolerating markdown fences / stray text."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise ValueError("The AI returned an unreadable response. Please try again.")


def _clamp(value, default=0):
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _as_list(value):
    return value if isinstance(value, list) else []


def normalize_analysis(data: dict) -> dict:
    """Make sure every field exists and has the right type."""
    sections = data.get("section_scores") if isinstance(data.get("section_scores"), dict) else {}
    job_match = sections.get("job_match")
    return {
        "overall_score": _clamp(data.get("overall_score")),
        "section_scores": {
            "Formatting & Structure": _clamp(sections.get("formatting_and_structure")),
            "Keywords & Skills": _clamp(sections.get("keywords_and_skills")),
            "Impact & Achievements": _clamp(sections.get("impact_and_achievements")),
            "Clarity & Grammar": _clamp(sections.get("clarity_and_grammar")),
            "Job Match": None if job_match is None else _clamp(job_match),
        },
        "summary": str(data.get("summary", "")).strip(),
        "strengths": [str(x) for x in _as_list(data.get("strengths"))],
        "weaknesses": [str(x) for x in _as_list(data.get("weaknesses"))],
        "missing_keywords": [str(x) for x in _as_list(data.get("missing_keywords"))],
        "improvements": [
            {
                "priority": str(i.get("priority", "Medium")).capitalize(),
                "issue": str(i.get("issue", "")),
                "fix": str(i.get("fix", "")),
            }
            for i in _as_list(data.get("improvements")) if isinstance(i, dict)
        ],
        "bullet_rewrites": [
            {"original": str(b.get("original", "")), "improved": str(b.get("improved", ""))}
            for b in _as_list(data.get("bullet_rewrites")) if isinstance(b, dict)
        ],
    }


def analyze_with_groq(api_key: str, model: str, resume_text: str, job_description: str) -> dict:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(resume_text, job_description)},
        ],
        temperature=0.2,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return normalize_analysis(parse_json_response(raw))


def combine_scores(ai_score: int, rule_score: int) -> int:
    return round(AI_WEIGHT * ai_score + RULE_WEIGHT * rule_score)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def score_label(score: int) -> str:
    if score >= 80:
        return "🟢 Excellent"
    if score >= 65:
        return "🟡 Good, but can improve"
    if score >= 50:
        return "🟠 Needs work"
    return "🔴 Needs major improvement"


def get_api_key() -> str:
    """Priority: Streamlit secrets -> environment variable -> sidebar input."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets.toml present
    return os.environ.get("GROQ_API_KEY", "")


def render_results(analysis: dict, rules: dict):
    final_score = combine_scores(analysis["overall_score"], rules["score"])

    st.divider()
    st.header("📊 Your Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall ATS Score", f"{final_score}/100")
    c2.metric("AI Evaluation", f"{analysis['overall_score']}/100")
    c3.metric("Rule-based Checks", f"{rules['score']}/100")
    st.progress(final_score / 100)
    st.subheader(score_label(final_score))
    if analysis["summary"]:
        st.info(analysis["summary"])

    # Section scores
    st.subheader("Score Breakdown")
    visible = {k: v for k, v in analysis["section_scores"].items() if v is not None}
    cols = st.columns(len(visible))
    for col, (name, val) in zip(cols, visible.items()):
        col.metric(name, f"{val}/100")

    # Strengths / weaknesses
    left, right = st.columns(2)
    with left:
        st.subheader("✅ Strengths")
        for s in analysis["strengths"] or ["No strengths reported."]:
            st.markdown(f"- {s}")
    with right:
        st.subheader("⚠️ Weaknesses")
        for w in analysis["weaknesses"] or ["No weaknesses reported."]:
            st.markdown(f"- {w}")

    # Improvements
    st.subheader("🛠️ Recommended Improvements")
    order = {"High": 0, "Medium": 1, "Low": 2}
    icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    for imp in sorted(analysis["improvements"], key=lambda x: order.get(x["priority"], 1)):
        with st.expander(f"{icons.get(imp['priority'], '🟠')} {imp['priority']} priority: {imp['issue']}"):
            st.markdown(f"**Fix:** {imp['fix']}")

    # Keywords
    if analysis["missing_keywords"]:
        st.subheader("🔑 Missing Keywords")
        st.write(", ".join(f"`{k}`" for k in analysis["missing_keywords"]))

    # Bullet rewrites
    if analysis["bullet_rewrites"]:
        st.subheader("✍️ Suggested Bullet Rewrites")
        for b in analysis["bullet_rewrites"]:
            st.markdown(f"**Before:** {b['original']}")
            st.markdown(f"**After:** {b['improved']}")
            st.write("")
        st.caption("Replace placeholders like [X%] with your real numbers. Never claim results you did not achieve.")

    # Rule-based details
    with st.expander("📋 Detailed rule-based checklist"):
        for c in rules["checks"]:
            if c["passed"]:
                st.markdown(f"✅ {c['label']}")
            else:
                st.markdown(f"❌ {c['label']} — _{c['tip']}_")

    # Download report
    report = build_text_report(final_score, analysis, rules)
    st.download_button("⬇️ Download report (.txt)", report, file_name="ats_report.txt", mime="text/plain")


def build_text_report(final_score: int, analysis: dict, rules: dict) -> str:
    lines = [
        "RESUME ATS REPORT",
        "=" * 40,
        f"Overall ATS Score: {final_score}/100",
        f"AI Evaluation: {analysis['overall_score']}/100",
        f"Rule-based Checks: {rules['score']}/100",
        "",
        "SUMMARY",
        analysis["summary"],
        "",
        "SCORE BREAKDOWN",
    ]
    for k, v in analysis["section_scores"].items():
        if v is not None:
            lines.append(f"- {k}: {v}/100")
    lines += ["", "STRENGTHS"] + [f"- {s}" for s in analysis["strengths"]]
    lines += ["", "WEAKNESSES"] + [f"- {w}" for w in analysis["weaknesses"]]
    lines += ["", "IMPROVEMENTS"]
    for i in analysis["improvements"]:
        lines.append(f"- [{i['priority']}] {i['issue']}\n    Fix: {i['fix']}")
    if analysis["missing_keywords"]:
        lines += ["", "MISSING KEYWORDS", ", ".join(analysis["missing_keywords"])]
    if analysis["bullet_rewrites"]:
        lines += ["", "BULLET REWRITES"]
        for b in analysis["bullet_rewrites"]:
            lines.append(f"- Before: {b['original']}\n  After:  {b['improved']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="AI Resume ATS Checker", page_icon="📄", layout="wide")
    st.title("📄 AI Resume ATS Checker")
    st.caption("Upload your resume, get an ATS score and specific ways to improve it. Powered by Groq.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = get_api_key()
        if api_key:
            st.success("Groq API key loaded.")
        else:
            api_key = st.text_input(
                "Groq API key", type="password",
                help="Get a free key at console.groq.com/keys",
            )
        model = st.selectbox("Model", DEFAULT_MODELS, index=0)
        st.markdown("---")
        st.caption(
            "🔒 Your resume is sent to Groq for analysis and is not stored by this app. "
            "Avoid uploading sensitive personal data you are not comfortable sharing."
        )

    # Inputs
    col_a, col_b = st.columns([1, 1])
    with col_a:
        uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    with col_b:
        job_description = st.text_area(
            "Target job description (optional)",
            height=170,
            placeholder="Paste the job description here to get a job-match score and tailored keywords...",
        )

    if st.button("🚀 Analyze Resume", type="primary", disabled=uploaded is None):
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()

        try:
            with st.spinner("Reading your resume..."):
                text = extract_resume_text(uploaded.name, uploaded.getvalue())
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception:
            st.error("Could not read this file. It may be corrupted. Try re-exporting it as a PDF or DOCX.")
            st.stop()

        if len(text.split()) < 50:
            st.error(
                "Very little text could be extracted. If your resume is a scanned image, "
                "an ATS cannot read it either. Please upload a text-based PDF or DOCX."
            )
            st.stop()

        rules = run_rule_based_checks(text)

        try:
            with st.spinner("Analyzing with AI... this takes a few seconds."):
                analysis = analyze_with_groq(api_key, model, text, job_description)
        except Exception as e:
            msg = str(e)
            if "401" in msg or "invalid_api_key" in msg.lower():
                st.error("Invalid Groq API key. Please check it and try again.")
            elif "429" in msg or "rate" in msg.lower():
                st.error("Groq rate limit reached. Wait a minute and try again.")
            elif "model" in msg.lower() and ("not found" in msg.lower() or "decommissioned" in msg.lower()):
                st.error("This model is unavailable. Pick a different model in the sidebar.")
            else:
                st.error(f"AI analysis failed: {msg}")
            st.stop()

        render_results(analysis, rules)

        with st.expander("🔍 View extracted resume text (what an ATS sees)"):
            st.text(text[:MAX_RESUME_CHARS])


if __name__ == "__main__":
    main()
