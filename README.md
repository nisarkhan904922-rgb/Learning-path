# 📄 AI Resume ATS Checker

A Streamlit app that scores your resume for Applicant Tracking Systems (ATS) and gives specific, prioritized improvements. Powered by the Groq API.

## Features
- Upload a resume as **PDF or DOCX**
- **ATS score (0-100)**, a blend of 70% AI evaluation and 30% rule-based checks
- Score breakdown: formatting, keywords, impact, clarity, and job match
- Strengths, weaknesses, and **prioritized improvements**
- **Missing keywords** and suggested **bullet rewrites**
- Optional **job description matching**
- Download the report as a text file
- View the extracted text to see what an ATS actually reads

## Project structure
```
.
├── app.py
├── requirements.txt
└── README.md
```

## Run locally
1. Get a free API key from [console.groq.com/keys](https://console.groq.com/keys).
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Paste your Groq API key in the sidebar, or set it beforehand:
   ```bash
   export GROQ_API_KEY="your_key_here"      # Windows PowerShell: $env:GROQ_API_KEY="your_key_here"
   ```

## Deploy on Streamlit Community Cloud
1. Push this project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

> Never commit your API key to GitHub. Use Streamlit Secrets or environment variables.

## Models
Default: `llama-3.3-70b-versatile` (best quality). Alternative: `llama-3.1-8b-instant` (faster). To use other models, edit `DEFAULT_MODELS` in `app.py` (see Groq's model list).

## Limitations
- Scanned/image-only PDFs can't be read (an ATS can't read them either).
- The score is an estimate. Real ATS systems differ, so treat it as guidance, not a guarantee.
- Resume text is sent to Groq for analysis; don't upload data you aren't comfortable sharing.

## Privacy
The app does not store uploaded files. Text is processed in memory and sent to the Groq API.
