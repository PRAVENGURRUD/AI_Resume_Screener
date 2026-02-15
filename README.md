# AI-Powered Resume Screener

An intelligent resume screening application that matches candidates to job descriptions using NLP and semantic analysis.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## What It Does

| Feature | Description |
|---------|-------------|
| **Resume Parsing** | Extracts skills, experience, education from PDF/DOCX/TXT files |
| **Semantic Matching** | Uses SBERT embeddings to understand skill similarities |
| **Configurable Scoring** | Adjust weights for skills, experience, education, keywords |
| **Batch Processing** | Compare and rank multiple candidates at once |
| **PDF Reports** | Generate downloadable analysis reports |

---

## How to Use

### 1. Choose Analysis Mode (Sidebar)
- **Single Resume**: Detailed analysis of one candidate
- **Batch Mode**: Rank multiple resumes against a job description

### 2. Set Scoring Weights (Sidebar)
| Weight | Default | What It Measures |
|--------|---------|------------------|
| Skill Overlap | 50% | How many required skills the candidate has |
| Experience | 30% | Years of experience alignment |
| Education | 10% | Degree and certification match |
| Keywords | 10% | Job description keyword coverage |

### 3. Enter Job Description
- Upload a file (PDF/DOCX/TXT)
- Paste text directly
- Select a sample JD

### 4. Upload Resume(s)
- Single mode: Upload one resume
- Batch mode: Upload multiple resumes (ZIP or individual files)

### 5. View Results
Navigate through tabs:
- **Analysis**: Match score, skill breakdown, visualizations
- **Batch Results**: Ranked candidate list (batch mode)
- **Evaluation**: Model performance metrics
- **Help**: Usage guide

---

## Project Structure

```
AI_Resume_Screener/
├── app.py                 # Streamlit web interface
├── resume_parser.py       # PDF/DOCX text extraction
├── nlp_extractor.py       # NLP entity extraction (skills, experience)
├── skill_matcher.py       # Keyword-based skill matching
├── semantic_matcher.py    # SBERT semantic similarity
├── skills_taxonomy.py     # Skills database and categorization
├── report_generator.py    # PDF report generation
├── evaluation.py          # Model evaluation metrics
├── requirements.txt       # Python dependencies
└── sample_data/           # Sample job descriptions and test data
```

---

## Dependencies

**Core:**
- `streamlit` - Web interface
- `PyPDF2`, `python-docx` - Document parsing
- `pandas`, `numpy` - Data processing
- `plotly` - Visualizations

**NLP (Optional but recommended):**
- `sentence-transformers` - Semantic matching
- `spacy` - Entity extraction
- `nltk` - Text processing

**Reports:**
- `fpdf2` - PDF generation

Install all with:
```bash
pip install -r requirements.txt
```

---

## Technical Details

### Scoring Algorithm

The final match score combines four components:

```
Score = (skill_weight × skill_score) + (exp_weight × exp_score) 
      + (edu_weight × edu_score) + (keyword_weight × keyword_score)
```

### Skill Matching Methods

1. **Exact Match**: Direct keyword comparison
2. **Taxonomy Match**: Recognizes related skills (e.g., "React" → "React.js")
3. **Semantic Match**: SBERT embeddings for context-aware similarity

### Key Modules

| Module | Purpose |
|--------|---------|
| `resume_parser.py` | Extracts raw text from documents |
| `nlp_extractor.py` | Identifies skills, experience, education entities |
| `semantic_matcher.py` | Computes similarity using sentence embeddings |
| `skill_matcher.py` | Handles exact and taxonomy-based matching |

---

## Sample Usage

**Example Job Description Input:**
```
Looking for a Python developer with 3+ years experience.
Required: Python, SQL, AWS, Docker
Nice to have: Machine Learning, Kubernetes
```

**Example Output:**
- Match Score: 78%
- Matched Skills: Python, SQL, AWS
- Missing Skills: Docker, Kubernetes
- Experience: 4 years (exceeds requirement)

---

## Future Improvements

- OpenAI/LLM integration for better extraction
- ATS integration
- Resume improvement suggestions
- Historical candidate tracking

---

## License

Created for educational purposes.

---

## Acknowledgments

Built with Streamlit, scikit-learn, Sentence-Transformers, and Plotly.
