"""
AI-Powered Resume Screener Application

A comprehensive Streamlit application for screening resumes against job descriptions
using advanced NLP, semantic matching, and configurable scoring algorithms.

Features:
- Multiple resume upload
- Job description file upload or text input
- Configurable scoring weights
- Semantic similarity matching (SBERT)
- PDF report generation
- Batch ranking mode
- Evaluation dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Import core modules
from resume_parser import ResumeParser
from skill_matcher import SkillMatcher, ScoringWeights

# Import new modules
try:
    from semantic_matcher import get_semantic_matcher
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    from report_generator import ReportGenerator
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

try:
    from evaluation import Evaluator, EvaluationLogger, ProcessingMetrics
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .score-high { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .score-medium { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .score-low { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .candidate-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'resume_data': None,
        'analysis_results': None,
        'all_resumes': [],
        'all_results': [],
        'jd_text': '',
        'jd_skills': [],
        'weights': ScoringWeights(),
        'current_jd': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_sample_jds() -> List[Dict]:
    """Load sample job descriptions from file."""
    jd_path = Path("sample_data/job_descriptions.json")
    if jd_path.exists():
        with open(jd_path, 'r') as f:
            data = json.load(f)
        return data.get('job_descriptions', [])
    return []


def extract_jd_from_file(file) -> str:
    """Extract text from uploaded JD file."""
    parser = ResumeParser()
    file_type = file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            return parser.read_pdf(file)
        elif file_type in ['docx', 'doc']:
            return parser.read_docx(file)
        elif file_type == 'txt':
            return file.read().decode('utf-8')
        else:
            st.error(f"Unsupported file type: {file_type}")
            return ""
    except Exception as e:
        st.error(f"Error reading JD file: {e}")
        return ""


def parse_jd_skills(jd_text: str) -> List[str]:
    """Extract skills from job description text."""
    # Try using taxonomy if available
    try:
        from skills_taxonomy import find_skills_in_text
        found = find_skills_in_text(jd_text)
        return [skill for skill, _ in found]
    except ImportError:
        # Fallback to simple extraction
        common_skills = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws',
            'docker', 'kubernetes', 'machine learning', 'tensorflow', 'pytorch'
        ]
        jd_lower = jd_text.lower()
        return [s.title() for s in common_skills if s in jd_lower]


def display_resume_info(resume_data: Dict):
    """Display extracted resume information."""
    st.subheader("📋 Extracted Resume Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if resume_data.get('name'):
            st.metric("Name", resume_data['name'])
        if resume_data.get('email'):
            st.metric("Email", resume_data['email'])
    
    with col2:
        if resume_data.get('phone'):
            st.metric("Phone", resume_data['phone'])
        st.metric("Word Count", resume_data.get('word_count', 0))
    
    with col3:
        st.metric("Skills Found", len(resume_data.get('skills', [])))
        years_exp = resume_data.get('years_of_experience')
        if years_exp:
            st.metric("Years of Experience", f"{years_exp:.1f}")
        else:
            st.metric("Experience Entries", len(resume_data.get('experience', [])))
    
    # OCR indicator
    if resume_data.get('ocr_used'):
        st.info("📷 OCR was used to extract text from this PDF")
    
    # Processing time
    if resume_data.get('processing_time'):
        st.caption(f"⏱️ Processing time: {resume_data['processing_time']:.3f}s")
    
    if resume_data.get('skills'):
        st.write("**Detected Skills:**")
        skills_str = ", ".join(resume_data['skills'][:30])
        if len(resume_data['skills']) > 30:
            skills_str += f" ... and {len(resume_data['skills']) - 30} more"
        st.info(skills_str)
    
    # Job titles
    if resume_data.get('job_titles'):
        with st.expander("💼 Detected Job Titles"):
            for title in resume_data['job_titles'][:10]:
                st.write(f"• {title}")
    
    # Certifications
    if resume_data.get('certifications'):
        with st.expander("🏆 Certifications"):
            for cert in resume_data['certifications']:
                st.write(f"• {cert}")
    
    if resume_data.get('education'):
        with st.expander("📚 Education Details"):
            for edu in resume_data['education'][:5]:
                st.write(f"• {edu}")
    
    if resume_data.get('experience'):
        with st.expander("💼 Work Experience"):
            for exp in resume_data['experience'][:5]:
                st.write(f"• {exp}")


def display_score_visualization(results: Dict):
    """Display score visualization charts."""
    st.subheader("📊 Score Analysis")
    
    breakdown = results.get('breakdown', {})
    component_scores = results.get('component_scores', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart for overall score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=results.get('overall_score', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Match Score"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 70], 'color': "#ffffcc"},
                    {'range': [70, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Component breakdown bar chart
        if component_scores:
            labels = ['Skill Overlap', 'Experience', 'Education', 'Keywords']
            values = [
                component_scores.get('skill_overlap', 0),
                component_scores.get('experience_alignment', 0),
                component_scores.get('education_certs', 0),
                component_scores.get('keyword_coverage', 0)
            ]
            
            fig_bar = px.bar(
                x=labels,
                y=values,
                title="Score Components (Weighted)",
                labels={'x': 'Component', 'y': 'Score Contribution'},
                color=values,
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Skill match distribution
    col3, col4 = st.columns(2)
    
    with col3:
        matched_count = breakdown.get('matched_count', 0)
        missing_count = len(breakdown.get('missing_skills', []))
        
        if matched_count + missing_count > 0:
            fig_pie = px.pie(
                values=[matched_count, missing_count],
                names=['Matched Skills', 'Missing Skills'],
                title="Skill Match Distribution",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col4:
        # Similarity scores comparison
        if breakdown.get('sbert_similarity') is not None or breakdown.get('tfidf_similarity') is not None:
            sim_data = []
            if breakdown.get('tfidf_similarity') is not None:
                sim_data.append({'Method': 'TF-IDF', 'Score': breakdown['tfidf_similarity']})
            if breakdown.get('sbert_similarity') is not None:
                sim_data.append({'Method': 'SBERT', 'Score': breakdown['sbert_similarity']})
            
            if sim_data:
                fig_sim = px.bar(
                    sim_data,
                    x='Method',
                    y='Score',
                    title='Semantic Similarity Comparison',
                    color='Score',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_sim, use_container_width=True)


def display_recommendations(results: Dict):
    """Display recommendations for improving resume."""
    st.subheader("💡 Recommendations")
    
    recommendations = results.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")
    else:
        st.success("Great job! Your resume matches well with the job requirements.")
    
    # Under-emphasized strengths
    strengths = results.get('under_emphasized_strengths', [])
    if strengths:
        st.subheader("🌟 Under-emphasized Strengths")
        st.write("These skills in your resume could be highlighted more:")
        st.warning(", ".join(strengths))


def display_batch_results(all_results: List[Dict]):
    """Display batch results in a table."""
    st.subheader("📊 Candidate Rankings")
    
    if not all_results:
        st.info("No results to display. Upload resumes and analyze them first.")
        return
    
    # Sort by score
    sorted_results = sorted(all_results, key=lambda x: x.get('overall_score', 0), reverse=True)
    
    # Create DataFrame for table
    table_data = []
    for i, result in enumerate(sorted_results, 1):
        matched_skills = result.get('matched_skills', [])
        missing_skills = result.get('breakdown', {}).get('missing_skills', [])
        
        table_data.append({
            'Rank': i,
            'Candidate': result.get('name', 'Unknown'),
            'Match %': f"{result.get('overall_score', 0):.1f}%",
            'Score': result.get('overall_score', 0),
            'Key Skills Found': ', '.join(matched_skills[:5]),
            'Missing Skills': ', '.join(missing_skills[:3]),
            'Skills Count': f"{len(matched_skills)}/{len(matched_skills) + len(missing_skills)}"
        })
    
    df = pd.DataFrame(table_data)
    
    # Apply styling
    def highlight_score(val):
        if isinstance(val, (int, float)):
            if val >= 80:
                return 'background-color: #d4edda'
            elif val >= 60:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        return ''
    
    styled_df = df.style.applymap(highlight_score, subset=['Score'])
    
    st.dataframe(
        df[['Rank', 'Candidate', 'Match %', 'Key Skills Found', 'Missing Skills']],
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization
    fig = px.bar(
        df,
        x='Candidate',
        y='Score',
        color='Score',
        color_continuous_scale='RdYlGn',
        title='Candidate Score Comparison',
        labels={'Score': 'Match Score (%)'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    return sorted_results


def generate_pdf_report(result: Dict, resume_data: Dict) -> Optional[str]:
    """Generate PDF report for a candidate."""
    if not REPORT_AVAILABLE:
        st.error("PDF report generation not available. Install fpdf2: pip install fpdf2")
        return None
    
    try:
        generator = ReportGenerator("reports")
        candidate_name = resume_data.get('name', 'Unknown')
        filepath = generator.generate_single_report(
            candidate_name=candidate_name,
            resume_data=resume_data,
            analysis_results=result
        )
        return filepath
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Analysis Mode",
            ["Single Resume", "Batch Mode"],
            help="Single mode for detailed analysis, Batch mode for ranking multiple candidates"
        )
        
        st.markdown("---")
        
        # Scoring weights configuration
        st.subheader("📊 Scoring Weights")
        st.caption("Adjust how different factors contribute to the final score")
        
        skill_weight = st.slider(
            "Skill Overlap", 
            min_value=0, max_value=100, value=50,
            help="Weight for matching required skills (default: 50%)"
        )
        
        exp_weight = st.slider(
            "Experience Alignment",
            min_value=0, max_value=100, value=30,
            help="Weight for years of experience match (default: 30%)"
        )
        
        edu_weight = st.slider(
            "Education & Certs",
            min_value=0, max_value=100, value=10,
            help="Weight for education and certifications (default: 10%)"
        )
        
        keyword_weight = st.slider(
            "Keyword Coverage",
            min_value=0, max_value=100, value=10,
            help="Weight for JD keyword coverage (default: 10%)"
        )
        
        # Normalize weights
        total = skill_weight + exp_weight + edu_weight + keyword_weight
        if total > 0:
            st.session_state.weights = ScoringWeights(
                skill_overlap=skill_weight/total,
                experience_alignment=exp_weight/total,
                education_certs=edu_weight/total,
                keyword_coverage=keyword_weight/total
            )
            
            # Display normalized weights
            st.caption(f"Normalized: {skill_weight/total*100:.0f}% / {exp_weight/total*100:.0f}% / {edu_weight/total*100:.0f}% / {keyword_weight/total*100:.0f}%")
        
        return mode


def render_jd_section():
    """Render the job description input section."""
    st.subheader("📋 Job Description")
    
    jd_input_method = st.radio(
        "Input Method",
        ["Upload File", "Enter Text", "Use Sample JD"],
        horizontal=True
    )
    
    jd_text = ""
    jd_skills = []
    required_exp = None
    
    if jd_input_method == "Upload File":
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            help="Upload the job description file"
        )
        if jd_file:
            jd_text = extract_jd_from_file(jd_file)
            if jd_text:
                jd_skills = parse_jd_skills(jd_text)
                st.success(f"✅ Extracted JD with {len(jd_skills)} skills detected")
    
    elif jd_input_method == "Enter Text":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            jd_text = st.text_area(
                "Paste Job Description",
                height=200,
                placeholder="Paste the full job description here..."
            )
        
        with col2:
            skills_input = st.text_area(
                "Required Skills (comma-separated)",
                height=100,
                placeholder="Python, React, AWS, Docker..."
            )
            if skills_input:
                jd_skills = [s.strip() for s in skills_input.split(',') if s.strip()]
            
            required_exp = st.number_input(
                "Required Years of Experience",
                min_value=0, max_value=30, value=0,
                help="Minimum years of experience required"
            )
        
        if jd_text and not jd_skills:
            jd_skills = parse_jd_skills(jd_text)
    
    else:  # Use Sample JD
        sample_jds = load_sample_jds()
        if sample_jds:
            jd_options = {jd['title']: jd for jd in sample_jds}
            selected_jd = st.selectbox(
                "Select a Sample Job Description",
                options=list(jd_options.keys())
            )
            
            if selected_jd:
                jd_data = jd_options[selected_jd]
                jd_text = jd_data.get('description', '')
                jd_skills = jd_data.get('required_skills', [])
                required_exp = jd_data.get('required_experience_years')
                
                with st.expander("View Job Description"):
                    st.write(jd_text)
                    st.write(f"**Required Skills:** {', '.join(jd_skills)}")
                    st.write(f"**Required Experience:** {required_exp} years")
        else:
            st.warning("No sample job descriptions found. Please add them to sample_data/job_descriptions.json")
    
    # Display detected skills
    if jd_skills:
        st.write(f"**Required Skills ({len(jd_skills)}):** {', '.join(jd_skills[:15])}")
        if len(jd_skills) > 15:
            st.caption(f"... and {len(jd_skills) - 15} more")
    
    return jd_text, jd_skills, required_exp


def analyze_single_resume(uploaded_file, jd_text: str, jd_skills: List[str], required_exp: Optional[float]):
    """Analyze a single resume against job description."""
    parser = ResumeParser()
    matcher = SkillMatcher(weights=st.session_state.weights)
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("Parsing resume..."):
        resume_data = parser.parse_resume(uploaded_file, file_extension)
    
    with st.spinner("Analyzing resume and calculating match score..."):
        results = matcher.calculate_detailed_score(
            resume_data=resume_data,
            jd_skills=jd_skills,
            jd_text=jd_text,
            required_experience_years=required_exp
        )
    
    # Log processing metrics if available
    if EVAL_AVAILABLE:
        try:
            logger = EvaluationLogger()
            metrics = ProcessingMetrics(
                resume_id=uploaded_file.name,
                processing_time=resume_data.get('processing_time', 0),
                word_count=resume_data.get('word_count', 0),
                skills_found=len(resume_data.get('skills', [])),
                ocr_used=resume_data.get('ocr_used', False),
                errors=[],
                timestamp=datetime.now().isoformat()
            )
            logger.log_processing(metrics)
        except Exception:
            pass
    
    return resume_data, results


def main():
    """Main application function."""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🤖 AI-Powered Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Render sidebar and get mode
    mode = render_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📄 Analysis", "📊 Batch Results", "📈 Evaluation"])
    
    with tab1:
        # Job Description Section
        jd_text, jd_skills, required_exp = render_jd_section()
        
        st.markdown("---")
        
        # Resume Upload Section
        st.subheader("📄 Resume Upload")
        
        if mode == "Single Resume":
            uploaded_file = st.file_uploader(
                "Upload Resume",
                type=["pdf", "docx"],
                help="Upload a resume in PDF or DOCX format"
            )
            
            analyze_button = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)
            
            if analyze_button and uploaded_file and jd_skills:
                try:
                    resume_data, results = analyze_single_resume(
                        uploaded_file, jd_text, jd_skills, required_exp
                    )
                    
                    st.session_state.resume_data = resume_data
                    st.session_state.analysis_results = results
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display overall score
                    overall_score = results.get('overall_score', 0)
                    score_class = "score-high" if overall_score >= 80 else "score-medium" if overall_score >= 60 else "score-low"
                    
                    matched_count = len(results.get('matched_skills', []))
                    total_skills = matched_count + len(results.get('breakdown', {}).get('missing_skills', []))
                    
                    st.markdown(f"""
                    <div class="score-card {score_class}">
                        <h2>Overall Match Score: {overall_score:.1f}%</h2>
                        <p>Matched {matched_count} out of {total_skills} required skills</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Create tabs for different views
                    detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
                        "📋 Resume Info", "📊 Analysis", "💡 Recommendations", "📈 Details"
                    ])
                    
                    with detail_tab1:
                        display_resume_info(resume_data)
                    
                    with detail_tab2:
                        display_score_visualization(results)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("✅ Matched Skills")
                            matched_skills = results.get('matched_skills', [])
                            if matched_skills:
                                for skill in matched_skills:
                                    st.success(f"✓ {skill}")
                            else:
                                st.warning("No skills matched.")
                        
                        with col2:
                            st.subheader("❌ Missing Skills")
                            missing_skills = results.get('breakdown', {}).get('missing_skills', [])
                            if missing_skills:
                                for skill in missing_skills:
                                    st.error(f"✗ {skill}")
                    
                    with detail_tab3:
                        display_recommendations(results)
                    
                    with detail_tab4:
                        st.subheader("📈 Detailed Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Similarity Scores:**")
                            breakdown = results.get('breakdown', {})
                            st.metric("Hybrid Score", f"{breakdown.get('hybrid_score', 0):.1f}%")
                            if breakdown.get('sbert_similarity') is not None:
                                st.metric("SBERT Similarity", f"{breakdown.get('sbert_similarity', 0):.1f}%")
                            st.metric("TF-IDF Similarity", f"{breakdown.get('tfidf_similarity', 0):.1f}%")
                        
                        with col2:
                            st.write("**Component Scores:**")
                            components = results.get('component_scores', {})
                            st.metric("Skill Overlap", f"{components.get('skill_overlap', 0):.1f}")
                            st.metric("Experience", f"{components.get('experience_alignment', 0):.1f}")
                            st.metric("Education", f"{components.get('education_certs', 0):.1f}")
                            st.metric("Keywords", f"{components.get('keyword_coverage', 0):.1f}")
                        
                        # Weights used
                        st.write("**Weights Used:**")
                        weights = results.get('weights_used', {})
                        weight_text = f"Skills: {weights.get('skill_overlap', 0)*100:.0f}% | "
                        weight_text += f"Experience: {weights.get('experience_alignment', 0)*100:.0f}% | "
                        weight_text += f"Education: {weights.get('education_certs', 0)*100:.0f}% | "
                        weight_text += f"Keywords: {weights.get('keyword_coverage', 0)*100:.0f}%"
                        st.caption(weight_text)
                    
                    # Download report button - generate PDF and provide download
                    st.markdown("---")
                    st.subheader("📥 Download Report")
                    
                    # Generate the PDF report
                    report_path = generate_pdf_report(results, resume_data)
                    if report_path and os.path.exists(report_path):
                        with open(report_path, 'rb') as f:
                            pdf_data = f.read()
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_data,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.warning("PDF report generation not available. Check if fpdf2 is installed.")
                
                except Exception as e:
                    st.error(f"❌ Error processing resume: {e}")
            
            elif analyze_button and not jd_skills:
                st.warning("⚠️ Please enter job requirements first.")
        
        else:  # Batch Mode
            uploaded_files = st.file_uploader(
                "Upload Multiple Resumes",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                help="Upload multiple resumes for batch ranking"
            )
            
            analyze_batch_button = st.button("🔍 Analyze All Resumes", type="primary", use_container_width=True)
            
            if analyze_batch_button and uploaded_files and jd_skills:
                all_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})")
                    
                    try:
                        resume_data, results = analyze_single_resume(
                            file, jd_text, jd_skills, required_exp
                        )
                        
                        results['name'] = resume_data.get('name', file.name)
                        results['filename'] = file.name
                        results['resume_data'] = resume_data
                        
                        all_results.append(results)
                        
                    except Exception as e:
                        st.warning(f"Error processing {file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                st.session_state.all_results = all_results
                
                # Display summary
                if all_results:
                    avg_score = sum(r.get('overall_score', 0) for r in all_results) / len(all_results)
                    top_candidate = max(all_results, key=lambda x: x.get('overall_score', 0))
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Candidates", len(all_results))
                    col2.metric("Average Score", f"{avg_score:.1f}%")
                    col3.metric("Top Candidate", top_candidate.get('name', 'Unknown'))
                    
                    st.success("Switch to 'Batch Results' tab to see detailed rankings")
            
            elif analyze_batch_button and not jd_skills:
                st.warning("⚠️ Please enter job requirements first.")
    
    with tab2:
        if st.session_state.all_results:
            sorted_results = display_batch_results(st.session_state.all_results)
            
            # Batch PDF report - generate and show download button directly
            if REPORT_AVAILABLE:
                st.markdown("---")
                try:
                    generator = ReportGenerator("reports")
                    # Format results for batch report
                    batch_results = []
                    for r in st.session_state.all_results:
                        batch_results.append({
                            'name': r.get('name', 'Unknown'),
                            'overall_score': r.get('overall_score', 0),
                            'matched_skills': r.get('matched_skills', []),
                            'breakdown': r.get('breakdown', {}),
                            'recommendations': r.get('recommendations', [])
                        })
                    
                    report_path = generator.generate_batch_report(batch_results, "Job Position")
                    
                    if report_path and os.path.exists(report_path):
                        with open(report_path, 'rb') as f:
                            pdf_data = f.read()
                        st.download_button(
                            label="📄 Download Batch Report (PDF)",
                            data=pdf_data,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error generating batch report: {e}")
        else:
            st.info("👈 Upload and analyze resumes in Batch Mode to see rankings here.")
    
    with tab3:
        st.subheader("📈 Evaluation Dashboard")
        
        if EVAL_AVAILABLE:
            labeled_path = "sample_data/labeled_pairs.json"
            
            if os.path.exists(labeled_path):
                evaluator = Evaluator(labeled_path)
                report = evaluator.get_full_evaluation_report()
                
                st.write("**Evaluation Metrics:**")
                
                col1, col2, col3 = st.columns(3)
                
                metrics = report.get('metrics', {})
                
                with col1:
                    spearman = metrics.get('spearman_correlation')
                    if spearman is not None:
                        st.metric("Spearman Correlation", f"{spearman:.4f}")
                    else:
                        st.metric("Spearman Correlation", "N/A")
                
                with col2:
                    mae = metrics.get('mae')
                    if mae is not None:
                        st.metric("Mean Absolute Error", f"{mae:.2f}")
                    else:
                        st.metric("Mean Absolute Error", "N/A")
                
                with col3:
                    st.metric("Labeled Pairs", report.get('total_labeled_pairs', 0))
                
                st.write("**Interpretation:**")
                st.info(report.get('interpretation', 'No interpretation available'))
                
                # Processing stats
                processing_stats = report.get('processing_stats', {})
                if processing_stats:
                    st.write("**Processing Statistics:**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Processed", processing_stats.get('total_processed', 0))
                    col2.metric("Avg Processing Time", f"{processing_stats.get('avg_processing_time', 0):.3f}s")
                    col3.metric("OCR Usage", f"{processing_stats.get('ocr_usage_rate', 0)*100:.1f}%")
            else:
                st.warning("No labeled evaluation data found. Add labeled_pairs.json to sample_data/")
                
                if st.button("Generate Sample Evaluation Data"):
                    from evaluation import create_sample_labeled_data
                    create_sample_labeled_data(labeled_path)
                    st.success("Sample evaluation data created!")
                    st.rerun()
        else:
            st.warning("Evaluation module not available. Install scipy: pip install scipy")


if __name__ == "__main__":
    main()
