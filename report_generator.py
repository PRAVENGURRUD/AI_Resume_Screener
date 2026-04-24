"""
PDF Report Generator Module

This module generates downloadable PDF reports for resume screening results
using fpdf2 library.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import fpdf2
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    FPDF = object  # Placeholder base class when fpdf2 not available
    logger.warning("fpdf2 not available. Install with: pip install fpdf2")


def sanitize_text(text: str) -> str:
    """Remove or replace Unicode characters not supported by Helvetica font."""
    if not text:
        return ""
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '•': '-',
        '●': '-',
        '○': '-',
        '◦': '-',
        '▪': '-',
        '▸': '>',
        '→': '->',
        '←': '<-',
        '✓': '[x]',
        '✔': '[x]',
        '✗': '[ ]',
        '✘': '[ ]',
        '★': '*',
        '☆': '*',
        '…': '...',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '©': '(c)',
        '®': '(R)',
        '™': '(TM)',
        '\u200b': '',  # Zero-width space
        '\xa0': ' ',   # Non-breaking space
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


class ReportPDF(FPDF):  # type: ignore
    """Custom PDF class for resume screening reports."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Add header to each page."""
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'AI Resume Screener Report', 0, align='C')
        self.ln(5)
        self.set_font('Helvetica', '', 8)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, align='C')
        self.ln(10)
    
    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, align='C')
    
    def chapter_title(self, title: str):
        """Add a chapter title."""
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 10, sanitize_text(title), 0, align='L', fill=True)
        self.ln(8)
    
    def section_title(self, title: str):
        """Add a section title."""
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, sanitize_text(title), 0)
        self.ln(6)
    
    def body_text(self, text: str):
        """Add body text."""
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, sanitize_text(text))
        self.ln(3)
    
    def add_score_box(self, score: float, label: str = "Match Score"):
        """Add a highlighted score box."""
        # Determine color based on score
        if score >= 80:
            self.set_fill_color(144, 238, 144)  # Light green
        elif score >= 60:
            self.set_fill_color(255, 255, 150)  # Light yellow
        else:
            self.set_fill_color(255, 182, 182)  # Light red
        
        self.set_font('Helvetica', 'B', 16)
        self.cell(60, 15, f'{label}: {score:.1f}%', 1, align='C', fill=True)
        self.ln(20)
    
    def add_skills_table(self, matched: List[str], missing: List[str]):
        """Add a skills comparison table."""
        self.section_title("Skills Analysis")
        
        # Table header
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(200, 200, 200)
        self.cell(95, 8, 'Matched Skills', 1, align='C', fill=True)
        self.cell(95, 8, 'Missing Skills', 1, align='C', fill=True)
        self.ln()
        
        # Table body
        self.set_font('Helvetica', '', 9)
        max_rows = max(len(matched), len(missing))
        
        for i in range(max_rows):
            matched_skill = matched[i] if i < len(matched) else ''
            missing_skill = missing[i] if i < len(missing) else ''
            
            # Alternate row colors
            if i % 2 == 0:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            
            self.cell(95, 6, sanitize_text(matched_skill), 1, fill=True)
            self.cell(95, 6, sanitize_text(missing_skill), 1, fill=True)
            self.ln()
        
        self.ln(5)
    
    def add_recommendations(self, recommendations: List[str]):
        """Add recommendations section."""
        self.section_title("Recommendations for Improvement")
        
        self.set_font('Helvetica', '', 10)
        for i, rec in enumerate(recommendations, 1):
            self.set_font('Helvetica', 'B', 10)
            self.cell(10, 6, f'{i}.')
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 6, sanitize_text(rec))
            self.ln(2)
        
        self.ln(5)


class ReportGenerator:
    """Generator for PDF reports."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_single_report(
        self,
        candidate_name: str,
        resume_data: Dict,
        analysis_results: Dict,
        jd_info: Optional[Dict] = None
    ) -> str:
        """
        Generate a PDF report for a single candidate.
        
        Args:
            candidate_name: Name of the candidate
            resume_data: Parsed resume data
            analysis_results: Analysis results from matcher
            jd_info: Optional job description info
            
        Returns:
            Path to generated PDF
        """
        if not FPDF_AVAILABLE:
            raise RuntimeError("fpdf2 not available. Install with: pip install fpdf2")
        
        pdf = ReportPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Title
        pdf.chapter_title(f"Resume Analysis: {candidate_name}")
        
        # Overall Score
        overall_score = analysis_results.get('overall_score', 0)
        pdf.add_score_box(overall_score)
        
        # Candidate Information
        pdf.section_title("Candidate Information")
        pdf.body_text(f"Name: {resume_data.get('name', 'N/A')}")
        pdf.body_text(f"Email: {resume_data.get('email', 'N/A')}")
        pdf.body_text(f"Phone: {resume_data.get('phone', 'N/A')}")
        
        if resume_data.get('years_of_experience'):
            pdf.body_text(f"Years of Experience: {resume_data.get('years_of_experience')}")
        
        if resume_data.get('job_titles'):
            pdf.body_text(f"Job Titles: {', '.join(resume_data.get('job_titles', [])[:5])}")
        
        pdf.ln(5)
        
        # Score Breakdown
        pdf.section_title("Score Breakdown")
        breakdown = analysis_results.get('breakdown', {})
        
        pdf.set_font('Helvetica', '', 10)
        pdf.body_text(f"Skill Match Score: {breakdown.get('hybrid_score', 0):.1f}%")
        pdf.body_text(f"TF-IDF Similarity: {breakdown.get('tfidf_similarity', 0):.1f}%")
        
        if breakdown.get('sbert_similarity') is not None:
            pdf.body_text(f"Semantic Similarity (SBERT): {breakdown.get('sbert_similarity', 0):.1f}%")
        
        pdf.body_text(f"Experience Bonus: +{analysis_results.get('experience_bonus', 0)} pts")
        pdf.body_text(f"Education Bonus: +{analysis_results.get('education_bonus', 0)} pts")
        
        pdf.ln(5)
        
        # Skills Table
        matched = analysis_results.get('matched_skills', [])
        missing = analysis_results.get('breakdown', {}).get('missing_skills', [])
        pdf.add_skills_table(matched, missing)
        
        # Detected Skills in Resume
        resume_skills = resume_data.get('skills', [])
        if resume_skills:
            pdf.section_title("Skills Found in Resume")
            skills_text = ', '.join(resume_skills[:20])
            if len(resume_skills) > 20:
                skills_text += f' ... and {len(resume_skills) - 20} more'
            pdf.body_text(skills_text)
            pdf.ln(5)
        
        # Certifications
        certifications = resume_data.get('certifications', [])
        if certifications:
            pdf.section_title("Certifications")
            pdf.body_text(', '.join(certifications))
            pdf.ln(5)
        
        # Recommendations
        recommendations = analysis_results.get('recommendations', [])
        if recommendations:
            pdf.add_recommendations(recommendations)
        
        # Education
        education = resume_data.get('education', [])
        if education:
            pdf.section_title("Education")
            for edu in education[:3]:
                pdf.body_text(f"• {edu}")
        
        # Generate filename
        safe_name = "".join(c for c in candidate_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{safe_name}_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        pdf.output(str(filepath))
        logger.info(f"Generated report: {filepath}")
        
        return str(filepath)
    
    def generate_batch_report(
        self,
        results: List[Dict],
        jd_title: str = "Job Position"
    ) -> str:
        """
        Generate a batch report for multiple candidates.
        
        Args:
            results: List of analysis results for each candidate
            jd_title: Title of the job position
            
        Returns:
            Path to generated PDF
        """
        if not FPDF_AVAILABLE:
            raise RuntimeError("fpdf2 not available. Install with: pip install fpdf2")
        
        pdf = ReportPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Title
        pdf.chapter_title(f"Candidate Ranking Report: {jd_title}")
        pdf.body_text(f"Total Candidates Analyzed: {len(results)}")
        pdf.ln(10)
        
        # Summary Table
        pdf.section_title("Candidate Rankings")
        
        # Table header
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(10, 8, '#', 1, align='C', fill=True)
        pdf.cell(50, 8, 'Candidate', 1, align='C', fill=True)
        pdf.cell(25, 8, 'Score', 1, align='C', fill=True)
        pdf.cell(50, 8, 'Key Skills', 1, align='C', fill=True)
        pdf.cell(55, 8, 'Missing Skills', 1, align='C', fill=True)
        pdf.ln()
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x.get('overall_score', 0), reverse=True)
        
        # Table body
        pdf.set_font('Helvetica', '', 8)
        for i, result in enumerate(sorted_results, 1):
            # Alternate row colors
            if i % 2 == 0:
                pdf.set_fill_color(245, 245, 245)
            else:
                pdf.set_fill_color(255, 255, 255)
            
            name = result.get('name', 'Unknown')[:25]
            score = result.get('overall_score', 0)
            matched = ', '.join(result.get('matched_skills', [])[:3])
            if len(result.get('matched_skills', [])) > 3:
                matched += '...'
            missing = ', '.join(result.get('breakdown', {}).get('missing_skills', [])[:3])
            if len(result.get('breakdown', {}).get('missing_skills', [])) > 3:
                missing += '...'
            
            pdf.cell(10, 6, str(i), 1, fill=True)
            pdf.cell(50, 6, sanitize_text(name), 1, fill=True)
            pdf.cell(25, 6, f'{score:.1f}%', 1, align='C', fill=True)
            pdf.cell(50, 6, sanitize_text(matched[:30]), 1, fill=True)
            pdf.cell(55, 6, sanitize_text(missing[:35]), 1, fill=True)
            pdf.ln()
        
        pdf.ln(10)
        
        # Top Candidates Detail
        pdf.chapter_title("Top Candidates Detail")
        
        for i, result in enumerate(sorted_results[:5], 1):
            pdf.section_title(f"{i}. {sanitize_text(result.get('name', 'Unknown'))} - {result.get('overall_score', 0):.1f}%")
            
            matched_skills = result.get('matched_skills', [])
            if matched_skills:
                pdf.body_text(f"Matched Skills: {', '.join(matched_skills[:10])}")
            
            missing_skills = result.get('breakdown', {}).get('missing_skills', [])
            if missing_skills:
                pdf.body_text(f"Missing Skills: {', '.join(missing_skills[:5])}")
            
            recommendations = result.get('recommendations', [])
            if recommendations:
                pdf.body_text(f"Key Recommendation: {recommendations[0]}")
            
            pdf.ln(5)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_report_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        pdf.output(str(filepath))
        logger.info(f"Generated batch report: {filepath}")
        
        return str(filepath)
    
    def generate_evaluation_report(self, evaluation_data: Dict) -> str:
        """
        Generate a report for evaluation metrics.
        
        Args:
            evaluation_data: Evaluation metrics from Evaluator
            
        Returns:
            Path to generated PDF
        """
        if not FPDF_AVAILABLE:
            raise RuntimeError("fpdf2 not available. Install with: pip install fpdf2")
        
        pdf = ReportPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Title
        pdf.chapter_title("System Evaluation Report")
        
        # Overview
        pdf.section_title("Evaluation Overview")
        pdf.body_text(f"Total Labeled Pairs: {evaluation_data.get('total_labeled_pairs', 0)}")
        pdf.body_text(f"Pairs with System Scores: {evaluation_data.get('pairs_with_system_scores', 0)}")
        pdf.ln(5)
        
        # Metrics
        pdf.section_title("Correlation Metrics")
        metrics = evaluation_data.get('metrics', {})
        
        spearman = metrics.get('spearman_correlation')
        if spearman is not None:
            pdf.body_text(f"Spearman Correlation: {spearman:.4f}")
            pdf.body_text(f"Spearman p-value: {metrics.get('spearman_p_value', 'N/A')}")
        
        pearson = metrics.get('pearson_correlation')
        if pearson is not None:
            pdf.body_text(f"Pearson Correlation: {pearson:.4f}")
        
        mae = metrics.get('mae')
        if mae is not None:
            pdf.body_text(f"Mean Absolute Error: {mae:.2f}")
        
        rmse = metrics.get('rmse')
        if rmse is not None:
            pdf.body_text(f"Root Mean Square Error: {rmse:.2f}")
        
        pdf.ln(5)
        
        # Interpretation
        pdf.section_title("Interpretation")
        pdf.body_text(evaluation_data.get('interpretation', 'No interpretation available'))
        
        pdf.ln(5)
        
        # Processing Stats
        processing_stats = evaluation_data.get('processing_stats', {})
        if processing_stats:
            pdf.section_title("Processing Statistics")
            pdf.body_text(f"Total Processed: {processing_stats.get('total_processed', 0)}")
            pdf.body_text(f"Avg Processing Time: {processing_stats.get('avg_processing_time', 0):.3f}s")
            pdf.body_text(f"Avg Skills Found: {processing_stats.get('avg_skills_found', 0):.1f}")
            pdf.body_text(f"OCR Usage Rate: {processing_stats.get('ocr_usage_rate', 0)*100:.1f}%")
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        pdf.output(str(filepath))
        logger.info(f"Generated evaluation report: {filepath}")
        
        return str(filepath)


# Convenience function
def generate_candidate_report(
    candidate_name: str,
    resume_data: Dict,
    analysis_results: Dict,
    output_dir: str = "reports"
) -> str:
    """
    Generate a PDF report for a candidate.
    
    Args:
        candidate_name: Candidate name
        resume_data: Resume data
        analysis_results: Analysis results
        output_dir: Output directory
        
    Returns:
        Path to generated PDF
    """
    generator = ReportGenerator(output_dir)
    return generator.generate_single_report(candidate_name, resume_data, analysis_results)
