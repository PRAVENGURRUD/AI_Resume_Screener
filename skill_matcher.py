"""
Skill Matcher Module

This module provides functionality to match resume skills with job description requirements
using various matching algorithms including semantic similarity, keyword matching, and 
configurable scoring weights.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import numpy as np

# Import taxonomy
try:
    from skills_taxonomy import get_taxonomy, SkillsTaxonomy
    TAXONOMY_AVAILABLE = True
except ImportError:
    TAXONOMY_AVAILABLE = False
    logger.warning("Skills taxonomy not available")

# Import semantic matcher
try:
    from semantic_matcher import get_semantic_matcher, SemanticMatcher
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning("Semantic matcher not available")


@dataclass
class ScoringWeights:
    """Configurable weights for scoring components."""
    skill_overlap: float = 0.50      # 50% - skill overlap weight
    experience_alignment: float = 0.30  # 30% - experience alignment weight
    education_certs: float = 0.10    # 10% - education and certifications weight
    keyword_coverage: float = 0.10   # 10% - keyword coverage weight
    
    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.skill_overlap + self.experience_alignment + self.education_certs + self.keyword_coverage
        return abs(total - 1.0) < 0.01
    
    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = self.skill_overlap + self.experience_alignment + self.education_certs + self.keyword_coverage
        if total > 0:
            self.skill_overlap /= total
            self.experience_alignment /= total
            self.education_certs /= total
            self.keyword_coverage /= total


class SkillMatcher:
    """Class to match skills between resume and job description with configurable weights."""
    
    # Default weights as per requirements
    DEFAULT_WEIGHTS = ScoringWeights(
        skill_overlap=0.50,
        experience_alignment=0.30,
        education_certs=0.10,
        keyword_coverage=0.10
    )
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize the SkillMatcher.
        
        Args:
            weights: Optional custom scoring weights
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        if not self.weights.validate():
            logger.warning("Weights don't sum to 1.0, normalizing...")
            self.weights.normalize()
        
        # Initialize TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                max_features=1000
            )
        else:
            self.vectorizer = None
        
        # Initialize taxonomy
        self.taxonomy = get_taxonomy() if TAXONOMY_AVAILABLE else None
        
        # Initialize semantic matcher
        self.semantic_matcher = None
        if SEMANTIC_AVAILABLE:
            try:
                self.semantic_matcher = get_semantic_matcher()
            except Exception as e:
                logger.warning(f"Failed to initialize semantic matcher: {e}")
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize skill name for better matching.
        
        Args:
            skill: Skill name to normalize
            
        Returns:
            str: Normalized skill name
        """
        # Use taxonomy if available
        if self.taxonomy:
            return self.taxonomy.normalize_skill(skill)
        
        skill = skill.lower().strip()
        # Remove special characters
        skill = re.sub(r'[^\w\s]', '', skill)
        # Common skill variations
        variations = {
            'js': 'javascript',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'db': 'database',
            'ui/ux': 'ui ux',
            'ci/cd': 'ci cd'
        }
        return variations.get(skill, skill)
    
    def exact_match(self, resume_text: str, jd_skills: List[str]) -> Tuple[List[str], float]:
        """
        Perform exact keyword matching between resume and job skills.
        
        Args:
            resume_text: Text from resume
            jd_skills: List of required skills from job description
            
        Returns:
            Tuple[List[str], float]: Matched skills and match score (0-100)
        """
        resume_lower = resume_text.lower()
        jd_skills_normalized = [self.normalize_skill(skill) for skill in jd_skills]
        
        matched = []
        for skill in jd_skills_normalized:
            if skill in resume_lower:
                matched.append(skill)
        
        if len(jd_skills_normalized) == 0:
            return [], 0.0
        
        score = (len(matched) / len(jd_skills_normalized)) * 100
        return matched, round(score, 2)
    
    def semantic_match(self, resume_text: str, jd_skills: List[str]) -> Tuple[List[str], float]:
        """
        Perform semantic similarity matching using TF-IDF and cosine similarity.
        
        Args:
            resume_text: Text from resume
            jd_skills: List of required skills from job description
            
        Returns:
            Tuple[List[str], float]: Matched skills and match score (0-100)
        """
        if not jd_skills:
            return [], 0.0
        
        if not SKLEARN_AVAILABLE or self.vectorizer is None:
            # Fallback to exact match if scikit-learn is not available
            return self.exact_match(resume_text, jd_skills)
        
        # Create job description text from skills
        jd_text = ' '.join(jd_skills)
        
        # Vectorize both texts
        try:
            vectors = self.vectorizer.fit_transform([resume_text.lower(), jd_text.lower()])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Convert similarity to percentage
            score = similarity * 100
            
            # Find which skills matched
            matched = []
            resume_lower = resume_text.lower()
            for skill in jd_skills:
                skill_normalized = self.normalize_skill(skill)
                if skill_normalized in resume_lower:
                    matched.append(skill)
            
            return matched, round(score, 2)
        except Exception:
            # Fallback to exact match if vectorization fails
            return self.exact_match(resume_text, jd_skills)
    
    def hybrid_match(self, resume_text: str, jd_skills: List[str]) -> Tuple[List[str], float, Dict]:
        """
        Perform hybrid matching combining exact and semantic matching.
        
        Args:
            resume_text: Text from resume
            jd_skills: List of required skills from job description
            
        Returns:
            Tuple[List[str], float, Dict]: Matched skills, overall score, and detailed breakdown
        """
        if not jd_skills:
            return [], 0.0, {}
        
        # Exact match
        exact_matched, exact_score = self.exact_match(resume_text, jd_skills)
        
        # Semantic match
        semantic_matched, semantic_score = self.semantic_match(resume_text, jd_skills)
        
        # Combine results
        all_matched = list(set(exact_matched + semantic_matched))
        
        # Weighted score: 70% exact match, 30% semantic
        hybrid_score = (exact_score * 0.7) + (semantic_score * 0.3)
        
        # Calculate missing skills
        missing_skills = [skill for skill in jd_skills if skill.lower() not in [m.lower() for m in all_matched]]
        
        breakdown = {
            'exact_match_score': exact_score,
            'semantic_match_score': semantic_score,
            'hybrid_score': round(hybrid_score, 2),
            'matched_count': len(all_matched),
            'total_required': len(jd_skills),
            'missing_skills': missing_skills
        }
        
        return all_matched, round(hybrid_score, 2), breakdown
    
    def calculate_detailed_score(
        self, 
        resume_data: Dict, 
        jd_skills: List[str],
        jd_text: Optional[str] = None,
        required_experience_years: Optional[float] = None,
        required_education: Optional[List[str]] = None
    ) -> Dict:
        """
        Calculate detailed matching score with configurable weights.
        
        Scoring breakdown (configurable):
        - Skill overlap: 50% (default)
        - Experience alignment: 30% (default)
        - Education/Certifications: 10% (default)
        - Keyword coverage: 10% (default)
        
        Args:
            resume_data: Dictionary containing parsed resume data
            jd_skills: List of required skills from job description
            jd_text: Full job description text for semantic matching
            required_experience_years: Required years of experience
            required_education: Required education/degrees
            
        Returns:
            Dict: Detailed scoring breakdown
        """
        resume_text = resume_data.get('text', '')
        resume_skills = resume_data.get('skills', [])
        
        # 1. SKILL OVERLAP SCORE (50% weight by default)
        matched_skills, skill_score, breakdown = self.hybrid_match(resume_text, jd_skills)
        
        # Add SBERT semantic similarity if available
        sbert_similarity = None
        if self.semantic_matcher and jd_text:
            try:
                similarity_results = self.semantic_matcher.compute_resume_jd_similarity(
                    resume_text, jd_text, method='both'
                )
                sbert_similarity = similarity_results.get('sbert_similarity')
                breakdown['sbert_similarity'] = sbert_similarity
                breakdown['tfidf_similarity'] = similarity_results.get('tfidf_similarity')
                breakdown['combined_similarity'] = similarity_results.get('combined_similarity')
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")
        
        # Use taxonomy for better matching if available
        if self.taxonomy:
            taxonomy_overlap = self.taxonomy.calculate_skill_overlap(resume_skills, jd_skills)
            breakdown['taxonomy_overlap'] = taxonomy_overlap
            # Use taxonomy score if higher
            if taxonomy_overlap['overlap_score'] > skill_score:
                skill_score = taxonomy_overlap['overlap_score']
                matched_skills = taxonomy_overlap['matched_skills']
                breakdown['missing_skills'] = taxonomy_overlap['missing_skills']
        
        skill_component = skill_score * self.weights.skill_overlap
        
        # 2. EXPERIENCE ALIGNMENT SCORE (30% weight by default)
        resume_years = resume_data.get('years_of_experience', 0) or 0
        required_years = required_experience_years or 0
        
        if required_years > 0:
            if resume_years >= required_years:
                experience_score = 100.0
            elif resume_years > 0:
                experience_score = min((resume_years / required_years) * 100, 100)
            else:
                experience_score = 0.0
        else:
            # No experience requirement, give partial credit based on experience
            experience_entries = len(resume_data.get('experience', []))
            experience_score = min(experience_entries * 20, 100)
        
        experience_component = experience_score * self.weights.experience_alignment
        breakdown['experience_score'] = round(experience_score, 2)
        breakdown['resume_years'] = resume_years
        
        # 3. EDUCATION & CERTIFICATIONS SCORE (10% weight by default)
        education_score = 0.0
        education_entries = resume_data.get('education', [])
        certifications = resume_data.get('certifications', [])
        
        # Points for education
        education_score += min(len(education_entries) * 25, 50)
        
        # Points for certifications
        education_score += min(len(certifications) * 10, 50)
        
        # Check for required education if specified
        if required_education:
            edu_text = ' '.join(education_entries).lower()
            matched_edu = sum(1 for req in required_education if req.lower() in edu_text)
            if matched_edu > 0:
                education_score = min(education_score + (matched_edu / len(required_education)) * 50, 100)
        
        education_component = education_score * self.weights.education_certs
        breakdown['education_score'] = round(education_score, 2)
        breakdown['certifications_found'] = certifications
        
        # 4. KEYWORD COVERAGE SCORE (10% weight by default)
        keyword_score = 0.0
        if jd_text:
            # Extract keywords from JD
            jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
            resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
            
            # Filter to meaningful words (length > 3)
            jd_keywords = {w for w in jd_words if len(w) > 3}
            resume_keywords = {w for w in resume_words if len(w) > 3}
            
            if jd_keywords:
                coverage = len(jd_keywords & resume_keywords) / len(jd_keywords)
                keyword_score = coverage * 100
        else:
            # Use skill-based keyword coverage
            keyword_score = skill_score * 0.5
        
        keyword_component = keyword_score * self.weights.keyword_coverage
        breakdown['keyword_coverage_score'] = round(keyword_score, 2)
        
        # FINAL SCORE CALCULATION
        final_score = skill_component + experience_component + education_component + keyword_component
        
        # Cap at 100
        final_score = min(final_score, 100)
        
        # Calculate bonuses (for display/legacy compatibility)
        skill_bonus = min(len(resume_skills) * 2, 10)
        experience_bonus = min(len(resume_data.get('experience', [])) * 2, 10)
        education_bonus = min(len(education_entries) * 2, 5)
        
        # Identify under-emphasized strengths
        extra_skills = []
        if self.taxonomy:
            overlap = self.taxonomy.calculate_skill_overlap(resume_skills, jd_skills)
            extra_skills = overlap.get('extra_skills', [])
        
        return {
            'overall_score': round(final_score, 2),
            'skill_match_score': skill_score,
            'matched_skills': matched_skills,
            'resume_skills_found': resume_skills,
            'skill_bonus': skill_bonus,
            'experience_bonus': experience_bonus,
            'education_bonus': education_bonus,
            'breakdown': breakdown,
            'component_scores': {
                'skill_overlap': round(skill_component, 2),
                'experience_alignment': round(experience_component, 2),
                'education_certs': round(education_component, 2),
                'keyword_coverage': round(keyword_component, 2)
            },
            'weights_used': {
                'skill_overlap': self.weights.skill_overlap,
                'experience_alignment': self.weights.experience_alignment,
                'education_certs': self.weights.education_certs,
                'keyword_coverage': self.weights.keyword_coverage
            },
            'under_emphasized_strengths': extra_skills[:5],
            'recommendations': self._generate_recommendations(
                matched_skills, jd_skills, breakdown, resume_data, extra_skills
            )
        }
    
    def _generate_recommendations(
        self, 
        matched: List[str], 
        required: List[str], 
        breakdown: Dict,
        resume_data: Optional[Dict] = None,
        extra_skills: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate actionable recommendations for improving resume match.
        
        Args:
            matched: List of matched skills
            required: List of required skills
            breakdown: Score breakdown dictionary
            resume_data: Full resume data
            extra_skills: Skills in resume not in requirements
            
        Returns:
            List[str]: List of 3-5 targeted recommendations
        """
        recommendations = []
        missing = breakdown.get('missing_skills', [])
        
        # Recommendation 1: Missing skills
        if missing:
            top_missing = missing[:3]
            recommendations.append(
                f"Add experience with these in-demand skills: {', '.join(top_missing)}. "
                f"Consider online courses or projects to demonstrate proficiency."
            )
        
        # Recommendation 2: Match ratio feedback
        match_ratio = breakdown.get('matched_count', 0) / max(breakdown.get('total_required', 1), 1)
        if match_ratio < 0.5:
            recommendations.append(
                "Your skills match is below 50%. Focus on acquiring the core technical skills "
                "listed in the job description through bootcamps, certifications, or side projects."
            )
        elif match_ratio < 0.7:
            recommendations.append(
                "Good foundation! To improve from 50-70% match, consider emphasizing your "
                "existing skills more prominently and adding relevant keywords throughout your resume."
            )
        
        # Recommendation 3: Experience
        if resume_data:
            resume_years = resume_data.get('years_of_experience', 0) or 0
            experience_score = breakdown.get('experience_score', 0)
            
            if experience_score < 50:
                recommendations.append(
                    "Quantify your experience: add specific metrics, project outcomes, and "
                    "technologies used. Example: 'Reduced API response time by 40% using Redis caching.'"
                )
        
        # Recommendation 4: Under-emphasized strengths
        if extra_skills and len(extra_skills) > 0:
            recommendations.append(
                f"Highlight these valuable skills more prominently: {', '.join(extra_skills[:3])}. "
                f"These are strengths that could differentiate you from other candidates."
            )
        
        # Recommendation 5: Education/Certifications
        if breakdown.get('education_score', 0) < 50:
            recommendations.append(
                "Consider adding relevant certifications (AWS, Google Cloud, Scrum, etc.) "
                "to strengthen your profile and demonstrate commitment to professional development."
            )
        
        # Recommendation 6: Keywords
        if breakdown.get('keyword_coverage_score', 0) < 50:
            recommendations.append(
                "Increase keyword alignment: review the job description and incorporate "
                "relevant terminology naturally throughout your resume, especially in your summary and experience sections."
            )
        
        # Return top 5 recommendations
        return recommendations[:5]


# Convenience function for backward compatibility
def skill_match(resume_text: str, jd_skills: List[str]) -> Tuple[List[str], float]:
    """
    Legacy function for skill matching.
    
    Args:
        resume_text: Text from resume
        jd_skills: List of required skills
        
    Returns:
        Tuple[List[str], float]: Matched skills and score
    """
    matcher = SkillMatcher()
    matched, score, _ = matcher.hybrid_match(resume_text, jd_skills)
    return matched, score
