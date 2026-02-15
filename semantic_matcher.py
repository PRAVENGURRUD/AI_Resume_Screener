"""
Semantic Matcher Module

This module provides semantic similarity matching using sentence-transformers (SBERT)
and other embedding-based approaches for comparing resumes with job descriptions.
"""

import re
from typing import List, Tuple, Dict, Optional
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Try to import sklearn for fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SemanticMatcher:
    """Semantic similarity matcher using SBERT embeddings."""
    
    # Class-level model cache to avoid reloading
    _model_cache = {}
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic matcher.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Options: 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (accurate)
        """
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        
        if SBERT_AVAILABLE:
            # Use cached model if available
            if model_name in SemanticMatcher._model_cache:
                self.model = SemanticMatcher._model_cache[model_name]
                logger.info(f"Using cached SBERT model: {model_name}")
            else:
                try:
                    logger.info(f"Loading SBERT model: {model_name}")
                    self.model = SentenceTransformer(model_name)
                    SemanticMatcher._model_cache[model_name] = self.model
                    logger.info(f"Successfully loaded SBERT model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load SBERT model: {e}")
                    self.model = None
        
        # Fallback TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                max_features=5000
            )
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if model not available
        """
        if self.model is None:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Get embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embeddings or None
        """
        if self.model is None:
            return None
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if self.model is not None:
            try:
                embedding1 = self.model.encode(text1, convert_to_numpy=True)
                embedding2 = self.model.encode(text2, convert_to_numpy=True)
                
                # Cosine similarity
                similarity = util.cos_sim(embedding1, embedding2).item()
                return max(0, min(1, similarity))  # Clamp to [0, 1]
            except Exception as e:
                logger.error(f"SBERT similarity failed: {e}")
        
        # Fallback to TF-IDF
        return self._tfidf_similarity(text1, text2)
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF based similarity as fallback.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            return 0.0
        
        try:
            vectors = self.tfidf_vectorizer.fit_transform([text1.lower(), text2.lower()])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"TF-IDF similarity failed: {e}")
            return 0.0
    
    def match_skills_semantic(
        self, 
        resume_text: str, 
        jd_skills: List[str],
        threshold: float = 0.5
    ) -> Tuple[List[Dict], float]:
        """
        Match skills using semantic similarity.
        
        Args:
            resume_text: Full resume text
            jd_skills: List of required skills from job description
            threshold: Minimum similarity score to consider a match
            
        Returns:
            Tuple of (list of match details, overall score)
        """
        if not jd_skills:
            return [], 0.0
        
        matches = []
        total_score = 0.0
        
        # Extract sentences from resume for better matching
        resume_sentences = self._split_into_sentences(resume_text)
        
        for skill in jd_skills:
            best_match = {
                'skill': skill,
                'matched': False,
                'best_score': 0.0,
                'matched_context': None
            }
            
            # First check for exact match
            if skill.lower() in resume_text.lower():
                best_match['matched'] = True
                best_match['best_score'] = 1.0
                # Find the context
                for sentence in resume_sentences:
                    if skill.lower() in sentence.lower():
                        best_match['matched_context'] = sentence
                        break
            else:
                # Try semantic matching
                if self.model is not None:
                    skill_embedding = self.model.encode(skill, convert_to_numpy=True)
                    
                    for sentence in resume_sentences:
                        if len(sentence.strip()) < 10:
                            continue
                        
                        try:
                            sentence_embedding = self.model.encode(sentence, convert_to_numpy=True)
                            similarity = util.cos_sim(skill_embedding, sentence_embedding).item()
                            
                            if similarity > best_match['best_score']:
                                best_match['best_score'] = similarity
                                best_match['matched_context'] = sentence
                        except Exception:
                            continue
                    
                    if best_match['best_score'] >= threshold:
                        best_match['matched'] = True
            
            matches.append(best_match)
            total_score += best_match['best_score']
        
        # Calculate overall score as percentage
        overall_score = (total_score / len(jd_skills)) * 100 if jd_skills else 0.0
        
        return matches, round(overall_score, 2)
    
    def compute_resume_jd_similarity(
        self, 
        resume_text: str, 
        jd_text: str,
        method: str = 'both'
    ) -> Dict[str, float]:
        """
        Compute overall similarity between resume and job description.
        
        Args:
            resume_text: Full resume text
            jd_text: Full job description text
            method: 'sbert', 'tfidf', or 'both'
            
        Returns:
            Dictionary with similarity scores
        """
        results = {}
        
        if method in ['sbert', 'both'] and self.model is not None:
            try:
                sbert_score = self.compute_similarity(resume_text, jd_text)
                results['sbert_similarity'] = round(sbert_score * 100, 2)
            except Exception as e:
                logger.error(f"SBERT similarity failed: {e}")
                results['sbert_similarity'] = None
        
        if method in ['tfidf', 'both']:
            tfidf_score = self._tfidf_similarity(resume_text, jd_text)
            results['tfidf_similarity'] = round(tfidf_score * 100, 2)
        
        # Combined score (weighted average if both available)
        if results.get('sbert_similarity') is not None and results.get('tfidf_similarity') is not None:
            # SBERT is more accurate, give it higher weight
            results['combined_similarity'] = round(
                0.7 * results['sbert_similarity'] + 0.3 * results['tfidf_similarity'], 2
            )
        elif results.get('sbert_similarity') is not None:
            results['combined_similarity'] = results['sbert_similarity']
        else:
            results['combined_similarity'] = results.get('tfidf_similarity', 0.0)
        
        return results
    
    def find_similar_skills(
        self, 
        skill: str, 
        skill_list: List[str],
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find similar skills from a list.
        
        Args:
            skill: Skill to match
            skill_list: List of skills to search in
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (skill, similarity) tuples
        """
        if not skill_list:
            return []
        
        if self.model is None:
            # Fallback to simple string matching
            return self._simple_skill_matching(skill, skill_list)
        
        try:
            skill_embedding = self.model.encode(skill, convert_to_numpy=True)
            list_embeddings = self.model.encode(skill_list, convert_to_numpy=True)
            
            similarities = util.cos_sim(skill_embedding, list_embeddings)[0]
            
            # Get top matches above threshold
            results = []
            for i, score in enumerate(similarities):
                if score >= threshold:
                    results.append((skill_list[i], float(score)))
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Skill similarity search failed: {e}")
            return self._simple_skill_matching(skill, skill_list)
    
    def _simple_skill_matching(self, skill: str, skill_list: List[str]) -> List[Tuple[str, float]]:
        """
        Simple string-based skill matching as fallback.
        
        Args:
            skill: Skill to match
            skill_list: List of skills
            
        Returns:
            List of (skill, score) tuples
        """
        skill_lower = skill.lower()
        results = []
        
        for s in skill_list:
            s_lower = s.lower()
            if skill_lower in s_lower or s_lower in skill_lower:
                # Calculate overlap ratio
                overlap = len(set(skill_lower.split()) & set(s_lower.split()))
                total = len(set(skill_lower.split()) | set(s_lower.split()))
                score = overlap / total if total > 0 else 0
                results.append((s, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?\n]+', text)
        # Also split on bullet points
        expanded = []
        for s in sentences:
            parts = re.split(r'[•●○◦▪▸►]\s*', s)
            expanded.extend(parts)
        
        return [s.strip() for s in expanded if len(s.strip()) > 5]
    
    def rank_candidates(
        self, 
        jd_text: str, 
        resumes: List[Dict],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Rank multiple candidates against a job description.
        
        Args:
            jd_text: Job description text
            resumes: List of resume dictionaries with 'text' and 'name' keys
            weights: Optional custom weights for scoring
            
        Returns:
            List of ranked candidates with scores
        """
        if not resumes:
            return []
        
        default_weights = {
            'semantic_similarity': 0.4,
            'skill_overlap': 0.4,
            'experience_match': 0.2
        }
        weights = weights or default_weights
        
        results = []
        jd_embedding = None
        
        if self.model is not None:
            try:
                jd_embedding = self.model.encode(jd_text, convert_to_numpy=True)
            except Exception:
                pass
        
        for resume in resumes:
            resume_text = resume.get('text', '')
            
            score_data = {
                'name': resume.get('name', 'Unknown'),
                'scores': {},
                'overall_score': 0.0
            }
            
            # Semantic similarity
            if jd_embedding is not None and self.model is not None:
                try:
                    resume_embedding = self.model.encode(resume_text, convert_to_numpy=True)
                    similarity = util.cos_sim(jd_embedding, resume_embedding).item()
                    score_data['scores']['semantic_similarity'] = similarity * 100
                except Exception:
                    score_data['scores']['semantic_similarity'] = self._tfidf_similarity(resume_text, jd_text) * 100
            else:
                score_data['scores']['semantic_similarity'] = self._tfidf_similarity(resume_text, jd_text) * 100
            
            # Calculate overall score
            total_weight = 0
            weighted_sum = 0
            for key, weight in weights.items():
                if key in score_data['scores']:
                    weighted_sum += score_data['scores'][key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                score_data['overall_score'] = round(weighted_sum / total_weight, 2)
            
            results.append(score_data)
        
        # Sort by overall score
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add rank
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results


# Singleton instance for efficiency
_semantic_matcher_instance = None


def get_semantic_matcher(model_name: str = "all-MiniLM-L6-v2") -> SemanticMatcher:
    """
    Get or create a singleton semantic matcher instance.
    
    Args:
        model_name: Model name to use
        
    Returns:
        SemanticMatcher instance
    """
    global _semantic_matcher_instance
    
    if _semantic_matcher_instance is None or _semantic_matcher_instance.model_name != model_name:
        _semantic_matcher_instance = SemanticMatcher(model_name)
    
    return _semantic_matcher_instance


# Convenience functions
def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-100)
    """
    matcher = get_semantic_matcher()
    return matcher.compute_similarity(text1, text2) * 100


def match_resume_to_jd(resume_text: str, jd_text: str) -> Dict:
    """
    Match a resume to a job description.
    
    Args:
        resume_text: Resume text
        jd_text: Job description text
        
    Returns:
        Dictionary with matching results
    """
    matcher = get_semantic_matcher()
    return matcher.compute_resume_jd_similarity(resume_text, jd_text, method='both')
