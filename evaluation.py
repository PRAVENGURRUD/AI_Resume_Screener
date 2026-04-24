"""
Evaluation Module

This module provides evaluation metrics for the resume screening system,
including Spearman correlation with human labels, processing time logging,
and comprehensive error handling and logging.
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import os

# Configure logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    LOGURU_AVAILABLE = False

# Statistics
try:
    from scipy import stats
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install with: pip install scipy")


@dataclass
class EvaluationPair:
    """Data class for a labeled evaluation pair."""
    pair_id: int
    resume_id: str
    jd_id: str
    human_score: float  # 0-100 human-assigned fit score
    system_score: Optional[float] = None  # System-calculated score
    notes: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Data class for processing metrics."""
    resume_id: str
    processing_time: float
    word_count: int
    skills_found: int
    ocr_used: bool
    errors: List[str]
    timestamp: str


class EvaluationLogger:
    """Logger for evaluation metrics and processing times."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the evaluation logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Log files
        self.metrics_file = self.log_dir / "processing_metrics.jsonl"
        self.errors_file = self.log_dir / "errors.jsonl"
        self.evaluation_file = self.log_dir / "evaluation_results.json"
        
        # Configure file logging
        if LOGURU_AVAILABLE:
            logger.add(
                self.log_dir / "app_{time}.log",
                rotation="10 MB",
                retention="7 days",
                level="INFO"
            )
    
    def log_processing(self, metrics: ProcessingMetrics):
        """
        Log processing metrics for a resume.
        
        Args:
            metrics: ProcessingMetrics object
        """
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
        
        logger.info(
            f"Processed resume {metrics.resume_id}: "
            f"{metrics.processing_time:.3f}s, {metrics.word_count} words, "
            f"{metrics.skills_found} skills"
        )
    
    def log_error(self, resume_id: str, error: str, context: Optional[Dict] = None):
        """
        Log an error.
        
        Args:
            resume_id: Resume identifier
            error: Error message
            context: Additional context
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'resume_id': resume_id,
            'error': error,
            'context': context or {}
        }
        
        with open(self.errors_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
        
        logger.error(f"Error processing {resume_id}: {error}")
    
    def get_processing_stats(self) -> Dict:
        """
        Get aggregated processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        if not self.metrics_file.exists():
            return {}
        
        times = []
        word_counts = []
        skills_counts = []
        ocr_count = 0
        total_count = 0
        
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    times.append(entry['processing_time'])
                    word_counts.append(entry['word_count'])
                    skills_counts.append(entry['skills_found'])
                    if entry.get('ocr_used'):
                        ocr_count += 1
                    total_count += 1
                except json.JSONDecodeError:
                    continue
        
        if not times:
            return {}
        
        return {
            'total_processed': total_count,
            'avg_processing_time': sum(times) / len(times),
            'min_processing_time': min(times),
            'max_processing_time': max(times),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'avg_skills_found': sum(skills_counts) / len(skills_counts),
            'ocr_usage_rate': ocr_count / total_count if total_count > 0 else 0
        }


class Evaluator:
    """Evaluator for comparing system scores with human labels."""
    
    def __init__(self, labeled_data_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            labeled_data_path: Path to labeled evaluation data
        """
        self.labeled_data_path = labeled_data_path
        self.labeled_pairs: List[EvaluationPair] = []
        self.logger = EvaluationLogger()
        
        if labeled_data_path and os.path.exists(labeled_data_path):
            self.load_labeled_data(labeled_data_path)
    
    def load_labeled_data(self, path: str):
        """
        Load labeled evaluation data from JSON file.
        
        Args:
            path: Path to JSON file
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.labeled_pairs = []
            for entry in data.get('pairs', []):
                pair = EvaluationPair(
                    pair_id=entry['pair_id'],
                    resume_id=entry['resume_id'],
                    jd_id=entry['jd_id'],
                    human_score=entry['human_score'],
                    system_score=entry.get('system_score'),
                    notes=entry.get('notes')
                )
                self.labeled_pairs.append(pair)
            
            logger.info(f"Loaded {len(self.labeled_pairs)} labeled pairs")
            
        except Exception as e:
            logger.error(f"Failed to load labeled data: {e}")
    
    def add_system_score(self, pair_id: int, system_score: float):
        """
        Add system score to a labeled pair.
        
        Args:
            pair_id: Pair identifier
            system_score: System-calculated score
        """
        for pair in self.labeled_pairs:
            if pair.pair_id == pair_id:
                pair.system_score = system_score
                break
    
    def calculate_spearman_correlation(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Spearman correlation between human and system scores.
        
        Returns:
            Tuple of (correlation coefficient, p-value) or (None, None) if not enough data
        """
        if not SCIPY_AVAILABLE:
            logger.error("scipy not available for correlation calculation")
            return None, None
        
        # Get pairs with both scores
        valid_pairs = [p for p in self.labeled_pairs if p.system_score is not None]
        
        if len(valid_pairs) < 3:
            logger.warning("Not enough data points for correlation (need at least 3)")
            return None, None
        
        human_scores = [p.human_score for p in valid_pairs]
        system_scores = [p.system_score for p in valid_pairs]
        
        try:
            correlation, p_value = stats.spearmanr(human_scores, system_scores)
            return correlation, p_value
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return None, None
    
    def calculate_pearson_correlation(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Pearson correlation between human and system scores.
        
        Returns:
            Tuple of (correlation coefficient, p-value) or (None, None)
        """
        if not SCIPY_AVAILABLE:
            return None, None
        
        valid_pairs = [p for p in self.labeled_pairs if p.system_score is not None]
        
        if len(valid_pairs) < 3:
            return None, None
        
        human_scores = [p.human_score for p in valid_pairs]
        system_scores = [p.system_score for p in valid_pairs]
        
        try:
            correlation, p_value = stats.pearsonr(human_scores, system_scores)
            return correlation, p_value
        except Exception as e:
            logger.error(f"Pearson correlation failed: {e}")
            return None, None
    
    def calculate_mae(self) -> Optional[float]:
        """
        Calculate Mean Absolute Error between human and system scores.
        
        Returns:
            MAE or None if not enough data
        """
        valid_pairs = [p for p in self.labeled_pairs if p.system_score is not None]
        
        if not valid_pairs:
            return None
        
        errors = [abs(p.human_score - p.system_score) for p in valid_pairs]
        return sum(errors) / len(errors)
    
    def calculate_rmse(self) -> Optional[float]:
        """
        Calculate Root Mean Square Error.
        
        Returns:
            RMSE or None if not enough data
        """
        if not SCIPY_AVAILABLE:
            return None
        
        valid_pairs = [p for p in self.labeled_pairs if p.system_score is not None]
        
        if not valid_pairs:
            return None
        
        errors = [(p.human_score - p.system_score) ** 2 for p in valid_pairs]
        return np.sqrt(sum(errors) / len(errors))
    
    def get_full_evaluation_report(self) -> Dict:
        """
        Generate a full evaluation report.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        spearman_corr, spearman_p = self.calculate_spearman_correlation()
        pearson_corr, pearson_p = self.calculate_pearson_correlation()
        
        valid_pairs = [p for p in self.labeled_pairs if p.system_score is not None]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_labeled_pairs': len(self.labeled_pairs),
            'pairs_with_system_scores': len(valid_pairs),
            'metrics': {
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'mae': self.calculate_mae(),
                'rmse': self.calculate_rmse()
            },
            'interpretation': self._interpret_correlation(spearman_corr),
            'processing_stats': self.logger.get_processing_stats()
        }
        
        return report
    
    def _interpret_correlation(self, correlation: Optional[float]) -> str:
        """
        Interpret correlation coefficient.
        
        Args:
            correlation: Correlation coefficient
            
        Returns:
            Human-readable interpretation
        """
        if correlation is None:
            return "Insufficient data for correlation analysis"
        
        abs_corr = abs(correlation)
        direction = "positive" if correlation > 0 else "negative"
        
        if abs_corr >= 0.9:
            strength = "very strong"
        elif abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "very weak or no"
        
        return f"{strength.capitalize()} {direction} correlation (r = {correlation:.3f})"
    
    def save_evaluation_report(self, output_path: Optional[str] = None):
        """
        Save evaluation report to file.
        
        Args:
            output_path: Path to save report (defaults to logs directory)
        """
        report = self.get_full_evaluation_report()
        
        output_path = output_path or str(self.logger.evaluation_file)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def save_labeled_data(self, output_path: str):
        """
        Save labeled data with system scores.
        
        Args:
            output_path: Path to save data
        """
        data = {
            'pairs': [asdict(p) for p in self.labeled_pairs],
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'total_pairs': len(self.labeled_pairs)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def create_sample_labeled_data(output_path: str = "sample_data/labeled_pairs.json"):
    """
    Create sample labeled evaluation data for testing.
    
    Args:
        output_path: Path to save the labeled data
    """
    # Sample labeled pairs (30 pairs as required)
    sample_pairs = [
        # High match pairs (human score 80-100)
        {"pair_id": 1, "resume_id": "resume_001", "jd_id": "jd_001", "human_score": 95, "notes": "Excellent Python/ML match"},
        {"pair_id": 2, "resume_id": "resume_002", "jd_id": "jd_001", "human_score": 88, "notes": "Strong skills, less experience"},
        {"pair_id": 3, "resume_id": "resume_003", "jd_id": "jd_002", "human_score": 92, "notes": "Perfect frontend match"},
        {"pair_id": 4, "resume_id": "resume_004", "jd_id": "jd_002", "human_score": 85, "notes": "Good React skills"},
        {"pair_id": 5, "resume_id": "resume_005", "jd_id": "jd_003", "human_score": 90, "notes": "Strong DevOps background"},
        
        # Medium-high match pairs (human score 60-79)
        {"pair_id": 6, "resume_id": "resume_006", "jd_id": "jd_001", "human_score": 75, "notes": "Some Python, limited ML"},
        {"pair_id": 7, "resume_id": "resume_007", "jd_id": "jd_002", "human_score": 70, "notes": "Angular not React"},
        {"pair_id": 8, "resume_id": "resume_008", "jd_id": "jd_003", "human_score": 78, "notes": "Good cloud, less K8s"},
        {"pair_id": 9, "resume_id": "resume_009", "jd_id": "jd_004", "human_score": 72, "notes": "Backend focus"},
        {"pair_id": 10, "resume_id": "resume_010", "jd_id": "jd_004", "human_score": 68, "notes": "Java instead of Python"},
        
        # Medium match pairs (human score 40-59)
        {"pair_id": 11, "resume_id": "resume_011", "jd_id": "jd_001", "human_score": 55, "notes": "Limited ML experience"},
        {"pair_id": 12, "resume_id": "resume_012", "jd_id": "jd_002", "human_score": 48, "notes": "Backend dev for frontend role"},
        {"pair_id": 13, "resume_id": "resume_013", "jd_id": "jd_003", "human_score": 52, "notes": "Some DevOps exposure"},
        {"pair_id": 14, "resume_id": "resume_014", "jd_id": "jd_005", "human_score": 58, "notes": "Partial skill match"},
        {"pair_id": 15, "resume_id": "resume_015", "jd_id": "jd_005", "human_score": 45, "notes": "Different tech stack"},
        
        # Low-medium match pairs (human score 20-39)
        {"pair_id": 16, "resume_id": "resume_016", "jd_id": "jd_001", "human_score": 35, "notes": "Different domain"},
        {"pair_id": 17, "resume_id": "resume_017", "jd_id": "jd_002", "human_score": 28, "notes": "Mobile dev, not web"},
        {"pair_id": 18, "resume_id": "resume_018", "jd_id": "jd_003", "human_score": 32, "notes": "QA background"},
        {"pair_id": 19, "resume_id": "resume_019", "jd_id": "jd_006", "human_score": 38, "notes": "Junior for senior role"},
        {"pair_id": 20, "resume_id": "resume_020", "jd_id": "jd_006", "human_score": 25, "notes": "Career changer"},
        
        # Low match pairs (human score 0-19)
        {"pair_id": 21, "resume_id": "resume_021", "jd_id": "jd_001", "human_score": 15, "notes": "No relevant skills"},
        {"pair_id": 22, "resume_id": "resume_022", "jd_id": "jd_002", "human_score": 12, "notes": "Different field entirely"},
        {"pair_id": 23, "resume_id": "resume_023", "jd_id": "jd_003", "human_score": 18, "notes": "Entry level, wrong domain"},
        {"pair_id": 24, "resume_id": "resume_024", "jd_id": "jd_007", "human_score": 10, "notes": "No tech background"},
        {"pair_id": 25, "resume_id": "resume_025", "jd_id": "jd_007", "human_score": 8, "notes": "Completely unrelated"},
        
        # Additional varied pairs
        {"pair_id": 26, "resume_id": "resume_001", "jd_id": "jd_005", "human_score": 65, "notes": "Cross-domain match"},
        {"pair_id": 27, "resume_id": "resume_002", "jd_id": "jd_006", "human_score": 78, "notes": "Senior experience"},
        {"pair_id": 28, "resume_id": "resume_003", "jd_id": "jd_007", "human_score": 42, "notes": "Partial overlap"},
        {"pair_id": 29, "resume_id": "resume_004", "jd_id": "jd_001", "human_score": 58, "notes": "Some ML background"},
        {"pair_id": 30, "resume_id": "resume_005", "jd_id": "jd_002", "human_score": 35, "notes": "DevOps for frontend role"}
    ]
    
    data = {
        'pairs': sample_pairs,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_pairs': len(sample_pairs),
            'description': 'Sample labeled data for evaluation with human fit scores (0-100)'
        }
    }
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Created sample labeled data with {len(sample_pairs)} pairs at {output_path}")
    return output_path


# Convenience functions
def log_resume_processing(
    resume_id: str,
    processing_time: float,
    word_count: int,
    skills_found: int,
    ocr_used: bool = False,
    errors: Optional[List[str]] = None
):
    """
    Log processing metrics for a resume.
    
    Args:
        resume_id: Resume identifier
        processing_time: Time taken to process
        word_count: Number of words in resume
        skills_found: Number of skills found
        ocr_used: Whether OCR was used
        errors: List of errors encountered
    """
    eval_logger = EvaluationLogger()
    metrics = ProcessingMetrics(
        resume_id=resume_id,
        processing_time=processing_time,
        word_count=word_count,
        skills_found=skills_found,
        ocr_used=ocr_used,
        errors=errors or [],
        timestamp=datetime.now().isoformat()
    )
    eval_logger.log_processing(metrics)


def run_evaluation(labeled_data_path: str) -> Dict:
    """
    Run evaluation and return results.
    
    Args:
        labeled_data_path: Path to labeled data
        
    Returns:
        Evaluation report dictionary
    """
    evaluator = Evaluator(labeled_data_path)
    return evaluator.get_full_evaluation_report()
