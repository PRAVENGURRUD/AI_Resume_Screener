"""
NLP-Based Entity Extraction Module

This module provides advanced NLP capabilities using spaCy for extracting
structured information from resumes including skills, experience, education,
job titles, and years of experience.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import spaCy
try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

# Try to import NLTK as fallback
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")


@dataclass
class ExtractedEntity:
    """Data class for extracted entities."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class ExperienceEntry:
    """Data class for work experience entries."""
    job_title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    years: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


@dataclass
class EducationEntry:
    """Data class for education entries."""
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[int] = None
    gpa: Optional[float] = None


class NLPExtractor:
    """Advanced NLP-based entity extractor for resumes."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NLP extractor.
        
        Args:
            model_name: spaCy model to use
        """
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                self._setup_matchers()
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"spaCy model '{model_name}' not found. Downloading...")
                try:
                    spacy.cli.download(model_name)
                    self.nlp = spacy.load(model_name)
                    self._setup_matchers()
                except Exception as e:
                    logger.error(f"Failed to download spaCy model: {e}")
        
        # Common patterns
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for extraction."""
        # Date patterns
        self.date_pattern = re.compile(
            r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'[\s,]*\d{4}|\d{1,2}/\d{4}|\d{4})',
            re.IGNORECASE
        )
        
        # Years of experience pattern
        self.experience_years_pattern = re.compile(
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp\.?)',
            re.IGNORECASE
        )
        
        # Duration pattern (e.g., "2019 - 2023", "Jan 2020 - Present")
        self.duration_pattern = re.compile(
            r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'[\s,]*)?(\d{4})\s*[-–—to]+\s*(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
            r'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
            r'Nov(?:ember)?|Dec(?:ember)?)[\s,]*)?(\d{4}|[Pp]resent|[Cc]urrent)',
            re.IGNORECASE
        )
        
        # Degree patterns
        self.degree_patterns = {
            'phd': re.compile(r'\b(?:Ph\.?D\.?|Doctor(?:ate)?|D\.Phil)\b', re.IGNORECASE),
            'masters': re.compile(r'\b(?:M\.?S\.?|M\.?A\.?|Master(?:\'?s)?|MBA|M\.?Tech|M\.?Eng)\b', re.IGNORECASE),
            'bachelors': re.compile(r'\b(?:B\.?S\.?|B\.?A\.?|Bachelor(?:\'?s)?|B\.?Tech|B\.?Eng|B\.?E\.?)\b', re.IGNORECASE),
            'associate': re.compile(r'\b(?:A\.?S\.?|A\.?A\.?|Associate(?:\'?s)?)\b', re.IGNORECASE),
            'diploma': re.compile(r'\b(?:Diploma|Certificate|Certification)\b', re.IGNORECASE)
        }
        
        # GPA pattern
        self.gpa_pattern = re.compile(r'(?:GPA|CGPA)[\s:]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?', re.IGNORECASE)
        
        # Job title patterns (common titles)
        self.job_titles = [
            'Software Engineer', 'Senior Software Engineer', 'Staff Engineer', 'Principal Engineer',
            'Software Developer', 'Senior Developer', 'Full Stack Developer', 'Frontend Developer',
            'Backend Developer', 'Web Developer', 'Mobile Developer', 'iOS Developer', 'Android Developer',
            'Data Scientist', 'Senior Data Scientist', 'Data Analyst', 'Data Engineer',
            'Machine Learning Engineer', 'ML Engineer', 'AI Engineer', 'Research Scientist',
            'DevOps Engineer', 'Site Reliability Engineer', 'SRE', 'Cloud Engineer',
            'Product Manager', 'Program Manager', 'Project Manager', 'Technical Program Manager',
            'Engineering Manager', 'Technical Lead', 'Tech Lead', 'Team Lead', 'Director',
            'QA Engineer', 'Test Engineer', 'Quality Assurance', 'SDET',
            'Business Analyst', 'Systems Analyst', 'Solutions Architect', 'Technical Architect',
            'UI/UX Designer', 'UX Designer', 'UI Designer', 'Product Designer',
            'Database Administrator', 'DBA', 'System Administrator', 'Network Engineer',
            'Security Engineer', 'Cybersecurity Analyst', 'Information Security',
            'Consultant', 'Technical Consultant', 'IT Consultant',
            'Intern', 'Software Intern', 'Engineering Intern', 'Research Intern'
        ]
    
    def _setup_matchers(self):
        """Setup spaCy matchers for entity extraction."""
        if not self.nlp:
            return
        
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Add job title patterns to phrase matcher
        job_title_patterns = [self.nlp.make_doc(title.lower()) for title in self.job_titles]
        self.phrase_matcher.add("JOB_TITLE", job_title_patterns)
        
        # Add skill patterns (will be expanded from taxonomy)
        # Pattern for years of experience: "X years of experience"
        exp_pattern = [
            {"IS_DIGIT": True},
            {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}},
            {"LOWER": "of", "OP": "?"},
            {"LOWER": {"IN": ["experience", "exp"]}}
        ]
        self.matcher.add("EXPERIENCE_YEARS", [exp_pattern])
    
    def extract_entities(self, text: str) -> Dict[str, List[ExtractedEntity]]:
        """
        Extract all entities from text using NLP.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of entity types to lists of extracted entities
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'skills': [],
            'job_titles': [],
            'degrees': [],
            'certifications': []
        }
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                entity = ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char
                )
                
                if ent.label_ == "PERSON":
                    entities['persons'].append(entity)
                elif ent.label_ == "ORG":
                    entities['organizations'].append(entity)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities['locations'].append(entity)
                elif ent.label_ == "DATE":
                    entities['dates'].append(entity)
            
            # Use phrase matcher for job titles
            if self.phrase_matcher:
                matches = self.phrase_matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    entities['job_titles'].append(ExtractedEntity(
                        text=span.text,
                        label="JOB_TITLE",
                        start=span.start_char,
                        end=span.end_char
                    ))
        
        elif NLTK_AVAILABLE:
            # Fallback to NLTK
            entities = self._extract_with_nltk(text)
        
        return entities
    
    def _extract_with_nltk(self, text: str) -> Dict[str, List[ExtractedEntity]]:
        """
        Fallback extraction using NLTK.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'skills': [],
            'job_titles': [],
            'degrees': [],
            'certifications': []
        }
        
        try:
            # Ensure NLTK data is downloaded
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('maxent_ne_chunker_tab', quiet=True)
            
            sentences = sent_tokenize(text)
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                tree = ne_chunk(tagged)
                
                for subtree in tree:
                    if hasattr(subtree, 'label'):
                        entity_text = ' '.join([token for token, pos in subtree.leaves()])
                        label = subtree.label()
                        
                        entity = ExtractedEntity(
                            text=entity_text,
                            label=label,
                            start=0,
                            end=len(entity_text)
                        )
                        
                        if label == "PERSON":
                            entities['persons'].append(entity)
                        elif label == "ORGANIZATION":
                            entities['organizations'].append(entity)
                        elif label in ["GPE", "LOCATION"]:
                            entities['locations'].append(entity)
        except Exception as e:
            logger.error(f"NLTK extraction failed: {e}")
        
        return entities
    
    def extract_years_of_experience(self, text: str) -> Tuple[Optional[float], List[str]]:
        """
        Extract total years of experience from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (total years, list of experience mentions)
        """
        mentions = []
        total_years = 0.0
        
        # Find explicit mentions of years of experience
        matches = self.experience_years_pattern.findall(text)
        for match in matches:
            years = float(match)
            mentions.append(f"{int(years)} years of experience")
            total_years = max(total_years, years)
        
        # Calculate from job durations
        durations = self.duration_pattern.findall(text)
        calculated_years = 0.0
        
        current_year = datetime.now().year
        for start_year, end_year in durations:
            try:
                start = int(start_year)
                if end_year.lower() in ['present', 'current']:
                    end = current_year
                else:
                    end = int(end_year)
                
                duration = end - start
                if 0 < duration <= 50:  # Reasonable duration
                    calculated_years += duration
                    mentions.append(f"{start_year} - {end_year}")
            except ValueError:
                continue
        
        # Use the maximum of explicit and calculated
        if calculated_years > total_years:
            total_years = calculated_years
        
        return total_years if total_years > 0 else None, mentions
    
    def extract_education(self, text: str) -> List[EducationEntry]:
        """
        Extract education entries from text.
        
        Args:
            text: Input text
            
        Returns:
            List of EducationEntry objects
        """
        education_entries = []
        
        # Common education section headers
        education_section = self._extract_section(text, [
            'education', 'academic', 'qualifications', 'degrees'
        ])
        
        if not education_section:
            education_section = text
        
        # Find degree mentions
        for degree_type, pattern in self.degree_patterns.items():
            matches = pattern.finditer(education_section)
            for match in matches:
                entry = EducationEntry()
                entry.degree = match.group()
                
                # Try to find the context around the degree
                start = max(0, match.start() - 100)
                end = min(len(education_section), match.end() + 200)
                context = education_section[start:end]
                
                # Extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', context)
                if year_match:
                    entry.year = int(year_match.group())
                
                # Extract GPA
                gpa_match = self.gpa_pattern.search(context)
                if gpa_match:
                    gpa = float(gpa_match.group(1))
                    scale = float(gpa_match.group(2)) if gpa_match.group(2) else 4.0
                    entry.gpa = gpa / scale * 4.0 if scale != 4.0 else gpa
                
                # Extract field of study (common fields)
                fields = [
                    'Computer Science', 'Information Technology', 'Software Engineering',
                    'Electrical Engineering', 'Mechanical Engineering', 'Data Science',
                    'Business Administration', 'Mathematics', 'Physics', 'Chemistry',
                    'Biology', 'Economics', 'Finance', 'Marketing', 'Psychology',
                    'Communications', 'English', 'History', 'Art', 'Design'
                ]
                for field in fields:
                    if field.lower() in context.lower():
                        entry.field = field
                        break
                
                education_entries.append(entry)
        
        return education_entries
    
    def extract_job_titles(self, text: str) -> List[str]:
        """
        Extract job titles from text.
        
        Args:
            text: Input text
            
        Returns:
            List of job titles found
        """
        found_titles = []
        text_lower = text.lower()
        
        # Check for known job titles
        for title in self.job_titles:
            if title.lower() in text_lower:
                found_titles.append(title)
        
        # Use NLP if available for additional extraction
        if self.nlp:
            doc = self.nlp(text)
            if self.phrase_matcher:
                matches = self.phrase_matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    if span.text not in found_titles:
                        found_titles.append(span.text)
        
        return list(set(found_titles))
    
    def extract_experience_entries(self, text: str) -> List[ExperienceEntry]:
        """
        Extract detailed work experience entries.
        
        Args:
            text: Input text
            
        Returns:
            List of ExperienceEntry objects
        """
        entries = []
        
        # Find experience section
        exp_section = self._extract_section(text, [
            'experience', 'employment', 'work history', 'professional experience',
            'career', 'positions'
        ])
        
        if not exp_section:
            exp_section = text
        
        # Split into potential entries (by dates or job titles)
        lines = exp_section.split('\n')
        current_entry = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for job title
            found_title = None
            for title in self.job_titles:
                if title.lower() in line.lower():
                    found_title = title
                    break
            
            # Check for date range
            duration_match = self.duration_pattern.search(line)
            
            if found_title or duration_match:
                if current_entry:
                    entries.append(current_entry)
                
                current_entry = ExperienceEntry()
                current_entry.job_title = found_title
                
                if duration_match:
                    current_entry.duration = duration_match.group()
                    try:
                        start_year = int(duration_match.group(1))
                        end_str = duration_match.group(2)
                        if end_str.lower() in ['present', 'current']:
                            end_year = datetime.now().year
                        else:
                            end_year = int(end_str)
                        current_entry.years = end_year - start_year
                    except ValueError:
                        pass
            elif current_entry:
                # Add to description
                if current_entry.description:
                    current_entry.description += ' ' + line
                else:
                    current_entry.description = line
        
        if current_entry:
            entries.append(current_entry)
        
        return entries
    
    def _extract_section(self, text: str, keywords: List[str]) -> Optional[str]:
        """
        Extract a section from text based on section headers.
        
        Args:
            text: Full text
            keywords: Keywords that might indicate section headers
            
        Returns:
            Extracted section text or None
        """
        lines = text.split('\n')
        section_start = None
        section_end = None
        
        # Common section headers to detect end of section
        all_section_headers = [
            'education', 'experience', 'skills', 'projects', 'certifications',
            'awards', 'publications', 'references', 'summary', 'objective',
            'professional', 'technical', 'work history', 'employment'
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this line is a section header we're looking for
            if section_start is None:
                for keyword in keywords:
                    if keyword in line_lower and len(line_lower) < 50:
                        section_start = i
                        break
            
            # Check if we've hit another section (end of our section)
            elif section_start is not None:
                for header in all_section_headers:
                    if header in line_lower and header not in keywords and len(line_lower) < 50:
                        section_end = i
                        break
                
                if section_end:
                    break
        
        if section_start is not None:
            if section_end is None:
                section_end = len(lines)
            return '\n'.join(lines[section_start:section_end])
        
        return None
    
    def extract_certifications(self, text: str) -> List[str]:
        """
        Extract certifications from text.
        
        Args:
            text: Input text
            
        Returns:
            List of certifications found
        """
        certifications = []
        
        # Common certifications
        cert_patterns = [
            r'AWS\s+(?:Certified\s+)?(?:Solutions\s+Architect|Developer|SysOps)',
            r'Azure\s+(?:Administrator|Developer|Solutions\s+Architect)',
            r'Google\s+Cloud\s+(?:Professional|Associate)',
            r'PMP|Project\s+Management\s+Professional',
            r'Scrum\s+Master|CSM|PSM',
            r'CISSP|CISM|CEH|Security\+',
            r'CompTIA\s+(?:A\+|Network\+|Security\+|Cloud\+)',
            r'Cisco\s+(?:CCNA|CCNP|CCIE)',
            r'Oracle\s+(?:Certified|OCA|OCP)',
            r'Microsoft\s+(?:Certified|MCSA|MCSE|Azure)',
            r'Kubernetes\s+(?:Administrator|Developer)|CKA|CKAD',
            r'Terraform\s+(?:Associate|Professional)',
            r'Docker\s+Certified',
            r'Red\s+Hat\s+(?:Certified|RHCE|RHCSA)',
            r'Six\s+Sigma|Lean\s+Six\s+Sigma',
            r'ITIL',
            r'Salesforce\s+(?:Certified|Administrator|Developer)'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        
        return list(set(certifications))
    
    def full_extraction(self, text: str) -> Dict:
        """
        Perform full extraction of all entity types.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all extracted information
        """
        entities = self.extract_entities(text)
        years, exp_mentions = self.extract_years_of_experience(text)
        
        return {
            'entities': entities,
            'years_of_experience': years,
            'experience_mentions': exp_mentions,
            'education': [vars(e) for e in self.extract_education(text)],
            'job_titles': self.extract_job_titles(text),
            'experience_entries': [vars(e) for e in self.extract_experience_entries(text)],
            'certifications': self.extract_certifications(text)
        }


# Convenience function
def extract_resume_entities(text: str) -> Dict:
    """
    Convenience function to extract all entities from resume text.
    
    Args:
        text: Resume text
        
    Returns:
        Dictionary with extracted information
    """
    extractor = NLPExtractor()
    return extractor.full_extraction(text)
