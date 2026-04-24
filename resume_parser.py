"""
Resume Parser Module

This module provides functionality to parse and extract information from resume files
(PDF and DOCX formats). It extracts personal information, skills, education, and experience.
Includes OCR fallback for image-based PDFs using pytesseract.
"""

import re
import time
import logging
from typing import Dict, List, Optional
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# OCR support
try:
    import pytesseract
    from pdf2image import convert_from_path, convert_from_bytes
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR not available. Install with: pip install pytesseract pdf2image Pillow")

# NLP support
try:
    from nlp_extractor import NLPExtractor
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLP extractor not available")

# Skills taxonomy
try:
    from skills_taxonomy import get_taxonomy, find_skills_in_text
    TAXONOMY_AVAILABLE = True
except ImportError:
    TAXONOMY_AVAILABLE = False

from email_validator import validate_email, EmailNotValidError  # type: ignore
import phonenumbers  # type: ignore
from phonenumbers import NumberParseException  # type: ignore


class ResumeParser:
    """Class to parse and extract information from resume files."""
    
    def __init__(self, use_nlp: bool = True, use_ocr: bool = True):
        """
        Initialize the ResumeParser with common patterns.
        
        Args:
            use_nlp: Whether to use NLP for enhanced extraction
            use_ocr: Whether to use OCR for image-based PDFs
        """
        self.use_nlp = use_nlp and NLP_AVAILABLE
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
        # Initialize NLP extractor if available
        self.nlp_extractor = None
        if self.use_nlp:
            try:
                self.nlp_extractor = NLPExtractor()
            except Exception as e:
                logger.warning(f"Failed to initialize NLP extractor: {e}")
                self.use_nlp = False
        
        # Initialize taxonomy
        self.taxonomy = None
        if TAXONOMY_AVAILABLE:
            self.taxonomy = get_taxonomy()
        
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        )
        self.skill_keywords = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb',
            'aws', 'docker', 'kubernetes', 'git', 'machine learning', 'ai',
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'flask', 'django',
            'fastapi', 'streamlit', 'html', 'css', 'bootstrap', 'angular',
            'vue', 'typescript', 'c++', 'c#', '.net', 'spring', 'express',
            'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka',
            'agile', 'scrum', 'devops', 'ci/cd', 'jenkins', 'terraform'
        ]
        
        # Processing metrics
        self.last_processing_time = 0.0
        self.ocr_used = False
    
    def read_pdf(self, file) -> str:
        """
        Extract text from a PDF file with OCR fallback.
        
        Args:
            file: File object or file path
            
        Returns:
            str: Extracted text from the PDF
            
        Raises:
            Exception: If PDF reading fails
        """
        self.ocr_used = False
        
        try:
            # First try normal text extraction
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Check if we got meaningful text
            if len(text.strip()) > 100:  # Threshold for meaningful content
                return text
            
            # If text is too short, try OCR
            logger.info("PDF text extraction yielded little content, trying OCR...")
            
        except Exception as e:
            logger.warning(f"Standard PDF extraction failed: {e}")
            text = ""
        
        # Try OCR if available and text extraction failed or was insufficient
        if self.use_ocr and OCR_AVAILABLE:
            try:
                ocr_text = self._extract_with_ocr(file)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    self.ocr_used = True
                    logger.info("OCR extraction successful")
                    return ocr_text
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
        
        if text.strip():
            return text
        
        raise Exception("Could not extract text from PDF. File may be encrypted or corrupted.")
    
    def _extract_with_ocr(self, file) -> str:
        """
        Extract text from PDF using OCR.
        
        Args:
            file: File object or file path
            
        Returns:
            str: OCR-extracted text
        """
        if not OCR_AVAILABLE:
            raise Exception("OCR is not available. Install pytesseract and pdf2image.")
        
        try:
            # Reset file pointer if it's a file object
            if hasattr(file, 'seek'):
                file.seek(0)
                file_bytes = file.read()
                images = convert_from_bytes(file_bytes)
            else:
                images = convert_from_path(file)
            
            text_parts = []
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i + 1}/{len(images)}")
                page_text = pytesseract.image_to_string(image)
                text_parts.append(page_text)
            
            return "\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    def read_docx(self, file) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file: File object or file path
            
        Returns:
            str: Extracted text from the DOCX
            
        Raises:
            Exception: If DOCX reading fails
        """
        # Try using python-docx if available
        if DOCX_AVAILABLE:
            try:
                doc = docx.Document(file)
                text = ""
                for para in doc.paragraphs:
                    text += para.text + "\n"
                return text
            except Exception as e:
                # Fall back to basic extraction if python-docx fails
                pass
        
        # Fallback: Extract text from DOCX using zipfile and XML parsing
        # DOCX files are ZIP archives containing XML files
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            
            # Reset file pointer if it's a file object
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Read DOCX as ZIP
            with zipfile.ZipFile(file, 'r') as docx_zip:
                # Extract text from main document XML
                try:
                    xml_content = docx_zip.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    
                    # Extract all text from XML
                    text_parts = []
                    for elem in root.iter():
                        if elem.text:
                            text_parts.append(elem.text)
                    
                    text = ' '.join(text_parts)
                    # Clean up extra whitespace
                    text = ' '.join(text.split())
                    return text
                except KeyError:
                    raise Exception("Could not find document.xml in DOCX file")
        except zipfile.BadZipFile:
            raise Exception("File is not a valid DOCX format")
        except Exception as e:
            raise Exception(
                f"Error reading DOCX file: {str(e)}. "
                "For better DOCX support, install python-docx: pip install python-docx"
            )
    
    def extract_email(self, text: str) -> Optional[str]:
        """
        Extract email address from text.
        
        Args:
            text: Text to search for email
            
        Returns:
            Optional[str]: Email address if found, None otherwise
        """
        emails = self.email_pattern.findall(text)
        if emails:
            try:
                validate_email(emails[0])
                return emails[0]
            except EmailNotValidError:
                return None
        return None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """
        Extract phone number from text.
        
        Args:
            text: Text to search for phone number
            
        Returns:
            Optional[str]: Phone number if found, None otherwise
        """
        phones = self.phone_pattern.findall(text)
        if phones:
            try:
                # Try to parse and format the phone number
                phone_str = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
                parsed = phonenumbers.parse(phone_str, "US")
                if phonenumbers.is_valid_number(parsed):
                    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            except (NumberParseException, IndexError):
                pass
        return None
    
    def extract_name(self, text: str) -> Optional[str]:
        """
        Extract candidate name (usually first line or after keywords).
        
        Args:
            text: Text to search for name
            
        Returns:
            Optional[str]: Name if found, None otherwise
        """
        lines = text.split('\n')
        # Usually name is in the first few lines
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 4 and not '@' in line:
                # Simple heuristic: name doesn't contain email or phone patterns
                if not self.email_pattern.search(line) and not self.phone_pattern.search(line):
                    return line
        return None
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from resume text using taxonomy and NLP.
        
        Args:
            text: Text to search for skills
            
        Returns:
            List[str]: List of found skills
        """
        found_skills = set()
        
        # Use taxonomy for comprehensive skill detection
        if self.taxonomy:
            skill_matches = find_skills_in_text(text)
            for canonical, matched in skill_matches:
                found_skills.add(canonical)
        
        # Fallback to keyword matching
        text_lower = text.lower()
        for skill in self.skill_keywords:
            if skill in text_lower:
                if self.taxonomy:
                    # Normalize using taxonomy
                    found_skills.add(self.taxonomy.normalize_skill(skill))
                else:
                    found_skills.add(skill.title())
        
        return list(found_skills)
    
    def extract_education(self, text: str) -> List[str]:
        """
        Extract education information from text.
        
        Args:
            text: Text to search for education
            
        Returns:
            List[str]: List of education entries
        """
        education_keywords = ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma']
        lines = text.split('\n')
        education = []
        capture = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in education_keywords):
                capture = True
                education.append(line.strip())
            elif capture and line.strip():
                if len(line.strip()) > 10:  # Likely continuation
                    education.append(line.strip())
                else:
                    capture = False
        
        return education[:5]  # Return top 5 education entries
    
    def extract_experience(self, text: str) -> List[str]:
        """
        Extract work experience from text.
        
        Args:
            text: Text to search for experience
            
        Returns:
            List[str]: List of experience entries
        """
        experience_keywords = ['experience', 'work', 'employment', 'position', 'role', 'job']
        lines = text.split('\n')
        experience = []
        capture = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in experience_keywords) and 'years' in line_lower:
                capture = True
            elif capture and line.strip():
                if len(line.strip()) > 15:  # Likely job description
                    experience.append(line.strip())
                    if len(experience) >= 10:  # Limit to 10 entries
                        break
        
        return experience[:10]
    
    def parse_resume(self, file, file_type: str) -> Dict:
        """
        Parse resume file and extract all information.
        
        Args:
            file: File object or file path
            file_type: Type of file ('pdf' or 'docx')
            
        Returns:
            Dict: Dictionary containing extracted resume information
        """
        start_time = time.time()
        
        try:
            if file_type.lower() == 'pdf':
                text = self.read_pdf(file)
            elif file_type.lower() == 'docx' or file_type.lower() == 'doc':
                text = self.read_docx(file)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Basic extraction
            result = {
                'text': text,
                'name': self.extract_name(text),
                'email': self.extract_email(text),
                'phone': self.extract_phone(text),
                'skills': self.extract_skills(text),
                'education': self.extract_education(text),
                'experience': self.extract_experience(text),
                'word_count': len(text.split()),
                'ocr_used': self.ocr_used,
            }
            
            # Enhanced extraction using NLP
            if self.use_nlp and self.nlp_extractor:
                try:
                    nlp_results = self.nlp_extractor.full_extraction(text)
                    
                    # Add NLP-extracted data
                    result['years_of_experience'] = nlp_results.get('years_of_experience')
                    result['experience_mentions'] = nlp_results.get('experience_mentions', [])
                    result['job_titles'] = nlp_results.get('job_titles', [])
                    result['certifications'] = nlp_results.get('certifications', [])
                    
                    # Enhanced education
                    nlp_education = nlp_results.get('education', [])
                    if nlp_education:
                        result['education_details'] = nlp_education
                    
                    # Enhanced experience
                    nlp_experience = nlp_results.get('experience_entries', [])
                    if nlp_experience:
                        result['experience_details'] = nlp_experience
                    
                    # Named entities
                    entities = nlp_results.get('entities', {})
                    result['organizations'] = [e.text if hasattr(e, 'text') else e.get('text', str(e)) 
                                               for e in entities.get('organizations', [])]
                    
                except Exception as e:
                    logger.warning(f"NLP extraction failed: {e}")
            
            # Record processing time
            self.last_processing_time = time.time() - start_time
            result['processing_time'] = round(self.last_processing_time, 3)
            
            return result
            
        except Exception as e:
            self.last_processing_time = time.time() - start_time
            raise Exception(f"Error parsing resume: {str(e)}")


# Convenience functions for backward compatibility
def read_pdf(file):
    """Legacy function for reading PDF files."""
    parser = ResumeParser()
    return parser.read_pdf(file)


def read_docx(file):
    """Legacy function for reading DOCX files."""
    parser = ResumeParser()
    return parser.read_docx(file)
