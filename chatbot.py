import os
import logging
from typing import List, Dict, Any, Optional
import json
import re
from fastapi import FastAPI, Path
import nltk
from pydantic import BaseModel
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# Define custom download directory
custom_dir = "nltk_data"

# Download stopwords to the specified folder
nltk.download('stopwords', download_dir=custom_dir)
nltk.download('punkt', download_dir=custom_dir)

from nltk.corpus import stopwords
nltk.data.path.append(custom_dir)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = [word.strip() for word in text.lower().split() if word.strip() != ""]
    return [word for word in words if word.isalnum() and word not in stop_words]


def calculate_relevance_score(query_text: str, result_text: str, base_score: float) -> float:
    """Calculate enhanced relevance score based on keyword matching and semantic similarity"""
    try:
        # Clean and tokenize both texts
        query_words = set(remove_stopwords(query_text.lower()))
        result_words = set(remove_stopwords(result_text.lower()))
        
        # Calculate keyword overlap
        if not query_words:
            keyword_overlap = 0
        else:
            common_words = query_words.intersection(result_words)
            keyword_overlap = len(common_words) / len(query_words)
        
        # Boost score for exact medical code matches
        query_codes = set(re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', query_text.upper()))
        result_codes = set(re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', result_text.upper()))
        
        code_match_bonus = 0.3 if query_codes.intersection(result_codes) else 0
        
        # Calculate final relevance score
        relevance_score = (base_score * 0.7) + (keyword_overlap * 0.2) + code_match_bonus + (len(result_text.strip()) / 1000 * 0.1)
        
        return min(relevance_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        logger.error(f"Error calculating relevance score: {e}")
        return base_score


def filter_by_content_quality(results: List[Dict[str, Any]], min_length: int = None) -> List[Dict[str, Any]]:
    """Filter results based on content quality and length"""
    if min_length is None:
        min_length = RAG_CONFIG["MIN_TEXT_LENGTH"]
    
    filtered_results = []
    
    for result in results:
        text = result.get("text", "").strip()
        
        # Skip very short or empty results
        if len(text) < min_length:
            continue
            
        # Skip results that are mostly punctuation or numbers
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.3:
            continue
            
        # Skip repetitive content
        words = text.split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.3:  # Too repetitive
                continue
        
        filtered_results.append(result)
    
    return filtered_results

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Document groups and their collection names
DOCUMENT_GROUPS = {
    "Group 1": {
        "collection":"Medical_Coder_",
        "files": ["RAG1.pdf", "RAG1_1.xlsx"],
        "description": "ICD-10 Guidelines",
        "priority": 1
    },
    "Group 2": {
        "collection":"Medical_Coder_",
        "files": ["RAG2.xlsx", "RAG2_1.pdf", "RAG2_2.pdf", "RAG2_3.pdf"],
        "description": "ICD-10 Index",
        "priority": 1
    },
    "Group 3": {
        "collection":"Medical_Coder_",
        "files": ["RAG3.csv"],
        "description": "ICD-10 Tabular List",
        "priority": 1
    }
}


RAG_CONFIG = {
    "MAX_RESULTS": 15,           # Maximum results to retrieve
    "MIN_TEXT_LENGTH": 33,      # Minimum text length for quality filtering
    "SIMILARITY_THRESHOLD": 0.7, # Minimum similarity score
    "MAX_SECTION_LENGTH": 1200, # Maximum length per source section
    "MAX_TEXT_LENGTH": 500      # Maximum length per individual result
}
# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_embedding(text: str) -> List[float]:
    """Generate embedding using Gemini's text-embedding model"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Error generating Gemini embedding: {e}")
        # Add retry logic instead of zero vector
        import time
        time.sleep(1)  # Brief delay
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except:
            raise Exception("Failed to generate embedding after retry")

def generate_gemini_response(messages: List[Dict[str, str]], temperature: float = 0.5, max_tokens: int = 2048) -> str:
    """Generate response using Gemini Flash 2.0"""
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Convert messages to Gemini format
        # Gemini expects a conversation format, so we'll combine system and user messages
        system_message = ""
        conversation_parts = []
        
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            elif message["role"] == "user":
                conversation_parts.append(f"User: {message['content']}")
            elif message["role"] == "assistant":
                conversation_parts.append(f"Assistant: {message['content']}")
        
        # Combine system message with conversation
        full_prompt = f"{system_message}\n\n" + "\n\n".join(conversation_parts)
        
        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return "I'm sorry, I encountered an error while generating your response. Please try again."

# Initialize FastAPI app
app = FastAPI(title="Multi-Source RAG Medical Coding Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    chat_id: str
    user_id: str = "default_user"
    is_new_chat: bool = False  # Flag to indicate if this is a new chat session

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    structured_query: Optional[Dict[str, Any]] = None
    conversation_context: Optional[Dict[str, Any]] = None  # New field for context info

def is_medical_query(question: str) -> bool:
    """Enhanced function to determine if the question is related to medical coding"""
    
    # Expanded medical keywords including clinical terms
    medical_keywords = [
        # Coding-specific terms
        'icd', 'code', 'diagnosis', 'medical', 'condition', 'disease', 'symptom',
        'guideline', 'documentation', 'requirement', 'coding', 'clinical', 'health',
        'patient', 'treatment', 'procedure', 'assessment', 'record', 'chart',
        
        # Clinical conditions and terms
        'pneumonia', 'sepsis', 'diabetes', 'hypertension', 'fracture', 'infection',
        'cancer', 'tumor', 'carcinoma', 'syndrome', 'disorder', 'injury', 'wound',
        'acute', 'chronic', 'malignant', 'benign', 'bilateral', 'unilateral',
        'myocardial', 'infarction', 'stroke', 'cerebrovascular', 'respiratory',
        'cardiovascular', 'gastrointestinal', 'neurological', 'psychiatric',
        'orthopedic', 'dermatological', 'ophthalmological', 'urological',
        
        # Anatomical terms
        'heart', 'lung', 'liver', 'kidney', 'brain', 'spine', 'bone', 'joint',
        'muscle', 'nerve', 'blood', 'vessel', 'artery', 'vein', 'abdomen',
        'chest', 'head', 'neck', 'arm', 'leg', 'hand', 'foot', 'eye', 'ear',
        
        # Medical procedures and treatments
        'surgery', 'operation', 'biopsy', 'transplant', 'dialysis', 'chemotherapy',
        'radiation', 'therapy', 'rehabilitation', 'medication', 'antibiotic',
        'anesthesia', 'ventilator', 'icu', 'intensive care',
        
        # Pathology terms
        'bacterial', 'viral', 'fungal', 'parasitic', 'infectious', 'inflammatory',
        'autoimmune', 'genetic', 'congenital', 'acquired', 'traumatic',
        'degenerative', 'metastatic', 'hemorrhage', 'thrombosis', 'embolism',
        
        # Specific organisms and conditions
        'staphylococcus', 'streptococcus', 'mrsa', 'pneumococcus', 'influenza',
        'covid', 'hepatitis', 'tuberculosis', 'malaria', 'hiv', 'aids',
        'alzheimer', 'parkinson', 'asthma', 'copd', 'emphysema',
        
        # Severity and acuity indicators
        'severe', 'mild', 'moderate', 'critical', 'stable', 'unstable',
        'progressive', 'recurrent', 'refractory', 'resistant', 'sensitive',
        
        # Medical specialties context
        'cardiology', 'pulmonology', 'neurology', 'oncology', 'pediatrics',
        'geriatrics', 'psychiatry', 'radiology', 'pathology', 'anesthesiology'
    ]
    
    # Medical code patterns (ICD-10, CPT, etc.)
    code_patterns = [
        r'\b[A-Z]\d{2}(?:\.\d{1,3})?\b',  # ICD-10 codes like A92.5
        r'\b[A-Z]\d{2}\b',  # ICD-10 codes without decimal like A92
        r'\blookup\s+code\b',
        r'\bcode\s+for\b',
        r'\bfind\s+code\b',
        r'\bsearch\s+for\s+[A-Z]\d{2}',  # "search for A92.5"
    ]
    
    question_lower = question.lower()
    
    # Check for common greetings or purely general chat
    general_chat_patterns = [
        'hello', 'hi there', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'how are you', 'thank you', 'thanks', 'bye', 'goodbye',
        'what is your name', 'who are you', 'can you help me'
    ]
    
    # If it's a pure greeting without medical context, return False
    if any(pattern in question_lower for pattern in general_chat_patterns) and len(question.split()) <= 5:
        return False
    
    # Check for medical code patterns first
    for pattern in code_patterns:
        if re.search(pattern, question.upper()):
            return True
    
    # Check for medical keywords
    if any(keyword in question_lower for keyword in medical_keywords):
        return True
    
    # Advanced pattern matching for complex medical scenarios
    medical_patterns = [
        # Age + gender + medical condition pattern
        r'\b\d+[-\s]*year[-\s]*old\s+(male|female|man|woman|patient)',
        
        # "Diagnosed with" pattern
        r'\b(diagnosed|presents)\s+with\b',
        
        # "Due to" medical causation pattern
        r'\b(due\s+to|caused\s+by|secondary\s+to)\s+\w+',
        
        # Medical severity indicators
        r'\b(severe|acute|chronic|mild|moderate)\s+\w+(sis|tion|ity|ism|oma)\b',
        
        # Admission/treatment context
        r'\b(admitted|hospitalized|treated|discharged)\s+(to|from|for|with)\b',
        
        # Multiple condition indicators
        r'\band\s+\w+(sis|tion|ity|ism|oma)\b.*\bwith\b',
        
        # What are the codes pattern
        r'\bwhat\s+(are\s+the\s+)?(correct\s+)?(icd|codes?)\b',
        
        # Medical scenario pattern
        r'\b(scenario|case|situation)\b.*\b(code|coding|icd)\b'
    ]
    
    for pattern in medical_patterns:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    
    # Check for medical abbreviations
    medical_abbreviations = [
        'mrsa', 'uti', 'copd', 'chf', 'mi', 'cva', 'dvt', 'pe', 'icu', 'er',
        'cad', 'dm', 'htn', 'afib', 'cabg', 'ptca', 'tia', 'ards', 'ckd'
    ]
    
    if any(abbr in question_lower for abbr in medical_abbreviations):
        return True
    
    # Contextual indicators - if question is complex and mentions medical-sounding terms
    if len(question.split()) > 10:  # Complex query
        medical_indicators = [
            'resistant', 'shock', 'failure', 'syndrome', 'episode', 'attack',
            'reaction', 'complication', 'manifestation', 'exacerbation'
        ]
        if any(indicator in question_lower for indicator in medical_indicators):
            return True
    
    return False

def is_follow_up_query(question: str, conversation_history: List[Dict[str, str]]) -> bool:
    """Determine if the current question is a follow-up to previous conversation"""
    if not conversation_history:
        return False
    
    follow_up_indicators = [
        'what about', 'how about', 'also', 'additionally', 'furthermore',
        'can you also', 'what if', 'in that case', 'regarding that',
        'about the previous', 'from the last', 'following up',
        'more about', 'expand on', 'clarify', 'explain further',
        'same patient', 'same case', 'this patient', 'this case'
    ]
    
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in follow_up_indicators)

def extract_conversation_context(conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract relevant context from conversation history"""
    context = {
        "previous_codes": [],
        "patient_info": {},
        "topics_discussed": [],
        "last_query_intent": None
    }
    
    try:
        # Analyze the conversation to extract context
        for i, message in enumerate(conversation_history):
            if message["role"] == "user":
                # Extract potential medical codes mentioned

                code_patterns = [
                    r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b',  # ICD-10 codes
                    r'\b\d{5}(?:-\d{2})?\b'  # CPT codes
                ]
                for pattern in code_patterns:
                    codes = re.findall(pattern, message["content"])
                    context["previous_codes"].extend(codes)
                
                # Extract patient demographics/context
                if any(term in message["content"].lower() for term in ['patient', 'year old', 'male', 'female']):
                    age_match = re.search(r'(\d+)\s*year[s]?\s*old', message["content"].lower())
                    gender_match = re.search(r'\b(male|female|man|woman)\b', message["content"].lower())
                    
                    if age_match:
                        context["patient_info"]["age"] = age_match.group(1)
                    if gender_match:
                        context["patient_info"]["gender"] = gender_match.group(1)
            
            elif message["role"] == "assistant":
                # Extract topics from assistant responses
                if "ICD-10" in message["content"]:
                    context["topics_discussed"].append("ICD-10 coding")
                if "CPT" in message["content"]:
                    context["topics_discussed"].append("CPT procedures")
        
        # Remove duplicates
        context["previous_codes"] = list(set(context["previous_codes"]))
        context["topics_discussed"] = list(set(context["topics_discussed"]))
        
    except Exception as e:
        logger.error(f"Error extracting conversation context: {e}")
    
    return context

def format_conversation_history_for_prompt(conversation_history: List[Dict[str, str]]) -> str:
    """Format conversation history for prompt inclusion"""
    if not conversation_history:
        return "No previous conversation."
    
    formatted_history = []
    for message in conversation_history:
        role = "User" if message["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {message['content']}")
    
    return "\n".join(formatted_history[-10:])  # Last 5 exchanges


def enhance_query_for_retrieval(query: str) -> str:
    """Enhanced query enhancement with better medical term mapping and code expansion for combo queries"""
    try:
        # Expanded medical term mappings for better retrieval
        medical_mappings = {
            # Common to medical terms
            'sugar disease': 'diabetes mellitus',
            'high blood pressure': 'hypertension',
            'heart attack': 'myocardial infarction',
            'stroke': 'cerebrovascular accident',
            'broken bone': 'fracture',
            'stomach ache': 'abdominal pain',
            'headache': 'cephalgia',
            'flu': 'influenza',
            'cold': 'upper respiratory infection',
            
            # Bacterial/organism terms
            'mrsa': 'methicillin-resistant staphylococcus aureus',
            'staph': 'staphylococcus',
            'strep': 'streptococcus',
            'e. coli': 'escherichia coli',
            'c. diff': 'clostridium difficile',
            
            # Condition synonyms
            'blood poisoning': 'sepsis',
            'septic shock': 'severe sepsis with septic shock',
            'lung infection': 'pneumonia',
            'chest infection': 'respiratory infection',
            'kidney infection': 'pyelonephritis',
            'bladder infection': 'cystitis',
            'skin infection': 'cellulitis',
            
            # Severity indicators
            'serious': 'severe',
            'bad': 'severe',
            'mild': 'mild',
            'getting worse': 'progressive',
            'came back': 'recurrent'
        }
        
        enhanced_query = query.lower()
        
        # Replace common terms with medical equivalents
        for common_term, medical_term in medical_mappings.items():
            if common_term in enhanced_query:
                enhanced_query = enhanced_query.replace(common_term, medical_term)
        
        # --- EVEN STRONGER: Expand "htn" and "ckd" queries for better RAG retrieval ---
        if (("htn" in enhanced_query or "hypertension" in enhanced_query) and
            ("ckd" in enhanced_query or "chronic kidney disease" in enhanced_query)):
            # Try to extract CKD stage (allow "ckd3", "ckd 3", "stage 3", etc.)
            stage = None
            match = re.search(r'(ckd\s*[\-:]?\s*|stage\s*)?(\d)', enhanced_query)
            if match:
                stage = match.group(2)
            # Build expanded query with all possible relevant codes and synonyms
            expanded = [
                "hypertension",
                "chronic kidney disease",
                "hypertensive chronic kidney disease",
                "htn",
                "ckd",
                "combination code",
                "I12", "I12.0", "I12.9", "I13", "I13.0", "I13.1", "I13.2", "I13.10", "I13.11", "I13.2",
                "N18", "N18.3", "N18.30", "N18.31", "N18.32", "N18.39",
                "renal disease",
                "kidney disease",
                "kidney failure",
                "renal failure",
                "renal insufficiency",
                "CKD stage 3",
                "stage 3 CKD",
                "stage III CKD",
                "stage 3 chronic kidney disease",
                "stage III chronic kidney disease",
                "chronic renal failure",
                "chronic renal insufficiency",
                "hypertensive renal disease",
                "hypertensive renal failure",
                "hypertensive kidney disease",
                "hypertensive nephropathy",
                "hypertensive nephrosclerosis",
                "ICD-10",
                "ICD 10",
                "ICD10",
                "ICD10CM",
                "ICD-10-CM",
                "code",
                "diagnosis",
                "classification"
            ]
            if stage:
                expanded += [
                    f"stage {stage} ckd",
                    f"ckd stage {stage}",
                    f"n18.{stage}",
                    f"n18.{stage}0", f"n18.{stage}1", f"n18.{stage}2", f"n18.{stage}9"
                ]
            # Join all terms for a dense query
            enhanced_query = " ".join(expanded)
        else:
            # Add ICD-10 context keywords for better retrieval
            icd_context_terms = ['icd-10', 'code', 'diagnosis', 'classification']
            if not any(term in enhanced_query for term in icd_context_terms):
                if any(phrase in enhanced_query for phrase in ['what are the', 'correct codes', 'icd codes', 'code for']):
                    enhanced_query += ' icd-10 diagnosis code'
                elif 'diagnosis' not in enhanced_query:
                    enhanced_query += ' diagnosis'
        return enhanced_query
    except Exception as e:
        logger.error(f"Error enhancing query: {e}")
        return query
    
def structure_user_input_with_context(question: str, conversation_context: Dict[str, Any] = None, conversation_history: List[Dict[str, str]] = None) -> str:
    """Rephrase user query using the preprocessing prompt"""
    try:
        # Use the new preprocessing prompt from the document
        system_prompt = """You are an expert medical coding assistant specializing in ICD-10. Your task is to rephrase a user's query to make it clear, concise, and optimized for retrieving relevant information from ICD-10 datasets (Guideline, Alphabetic Index, and Tabular List). The rephrased query should:

1. Preserve the original intent of the user's query.

2. Use precise medical terminology when applicable, aligning with ICD-10 standards.

3. Clarify ambiguous terms (e.g., "sugar disease" to "diabetes mellitus").

4. Normalize informal or conversational language into a structured format suitable for retrieval.

5. Incorporate relevant context from the conversation history to maintain continuity in multi-turn interactions.

6. Handle misspellings, synonyms, or vague phrasing by mapping to standard medical terms.

7. For complex cases or multiple conditions, structure the query:
   a. Check list of condition to identify diagnosis vs symptoms
   b. If symptom, check if it is a associated symptom for any other user listed diagnosis.
      i. If Yes: Drop the diagnosis
      ii. If No: capture the symptom
   c. Exception: if symptom is specified as atypical or any similar term that means per documentations it is unsure about symptom etiology, then capture the symptom 
   d. If more than one diagnosis is documented then check if any of the diagnosis is linked as per icd-10-cm:
      i. "Due to": check the documentation and if two or more condition is linked with "due to" or "secondary to" then consider those condition for combination code look up

8. Output only the rephrased query as a single sentence, avoiding explanations or additional text."""

        # Format conversation history
        conversation_history_text = format_conversation_history_for_prompt(conversation_history)
        
        # Create the user message with the template format
        user_message = f"""*Conversation History*: {conversation_history_text}

*Current User Query*: {question}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        rephrased_query = generate_gemini_response(messages, temperature=0.2, max_tokens=512)
        
        return rephrased_query.strip()

    except Exception as e:
        logger.error(f"Error rephrasing user input: {e}")
        return question




def extract_structured_query_intent(question: str, rephrased_query: str) -> Dict[str, Any]:
    """Extract structured query intent for CSV/Excel lookups with improved parsing - FIXED VERSION"""
    
    query_intent = {
        "is_structured_lookup": False,
        "lookup_type": None,
        "search_terms": [],
        "target_columns": [],
        "exact_match": False,
        "original_query": question  # Store original query for reference
    }
    
    # IMPROVED: More comprehensive code patterns
    code_patterns = [
        r'\b[A-Z]\d{2}(?:\.\d{1,3})?\b',  # ICD-10 codes like A92.5
        r'\b[A-Z]\d{2}\b',  # ICD-10 codes without decimal like A92
        r'\blookup\s+code\b',
        r'\bcode\s+for\b',
        r'\bfind\s+code\b',
        r'\bsearch\s+for\s+[A-Z]\d{2}',  # "search for A92.5"
    ]
    
    # Enhanced column references with more variations
    column_references = {
        'code description': ['Code Description', 'Description'],
        'keyword': ['Keyword for this Code', 'Keyword'],
        'synonym': ['Synonym'],
        'excludes1': ['Excludes1 Code(s)', 'Excludes1', 'excludes 1'],
        'excludes2': ['Excludes2 Code(s)', 'Excludes2', 'excludes 2'], 
        'includes': ['Includes Code(s)', 'Includes'],
        'laterality': ['Laterality'],
        'specificity': ['Specificity', 'Gender Specificity', 'Age Specificity'],
        'code first': ['Code First'],
        'use additional code': ['Use Additional Code'],
        'code also': ['Code Also']
    }

    
    question_lower = question.lower()
    rephrased_lower = rephrased_query.lower()
    
    # IMPROVED: Better code extraction
    extracted_codes = []

    all_texts = [question.upper(), rephrased_query.upper()]
    for text in all_texts:
        for pattern in code_patterns:
            matches = re.findall(pattern, text)
            extracted_codes.extend(matches)

    
    # Remove duplicates and ensure we have valid codes
    # Only keep the most specific version (e.g., R98.5 over R98)
    extracted_codes = list(set(extracted_codes))

    # Filter out base codes if more specific ones are present
    specific_codes = set(c for c in extracted_codes if '.' in c)
    if specific_codes:
        extracted_codes = [c for c in extracted_codes if '.' in c or all(not c.startswith(sc.split('.')[0]) for sc in specific_codes)]


    
    if extracted_codes:
        query_intent["is_structured_lookup"] = True
        query_intent["lookup_type"] = "code_lookup"
        query_intent["search_terms"] = extracted_codes
        logger.info(f"Extracted codes for structured lookup: {extracted_codes}")
    
    # Check for column-specific queries with improved matching
    for col_ref, col_names in column_references.items():
        if col_ref in question_lower or col_ref in rephrased_lower:
            query_intent["is_structured_lookup"] = True
            query_intent["target_columns"].extend(col_names)
            
            # Special handling for excludes queries
            if 'excludes' in col_ref:
                query_intent["lookup_type"] = "excludes_lookup"
    
    # Check for exact match indicators
    exact_match_indicators = ['exact', 'specific', 'precise', 'find the code', 'what are the excludes']
    if any(indicator in question_lower for indicator in exact_match_indicators):
        query_intent["exact_match"] = True
    
    # IMPROVED: Better detection of structured queries
    # If we have specific ICD codes mentioned, it's likely a structured lookup
    if extracted_codes:
        query_intent["is_structured_lookup"] = True
        
    # If looking for specific information about codes, it's structured
    if any(phrase in question_lower for phrase in [
        'what is', 'what are', 'find', 'lookup', 'search for', 'get', 'show me'
    ]) and extracted_codes:
        query_intent["is_structured_lookup"] = True
    
    return query_intent

def normalize_query(query: str) -> str:
    """Normalize query for consistent processing"""
    # Convert to lowercase, remove extra spaces, normalize punctuation
    normalized = re.sub(r'\s+', ' ', query.strip().lower())
    normalized = re.sub(r'[^\w\s\-\.]', ' ', normalized)
    return normalized.strip()


def search_single_collection_with_filtering(rephrased_query: str, limit: int = None) -> List[Dict[str, Any]]:
    """Enhanced search with improved filtering and relevance scoring.
    For multi-condition queries (e.g., htn and ckd3), retrieve all relevant chunks for each condition.
    """
    if limit is None:
        limit = RAG_CONFIG["MAX_RESULTS"]

    try:
        # Generate embedding for the rephrased query
        query_embedding = get_gemini_embedding(rephrased_query)

        # Search with higher initial limit for better filtering
        initial_limit = 50  # Increase to get more diverse results

        search_result = qdrant_client.search(
            collection_name="Medical_Coder_",
            query_vector=query_embedding,
            limit=initial_limit,
            score_threshold=RAG_CONFIG["SIMILARITY_THRESHOLD"]
        )

        rephrased_query = normalize_query(rephrased_query)
        processed_results = []
        seen_texts = set()

        # --- NEW: Extract key terms for multi-condition queries ---
        # Extract all ICD codes and key medical terms from the query
        code_terms = set(re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', rephrased_query.upper()))
        # Also extract common disease/condition keywords
        condition_terms = set(re.findall(r'\b(?:hypertension|htn|ckd|chronic kidney disease|stage \d|n18\.\d)\b', rephrased_query.lower()))

        # For each result, check if it matches any code or condition term
        for result in search_result:
            text_content = result.payload.get("text", "").strip()
            if len(text_content) < RAG_CONFIG["MIN_TEXT_LENGTH"]:
                continue

            import hashlib
            original_text = result.payload.get("text", "").strip()
            text_hash = hashlib.md5(original_text.encode()).hexdigest()
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            # --- NEW: Check for code/condition match ---
            text_upper = text_content.upper()
            text_lower = text_content.lower()
            code_match = any(code in text_upper for code in code_terms)
            cond_match = any(term in text_lower for term in condition_terms)
            # If either matches, keep the chunk
            if code_terms or condition_terms:
                if not (code_match or cond_match):
                    continue

            # ...existing logic for truncation and scoring...
            max_length = RAG_CONFIG["MAX_TEXT_LENGTH"]
            if any(indicator in text_content for indicator in ['|', 'Row ', 'Lookup Code:', 'ICD Code:']):
                max_length = RAG_CONFIG["MAX_TEXT_LENGTH"] * 3
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "..."

            relevance_score = calculate_relevance_score(
                rephrased_query,
                text_content,
                result.score
            )

            file_name = result.payload.get("metadata", {}).get("file_name", "")
            source_group, source_description = get_source_info(file_name)

            processed_results.append({
                "text": text_content,
                "metadata": result.payload.get("metadata", {}),
                "score": result.score,
                "relevance_score": relevance_score,
                "source_group": source_group,
                "source_priority": 1,
                "source_description": source_description
            })

        # ...existing code for filtering, sorting, and diversity...
        quality_filtered = filter_by_content_quality(processed_results)
        quality_filtered.sort(key=lambda x: x["relevance_score"], reverse=True)
        diverse_results = apply_source_diversity(quality_filtered, limit)
        logger.info(f"Retrieved {len(diverse_results)} high-quality results from {len(search_result)} initial results")
        return diverse_results[:limit]
    except Exception as e:
        logger.error(f"Error in enhanced collection search: {e}")
        return []


def search_structured_data(query_intent: Dict[str, Any], rephrased_query: str, limit: int = None) -> List[Dict[str, Any]]:
    """Search structured data (CSV/Excel) with improved code matching - FIXED VERSION"""
    if limit is None:
        limit = RAG_CONFIG["MAX_RESULTS"]
    try:
        query_embedding = get_gemini_embedding(rephrased_query)
        search_result = qdrant_client.search(
            collection_name="Medical_Coder_",
            query_vector=query_embedding,
            limit=200,
            score_threshold=0.3
        )
        structured_results = []
        target_codes = set()
        if query_intent.get("search_terms"):
            target_codes.update(code.upper() for code in query_intent["search_terms"])

        original_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', query_intent.get("original_query", "").upper())
        target_codes.update(original_codes)
        more_specific = {code for code in target_codes if '.' in code}
        if more_specific:
            target_codes = {code for code in target_codes if '.' in code or all(not code.startswith(ms.split('.')[0]) for ms in more_specific)}
        logger.info(f"Looking for codes: {target_codes}")

        # --- NEW: For RAG3, collect all candidate chunks and pick the best match ---
        rag3_candidates = []
        for result in search_result:
            metadata = result.payload.get("metadata", {})
            file_name = metadata.get("file_name", "")
            text_content = result.payload.get("text", "").strip()
            is_rag3 = file_name == "RAG3.csv"
            code_match = False
            matched_code = None
            extracted_column_value = None

            if is_rag3 and target_codes:
                for code in target_codes:
                    # Match both Lookup Code and ICD Code in the chunk
                    lookup_match = re.search(rf"Lookup Code:\s*{re.escape(code)}\b", text_content, re.IGNORECASE)
                    icd_match = re.search(rf"ICD Code:\s*{re.escape(code)}\b", text_content, re.IGNORECASE)
                    if lookup_match or icd_match:
                        # Find which column is requested
                        col_names = query_intent.get("target_columns", [])
                        if not col_names and "exclude" in query_intent.get("original_query", "").lower():
                            col_names = ["Excludes1 Code(s)"]
                        # Try to extract the requested column(s)
                        for col in col_names:
                            match = re.search(rf"{re.escape(col)}:\s*([^|]*)", text_content, re.IGNORECASE)
                            if match:
                                extracted_column_value = match.group(1).strip()
                                break
                        # If no specific column found, try to extract Excludes1 by default
                        if not extracted_column_value:
                            match = re.search(r"Excludes1 Code\(s\):\s*([^|]*)", text_content, re.IGNORECASE)
                            if match:
                                extracted_column_value = match.group(1).strip()
                        code_match = True
                        matched_code = code
                        # Score: prefer chunks with both Lookup Code and ICD Code match
                        match_score = int(bool(lookup_match)) + int(bool(icd_match))
                        rag3_candidates.append((match_score, {
                            "text": text_content,
                            "metadata": metadata,
                            "score": result.score,
                            "relevance_score": 1.0,
                            "source_group": get_source_info(file_name)[0],
                            "source_priority": 1,
                            "source_description": get_source_info(file_name)[1],
                            "is_structured": True,
                            "code_match": code_match,
                            "matched_code": matched_code,
                            "extracted_column_value": extracted_column_value
                        }))
                        break  # Only one code per chunk
            # ...existing fallback for other files...
            elif not is_rag3:
                # For RAG1/RAG2: match code or keyword in Q/A
                found = False
                for code in target_codes:
                    if code in text_content:
                        found = True
                        break
                if found:
                    source_group, source_description = get_source_info(file_name)
                    structured_results.append({
                        "text": text_content,
                        "metadata": metadata,
                        "score": result.score,
                        "relevance_score": result.score,
                        "source_group": source_group,
                        "source_priority": 1,
                        "source_description": source_description,
                        "is_structured": False,
                        "code_match": True,
                        "matched_code": code,
                        "extracted_column_value": None
                    })
        # Pick the best RAG3 candidate(s)
        if rag3_candidates:
            rag3_candidates.sort(reverse=True, key=lambda x: x[0])  # highest match_score first
            # Only keep top N
            structured_results = [c[1] for c in rag3_candidates[:limit]] + structured_results
        return structured_results[:limit]
    except Exception as e:
        logger.error(f"Error in structured data search: {e}")
        return []

def calculate_structured_relevance(query_intent: Dict[str, Any], text_content: str, base_score: float) -> float:
    """Calculate relevance score for structured data with improved code matching - FIXED VERSION"""
    try:
        relevance_score = base_score
        
        # Extract ICD codes from query using regex
    
        query_codes = set(re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', query_intent.get("original_query", "").upper()))
        
        # Also check search terms
        if query_intent.get("search_terms"):
            for term in query_intent["search_terms"]:
                query_codes.add(term.upper())
        
        # Look for exact ICD code matches in the structured data - IMPROVED MATCHING
        text_upper = text_content.upper()
        for code in query_codes:
            # PRIORITY 1: Check for exact code match in Lookup Code field (MAIN FIX)
            if f"LOOKUP CODE: {code}" in text_upper:
                relevance_score += 0.7  # Highest boost for Lookup Code match
                break
            # PRIORITY 2: Check for exact code match in ICD Code field
            elif f"ICD CODE: {code}" in text_upper:
                relevance_score += 0.6  # High boost for ICD Code match
                break
            # PRIORITY 3: Check for pipe-separated format
            elif f"| {code} |" in text_upper:
                relevance_score += 0.5  # Good boost for structured format
                break
            # NEW: If chunk contains only one code and it's the target, boost
            else:
                found_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', text_upper)
                if len(set(found_codes)) == 1 and found_codes[0] == code:
                    relevance_score += 0.7
                    break
        
        # Boost for target column matches (like "Excludes1")
        if query_intent.get("target_columns"):
            for col in query_intent["target_columns"]:
                if col.upper() in text_upper:
                    relevance_score += 0.3
        
        # Look for specific field queries (like "excludes1")
        excludes_query = any(term in query_intent.get("original_query", "").lower() 
                           for term in ["excludes1", "excludes 1", "excludes1 code"])
        if excludes_query and "EXCLUDES1 CODE(S):" in text_upper:
            relevance_score += 0.4
        
        # Boost for structured content indicators
        structured_indicators = ['LOOKUP CODE:', 'ICD CODE:', 'CODE DESCRIPTION:', 'ROW ']
        if any(indicator in text_upper for indicator in structured_indicators):
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating structured relevance: {e}")
        return base_score

def get_source_info(file_name: str) -> tuple:
    """Get source group and description based on file name"""
    if file_name in ["RAG1.pdf", "RAG1_1.xlsx"]:
        return "Group 1", "ICD-10 Guidelines"
    elif file_name in ["RAG2.xlsx", "RAG2_1.pdf", "RAG2_2.pdf", "RAG2_3.pdf"]:
        return "Group 2", "ICD-10 Index"
    elif file_name in ["RAG3.csv"]:
        return "Group 3", "ICD-10 Tabular List"
    else:
        return "Unknown", "Unknown"

def apply_source_diversity(results: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """Apply diversity filtering to ensure balanced representation from different sources"""
    if len(results) <= target_count:
        return results
    
    # Group by source
    source_groups = {}
    for result in results:
        source = result["source_group"]
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(result)
    
    # Calculate target per source (balanced approach)
    # Sort sources for consistent ordering:
    sources = sorted(source_groups.keys())  # Add sort here
    results_per_source = max(1, target_count // len(sources))
    remainder = target_count % len(sources)
    
    diverse_results = []
    
    # Take top results from each source
    for i, source in enumerate(sources):
        source_results = source_groups[source]
        take_count = results_per_source + (1 if i < remainder else 0)
        diverse_results.extend(source_results[:take_count])
    
    # Sort final results by relevance
    diverse_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return diverse_results


# Add this at the beginning of the organize_rag_results_by_source function:
def organize_rag_results_by_source(rag_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """Organize RAG results by source type with length limits and structured data handling"""
    organized = {
        "guideline_context": "",
        "index_context": "",
        "tabular_context": "",
        "structured_data": ""  # Add this new field
    }
    
    # Use config value for max section length
    MAX_LENGTH_PER_SECTION = RAG_CONFIG["MAX_SECTION_LENGTH"]
    
    for result in rag_results:
        source_group = result.get("source_group", "")
        text = result.get("text", "").strip()
        is_structured = result.get("is_structured", False)
        
        # Truncate very long texts using config
        if len(text) > RAG_CONFIG["MAX_TEXT_LENGTH"]:
            text = text[:RAG_CONFIG["MAX_TEXT_LENGTH"]] + "..."
        
        # Handle structured data separately - prioritize complete data
        if is_structured:
            # For structured data, allow longer sections to preserve complete records
            if len(organized["structured_data"]) + len(text) < (MAX_LENGTH_PER_SECTION * 2):
                organized["structured_data"] += f"{text}\n\n"
        elif source_group == "Group 1":  # Guidelines
            if len(organized["guideline_context"]) + len(text) < MAX_LENGTH_PER_SECTION:
                organized["guideline_context"] += f"{text}\n\n"
        elif source_group == "Group 2":  # Index
            if len(organized["index_context"]) + len(text) < MAX_LENGTH_PER_SECTION:
                organized["index_context"] += f"{text}\n\n"
        elif source_group == "Group 3":  # Tabular List
            if len(organized["tabular_context"]) + len(text) < MAX_LENGTH_PER_SECTION:
                organized["tabular_context"] += f"{text}\n\n"
    
    # Clean up and provide fallback
    for key in organized:
        organized[key] = organized[key].strip()
        if not organized[key]:
            organized[key] = "No relevant information found."
    
    return organized

def generate_rag_response_with_context(user_question: str, rephrased_query: str, rag_results: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None, conversation_context: Dict[str, Any] = None) -> str:
    """Generate response using the new RAG processing prompt, with Excludes1/2 logic enforced."""
    try:
        # Use the new RAG processing prompt from the document
        system_prompt = """You are an expert ICD-10-CM medical coding assistant. When the user provides a query involving one or more diagnoses, your first step is to:

1. Identify if any diagnoses are clinically or linguistically linked (e.g., with, due to, associated with, manifestation of, secondary to, resulting in).

2. Immediately evaluate for ICD-10-CM combination codes using the Tabular List and Alphabetic Index.

3. Apply combination code logic before considering individual codes.

4. If a valid combination code exists, assign only the combination code, unless Excludes2 permits coding both.

5. If no combination code applies, proceed with assigning separate codes, following sequencing, specificity, and instructional note rules.

**Guidelines to Follow**

1. **Strict Adherence to ICD-10-CM Guidelines (2025)**
   - Use the most recent ICD-10-CM guidelines and supporting data from the Guideline, Alphabetic Index, and Tabular List.
   - If RAG data is incomplete or unavailable, rely solely on embedded ICD-10-CM knowledge.

2. **Avoid Hallucination and Assumptions**
   - Do not include unsupported information.
   - Do not infer diagnoses or relationships unless explicitly stated or medically inferable per ICD-10-CM rules.

3. **Conversation Continuity**
   - Reference prior turns for context in multi-turn queries.
   - Do not lose track of partially answered or follow-up questions.

4. **Instructional Notes to Follow**
   While interpreting the Tabular List:
   - Include Notes – Conditions covered by a code
   - Exclude1 – Mutually exclusive; do not code both
   - Exclude2 – May code both if present
   - Code First – Sequence etiology first
   - Use Additional Code – Report an additional code for cause/severity
   - Code Also – Report both when appropriate
   - Respect Laterality, Gender, and Age specificity

5. **Combination Code Enforcement**
   - Prioritize resolving combination codes when multiple diagnoses are listed.
   - Do not assign separate codes if a combination code applies, unless Exclude2 note allows it.
   - Use Alphabetic Index and Tabular List crosswalk to confirm.

6. **Specificity, Severity, and Hierarchy**
   - Always assign the most specific code (e.g., laterality, severity).
   - Use unspecified codes only when required information is truly absent.

7. **Query Types Handling**
   - Code Lookup: Provide precise code and brief description.
   - Guideline/Rule Lookup: Return clear instruction from ICD-10-CM Guideline and Tabular List.
   - General Medical Concept: Explain the term per ICD-10-CM definition, not clinical advice.

8. **Clarification for Missing Data**
   - If the query lacks specificity (e.g., type, site, laterality), use the unspecified code only if allowed and include a single clarifying question under "Clarification (if needed)".

9. **Complex & Multi-Diagnosis Handling**
   - Always evaluate the full list of diagnoses before coding.
   - Check for combination code eligibility before assigning separate codes.
   - Use correct sequencing: principal diagnosis first, then secondary and supplemental (e.g., Z3A. for gestational age if applicable).

10. **Final Validation (Pre-Output Check)**
    Before finalizing output:
    - Ensure combination code logic was fully applied.
    - Validate sequencing, specificity, and inclusion/exclusion notes.
    - Double-check that redundant codes are not included when a combination code exists.

**Structured Data Handling**
11. **CSV/Excel Data Interpretation**
    - When processing structured data (CSV/Excel files), interpret the content as tabular data with specific columns
    - For RAG3.csv (Tabular List): Use columns like 'ICD Code', 'Code Description', 'Keyword for this Code', 'Synonym', 'Excludes1/2', 'Includes', etc.
    - For Excel files with Q&A format: Interpret as question-answer pairs for guidelines and index lookups
    - Present structured data in a clear, organized format

12. **Code Lookup Enhancement**
    - When a specific ICD code is mentioned, prioritize exact matches from the tabular data
    - Cross-reference code descriptions, keywords, and synonyms for comprehensive answers
    - Include relevant excludes, includes, and coding instructions when available from structured data

**Response Format**:
**Response Format**:
- **Answer**: Provide a concise response with ICD-10 code(s) highlighted (e.g., **E11.9**) or relevant information. For structured data queries, extract and present the exact field values.
- **Rationale**: Explain the response, referencing the Guideline, Include/Exclude notes, Code also/Code first/Use Additional Code instructions, and any relevant laterality, gender, or age specificity from the Tabular List.
- **Clarification (if needed)**: Include a single question if clarification is needed for specificity or missing context; otherwise omit this section.
- **Disclaimer**: Always include: "This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder."

**Adhere to ICD-10-CM Guidelines**: Follow official coding conventions, including sequencing rules and specificity requirements, as outlined in the Guideline dataset.

**Avoid Non-ICD-10 Content**: Do not include unrelated information (e.g., general health advice or CPT) unless supported by the datasets.

**For Structured Data (CSV/Excel) Queries**:
- When processing CSV/Excel data queries, ALWAYS prioritize and use the retrieved structured data from the RAG context
- Do not rely on embedded knowledge when structured data is available
- Present the structured data information clearly, referencing the specific data fields retrieved
- If structured data context is provided, base your answer primarily on that data

**CSV Data Format Interpretation**:
When you see the ICD-10 code data in the following format:

"Row 668: Lookup Code: A92.5 | ICD Code: A92.5 | Code Description: Zika virus disease | Excludes1 Code(s): P35.4, B33.1"

Your job is to extract the relevant field based on the user's question. For example:
- If the user asks "Excludes1 codes for A92.5", return: **P35.4, B33.1**

Never say "No Excludes1 codes listed" unless the value after "Excludes1 Code(s):" is explicitly missing."""

        # Format conversation history
        conversation_history_text = format_conversation_history_for_prompt(conversation_history)
        
        # Organize RAG results by source
        organized_context = organize_rag_results_by_source(rag_results)


        # Update the user_message in generate_rag_response_with_context:
        user_message = f"""**Conversation History**: {conversation_history_text}

**Rephrased User Query**: {rephrased_query}

**Retrieved Context**:

- **Guideline**: {organized_context['guideline_context']}

- **Alphabetic Index**: {organized_context['index_context']}

- **Tabular List**: {organized_context['tabular_context']}

- **Structured Data**: {organized_context['structured_data']}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # --- ENHANCED: Enforce Excludes1/2 logic, always keep the code that lists the other in its Excludes1 ---
    
        code_info = {}
        for result in rag_results:
            text = result.get("text", "")
            codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', text)
            excludes1 = re.findall(r'Excludes1 Code\(s\):\s*([A-Z0-9\.,\s]*)', text)
            excludes2 = re.findall(r'Excludes2 Code\(s\):\s*([A-Z0-9\.,\s]*)', text)
            for code in codes:
                code_info.setdefault(code, {"excludes1": set(), "excludes2": set()})
                for ex in excludes1:
                    code_info[code]["excludes1"].update([c.strip() for c in ex.split(",") if c.strip()])
                for ex in excludes2:
                    code_info[code]["excludes2"].update([c.strip() for c in ex.split(",") if c.strip()])

        all_codes = set(code_info.keys())
        codes_to_output = set(all_codes)
        # --- ENHANCED LOGIC: If code2 is listed in code1's Excludes1, keep only code1 ---
        for code1 in all_codes:
            for code2 in all_codes:
                if code1 == code2:
                    continue
                if code2 in code_info.get(code1, {}).get("excludes1", set()):
                    # code2 is excluded by code1, so keep code1, remove code2
                    codes_to_output.discard(code2)

        # Compose a context string from all RAG results, but only for allowed codes
        context_lines = []
        for result in rag_results:
            text = result.get("text", "").strip()
            codes_in_chunk = set(re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', text))
            # Only include info for codes that are not excluded
            if codes_in_chunk & codes_to_output:
                file_name = result.get("metadata", {}).get("file_name", "")
                code_matches = ", ".join(sorted(codes_in_chunk & codes_to_output))
                # --- NEW: Add more accurate extraction for description and details ---
                desc = ""
                detail = ""
                # Try to extract the most relevant description (prefer exact code match)
                for code in sorted(codes_in_chunk & codes_to_output):
                    desc_match = re.search(rf"{code}.*?Code Description:\s*([^|]*)", text)
                    if not desc_match:
                        desc_match = re.search(r"Code Description:\s*([^|]*)", text)
                    if desc_match:
                        desc = desc_match.group(1).strip()
                        break
                # Try to extract the most relevant answer/detail (prefer exact code match)
                for code in sorted(codes_in_chunk & codes_to_output):
                    ans_match = re.search(rf"{code}.*?Answer:\s*(.*)", text)
                    if not ans_match:
                        ans_match = re.search(r"Answer:\s*(.*)", text)
                    if ans_match:
                        detail = ans_match.group(1).strip()
                        break
                if code_matches:
                    context_lines.append(f"Codes: {code_matches}")
                if desc:
                    context_lines.append(f"Description: {desc}")
                if detail:
                    context_lines.append(f"Detail: {detail}")
                if file_name:
                    context_lines.append(f"Source: {file_name}")

        if not context_lines:
            context_lines.append("No ICD-10 codes found in the retrieved data.")

        # --- NEW: Remove duplicate context lines for accuracy ---
        seen = set()
        unique_context_lines = []
        for line in context_lines:
            if line not in seen:
                unique_context_lines.append(line)
                seen.add(line)

        gemini_system_prompt = (
            "You are an expert ICD-10-CM medical coding assistant. "
            "You are given ONLY the following retrieved context from official ICD-10-CM sources (Guideline, Index, Tabular List). "
            "Your job is to summarize and format the answer using ONLY the provided context. "
            "Do NOT use any outside knowledge or make up codes. "
            "If the context is insufficient, say so. "
            "If two codes are mutually exclusive per Excludes1, only output the code that lists the other in its Excludes1 (i.e., keep the code with the Excludes1 note, drop the excluded code). "
            "Always include a rationale and the disclaimer: "
            "\"This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder.\""
        )
        gemini_user_message = (
            f"User question: {user_question}\n\n"
            f"Retrieved context:\n" +
            "\n".join(unique_context_lines)
        )
        messages = [
            {"role": "system", "content": gemini_system_prompt},
            {"role": "user", "content": gemini_user_message}
        ]
        return generate_gemini_response(messages, temperature=0.1, max_tokens=1024)
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return "I'm sorry, I encountered an error while generating your answer. Please try again."
    

def generate_general_response(question: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Generate response for non-medical queries with conversation awareness"""
    try:
        system_prompt = (
            "You are a friendly and professional medical coding assistant. "
            "For general queries and greetings, provide helpful and welcoming responses. "
            "If the conversation turns to medical coding topics, inform the user that you can help with ICD-10-CM coding questions. "
            "Consider the conversation history to maintain continuity and provide contextually appropriate responses."
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Include more context for general chat
        
        messages.append({"role": "user", "content": question})

        return generate_gemini_response(messages, temperature=0.7, max_tokens=512)

    except Exception as e:
        logger.error(f"Error generating general response: {e}")
        return "Hello! I'm here to help with your ICD-10 coding questions. How can I assist you today?"

async def get_conversation_history(chat_id: str, user_id: str, limit: int = 50):  # Increased limit for full conversation
    """Retrieve full conversation history from Supabase"""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.getenv("SUPABASE_URL", "https://ilnnwhsktxtuwhkcbaup.supabase.co")
        supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlsbm53aHNrdHh0dXdoa2NiYXVwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU4MDkwMDEsImV4cCI6MjA2MTM4NTAwMX0.tL6-RiUQJykGwzss_mZ5-LUB6XbqeTu4ihs89jd7OKs")
        supabase_table = os.getenv("SUPABASE_TABLE_NAME", "chathistory")
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        response = supabase.table(supabase_table)\
            .select("*")\
            .eq("chat_id", chat_id)\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        if not response.data:
            return []
        
        history = []
        for msg in reversed(response.data):
            history.append({"role": "user", "content": msg["user_message"]})
            history.append({"role": "assistant", "content": msg["ai_message"]})
        
        return history
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return []

async def save_conversation_message(chat_id: str, user_id: str, user_message: str, ai_message: str):
    """Save conversation message to Supabase"""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.getenv("SUPABASE_URL", "https://ilnnwhsktxtuwhkcbaup.supabase.co")
        supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlsbm53aHNrdHh0dXdoa2NiYXVwIiwicm9zZSI6ImFub24iLCJpYXQiOjE3NDU4MDkwMDEsImV4cCI6MjA2MTM4NTAwMX0.tL6-RiUQJykGwzss_mZ5-LUB6XbqeTu4ihs89jd7OKs")
        supabase_table = os.getenv("SUPABASE_TABLE_NAME", "chathistory")
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        data = {
            "chat_id": chat_id,
            "user_id": user_id,
            "user_message": user_message,
            "ai_message": ai_message
        }
        
        supabase.table(supabase_table).insert(data).execute()
        
    except Exception as e:
        logger.error(f"Error saving conversation message: {e}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat API endpoint with new prompts"""
    try:
        # Get conversation history
        conversation_history = []
        if request.chat_id and not request.is_new_chat:
            conversation_history = await get_conversation_history(request.chat_id, request.user_id, limit=50)
        
        # Extract conversation context
        conversation_context = extract_conversation_context(conversation_history)
        # Determine if this is a follow-up query
        is_follow_up = is_follow_up_query(request.question, conversation_history)
        
        # Check if it's a medical query
        is_medical = is_medical_query(request.question) or (is_follow_up and any("medical" in topic.lower() for topic in conversation_context.get("topics_discussed", [])))
        
        if not is_medical:
            answer = generate_general_response(request.question, conversation_history)
            rag_results = []
            rephrased_query = None
        else:
            # Rephrase the user input using the new preprocessing prompt
            rephrased_query = structure_user_input_with_context(request.question, conversation_context, conversation_history)
            logger.info(f"Rephrased query: {rephrased_query}")
            
            # Extract structured query intent
            query_intent = extract_structured_query_intent(request.question, rephrased_query)
            
            rag_results = []  # ✅ Initialize early
            
            # Choose search strategy based on query intent
            if query_intent["is_structured_lookup"]:
                logger.info("Using structured data search with 0.8 threshold")
                rag_results = search_structured_data(query_intent, rephrased_query)

                logger.info(f"Structured search returned {len(rag_results)} results")
                
                # ✅ Early extraction and direct response if exact match found
                if rag_results and query_intent.get("search_terms") and len(query_intent["search_terms"]) > 0 and query_intent["search_terms"][0]:
                    logger.info(f"Search terms: {query_intent['search_terms']}")
                    logger.info(f"First search term: {query_intent['search_terms'][0]}")
                    matched_code = query_intent["search_terms"][0].upper()

                    structured_match = next(
                        (r for r in rag_results 
                        if r.get("is_structured") and (r.get("matched_code") or "").upper() == matched_code),
                        None
                    )

                    # --- NEW: If not found by matched_code, try fallback for single-code-per-chunk ---
                    if not structured_match:
                        for r in rag_results:
                            text = r.get("text", "").upper()
                            found_codes = set(re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', text))
                            if len(set(found_codes)) == 1 and matched_code in found_codes:
                                structured_match = r
                                break

                    if structured_match:
        
                        text = structured_match.get("text", "").strip()
                        extracted_column_value = structured_match.get("extracted_column_value")
                        query_text = request.question.lower()
                        # Detect which field(s) the user asked for
                        field_keywords = {
                            "description_value": ["description"],
                            "excludes1_value": ["excludes1", "exclude 1", "excludes 1"],
                            "excludes2_value": ["excludes2", "exclude 2", "excludes 2"], 
                            "includes_value": ["includes"],
                            "code_first_value": ["code first"],
                            "use_additional_code_value": ["use additional code"],
                            "code_also_value": ["code also"],
                            "keyword_value": ["keyword"],
                            "synonym_value": ["synonym"]
                        }
                        asked_fields = []
                        for field, keywords in field_keywords.items():
                            if any(k in query_text for k in keywords):
                                asked_fields.append(field)
                        # --- NEW: Always extract all relevant fields, not just the first one ---
                        fields_to_extract = {
                            "Excludes1 Code(s)": "excludes1_value",
                            "Excludes2 Code(s)": "excludes2_value",
                            "Includes Code(s)": "includes_value",
                            "Code First": "code_first_value",
                            "Use Additional Code": "use_additional_code_value",
                            "Code Also": "code_also_value",
                            "Synonym": "synonym_value",
                            "Keyword for this Code": "keyword_value",
                            "Code Description": "description_value"
                        }
                        # Extract all fields from the chunk
                        extracted_fields = {}
                        for field_label, var_name in fields_to_extract.items():
                            match = re.search(rf"{re.escape(field_label)}:\s*([^|]*)", text, re.IGNORECASE)
                            extracted_fields[var_name] = match.group(1).strip() if match else None
                        # For single-code-per-chunk, also try extracting fields line by line
                        if not any(extracted_fields.values()):
                            for line in text.splitlines():
                                for field_label, var_name in fields_to_extract.items():
                                    if line.strip().lower().startswith(field_label.lower()):
                                        val = line.split(":", 1)[-1].strip()
                                        if val:
                                            extracted_fields[var_name] = val
                        # If user asked for specific fields, only show those, else show all
                        answer_lines = [f"**Answer**:"]
                        found = False
                        if asked_fields:
                            for field in asked_fields:
                                label = field.replace("_value", "").replace("_", " ").title()
                                val = extracted_fields.get(field)
                                if val:
                                    answer_lines.append(f"{label} for **{matched_code}**: **{val}**")
                                    found = True
                        else:
                            # Show all available fields for the code
                            for field, label in [
                                ("description_value", "Description"),
                                ("excludes1_value", "Excludes1"),
                                ("excludes2_value", "Excludes2"),
                                ("includes_value", "Includes"),
                                ("code_first_value", "Code First"),
                                ("use_additional_code_value", "Use Additional Code"),
                                ("code_also_value", "Code Also"),
                                ("keyword_value", "Keyword"),
                                ("synonym_value", "Synonym")
                            ]:
                                val = extracted_fields.get(field)
                                if val:
                                    answer_lines.append(f"{label} for **{matched_code}**: **{val}**")
                                    found = True
                        if not found:
                            answer_lines.append(f"No structured information found for {matched_code} matching your request.")
                        formatted_answer = "\n\n".join([
                            "\n".join(answer_lines),
                            "**Rationale**: Extracted from the structured ICD-10 tabular list (RAG3.csv).",
                            "**Clarification (if needed)**: N/A",
                            "**Disclaimer**: This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder."
                        ])
                        await save_conversation_message(request.chat_id, request.user_id, request.question, formatted_answer)
                        return ChatResponse(
                            answer=formatted_answer,
                            sources=[structured_match],
                            structured_query={"rephrased_query": rephrased_query},
                            conversation_context={
                                "is_follow_up": is_follow_up,
                                "conversation_length": len(conversation_history),
                                "context_extracted": conversation_context
                            }
                        )
                    else:
                        # Safe extraction of matched_code with fallback
                        matched_code = (
                            query_intent["search_terms"][0].upper()
                            if query_intent.get("search_terms") and len(query_intent["search_terms"]) > 0 and query_intent["search_terms"][0]
                            else "UNKNOWN"
                        )

                        fallback_msg = "\n\n".join([
                            f"**Answer**: No structured data found for ICD-10 code: {matched_code}",
                            "**Rationale**: The code either does not exist in the structured database or lacks Excludes1 information.",
                            "**Clarification (if needed)**: Please check if the code is spelled correctly or try another code.",
                            "**Disclaimer**: This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder."
                        ])

                        await save_conversation_message(request.chat_id, request.user_id, request.question, fallback_msg)

                        return ChatResponse(
                            answer=fallback_msg,
                            sources=rag_results[:1] if rag_results else [],
                            structured_query={"rephrased_query": rephrased_query},
                            conversation_context={
                                "is_follow_up": is_follow_up,
                                "conversation_length": len(conversation_history),
                                "context_extracted": conversation_context
                            }
                        )
                else:
                    # Handle case where search_terms is empty or None
                    fallback_msg = "\n\n".join([
                        "**Answer**: Unable to identify a valid ICD-10 code from your query.",
                        "**Rationale**: The query could not be parsed to extract a specific ICD-10 code.",
                        "**Clarification (if needed)**: Please provide a specific ICD-10 code (e.g., 'J44.0', 'E11.9') for structured lookup.",
                        "**Disclaimer**: This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder."
                    ])

                    await save_conversation_message(request.chat_id, request.user_id, request.question, fallback_msg)

                    return ChatResponse(
                        answer=fallback_msg,
                        sources=[],
                        structured_query={"rephrased_query": rephrased_query},
                        conversation_context={
                            "is_follow_up": is_follow_up,
                            "conversation_length": len(conversation_history),
                            "context_extracted": conversation_context
                        }
                    )

                # Fallback to general search if structured failed
                if len(rag_results) == 0:
                    logger.info("No structured results found, trying general search")
                    enhanced_query = enhance_query_for_retrieval(rephrased_query)
                    rag_results = search_single_collection_with_filtering(enhanced_query, limit=3)
            else:
                # Use general search for non-structured medical queries
                enhanced_query = enhance_query_for_retrieval(rephrased_query)
                rag_results = search_single_collection_with_filtering(enhanced_query)

            logger.info(f"Retrieved {len(rag_results)} results from RAG sources")

            # ✅ Final fallback: Only generate answer if RAG results exist
            if rag_results:
                # --- NEW: Compose a context string from all RAG results and let Gemini rephrase only ---
                # Build a context string from all RAG results (codes, descriptions, details)
                context_lines = []
                for result in rag_results:
                    text = result.get("text", "").strip()
                    file_name = result.get("metadata", {}).get("file_name", "")
                    # Try to extract code(s) from the chunk
               
                    code_matches = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', text)
                    codes_in_chunk = ", ".join(sorted(set(code_matches)))
                    if codes_in_chunk:
                        context_lines.append(f"Codes: {codes_in_chunk}")
                    # Add description or answer if available
                    if "Code Description:" in text:
                        desc_match = re.search(r"Code Description:\s*([^|]*)", text)
                        if desc_match:
                            context_lines.append(f"Description: {desc_match.group(1).strip()}")
                    if "Answer:" in text:
                        ans_match = re.search(r"Answer:\s*(.*)", text)
                        if ans_match:
                            context_lines.append(f"Detail: {ans_match.group(1).strip()}")
                    # Optionally, add file/source info
                    if file_name:
                        context_lines.append(f"Source: {file_name}")
                if not context_lines:
                    context_lines.append("No ICD-10 codes found in the retrieved data.")

                # Compose a prompt for Gemini to only rephrase/format the answer, not to hallucinate
                gemini_system_prompt = (
                    "You are an expert ICD-10-CM medical coding assistant. "
                    "You are given ONLY the following retrieved context from official ICD-10-CM sources (Guideline, Index, Tabular List). "
                    "Your job is to summarize and format the answer using ONLY the provided context. "
                    "Do NOT use any outside knowledge or make up codes. "
                    "If the context is insufficient, say so. "
                    "If two codes are mutually exclusive per Excludes1, only output the code that lists the other in its Excludes1 (i.e., keep the code with the Excludes1 note, drop the excluded code). "
                    "Always include a rationale and the disclaimer: "
                    "\"This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder.\""
                )
                gemini_user_message = (
                    f"User question: {request.question}\n\n"
                    f"Retrieved context:\n" +
                    "\n".join(context_lines)
                )
                messages = [
                    {"role": "system", "content": gemini_system_prompt},
                    {"role": "user", "content": gemini_user_message}
                ]
                answer = generate_gemini_response(messages, temperature=0.2, max_tokens=1024)
            else:
                # --- Fallback: No RAG results, do NOT use Gemini knowledge, just return not found ---
                answer = "I'm sorry, I couldn't find relevant ICD-10 information for your query in the available datasets. Please verify the code or try rephrasing."

        # Save the conversation message
        await save_conversation_message(request.chat_id, request.user_id, request.question, answer)
        
        # Prepare sources information (now deduplicated)
        sources = []
        for result in rag_results:
            metadata = result["metadata"]
            sources.append({
                "file_name": metadata.get("file_name", "Unknown"),
                "source_group": result["source_group"],
                "source_description": result["source_description"],
                "source_priority": result["source_priority"],
                "score": result["score"],
                "text": result["text"],
                "metadata": metadata
            })
        
        return ChatResponse(
            answer=answer, 
            sources=sources,
            structured_query={"rephrased_query": rephrased_query} if rephrased_query else None,
            conversation_context={
                "is_follow_up": is_follow_up,
                "conversation_length": len(conversation_history),
                "context_extracted": conversation_context
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            answer="I'm sorry, I encountered an error while processing your request. Please try again.",
            sources=[],
            structured_query=None,
            conversation_context=None
        )

@app.post("/api/new-chat")
async def create_new_chat():
    """Create a new chat session"""
    import uuid
    new_chat_id = str(uuid.uuid4())
    return {"chat_id": new_chat_id, "message": "New chat session created"}

@app.get("/api/chat-history/{chat_id}")
async def get_chat_history(chat_id: str, user_id: str = "default_user"):
    """Get conversation history for a specific chat"""
    try:
        history = await get_conversation_history(chat_id, user_id, limit=100)
        return {"chat_id": chat_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return {"error": "Failed to retrieve chat history"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "groups": list(DOCUMENT_GROUPS.keys())}

from fastapi import Path

@app.get("/api/code-info/{code}")
async def get_code_info(
    code: str = Path(..., description="ICD-10 code to search for"),
    user_id: str = "default_user"
):
    """
    Search all RAG sources for the given ICD-10 code and return a formatted response.
    """
    try:
        # Normalize code
        code = code.strip().upper()
        # Prepare a synthetic user question and rephrased query
        user_question = f"Find all ICD-10 information for code {code}"
        rephrased_query = code

        # Build a synthetic query_intent for structured search
        query_intent = {
            "is_structured_lookup": True,
            "lookup_type": "code_lookup",
            "search_terms": [code],
            "target_columns": [],
            "exact_match": True,
            "original_query": user_question
        }

        # Search structured data (tabular)
        structured_results = search_structured_data(query_intent, rephrased_query, limit=5)

        # Search guidelines and index (unstructured)
        # Use enhanced_query to get more context
        enhanced_query = enhance_query_for_retrieval(rephrased_query)
        guideline_results = search_single_collection_with_filtering(enhanced_query, limit=5)

        # Merge all results, deduplicate by text hash
        import hashlib
        seen_hashes = set()
        all_results = []
        for result in (structured_results + guideline_results):
            text = result.get("text", "")
            h = hashlib.md5(text.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_results.append(result)

        # Organize by source for prompt
        organized_context = organize_rag_results_by_source(all_results)

        # Compose a synthetic conversation history for context (optional)
        conversation_history = [
            {"role": "user", "content": user_question}
               ]

        # Use the same RAG response generator as chat
        answer = generate_rag_response_with_context(
            user_question,
            rephrased_query,
            all_results,
            conversation_history=conversation_history,
            conversation_context=None
        )

        # Prepare sources for output
        sources = []
        for result in all_results:
            metadata = result.get("metadata", {})
            sources.append({
                "file_name": metadata.get("file_name", "Unknown"),
                "source_group": result.get("source_group"),
                "source_description": result.get("source_description"),
                "source_priority": result.get("source_priority"),
                "score": result.get("score"),
                "text": result.get("text"),
                "metadata": metadata
            })

        return {
            "code": code,
            "answer": answer,
            "sources": sources,
            "organized_context": organized_context
        }
    except Exception as e:
        logger.error(f"Error in code-info endpoint: {e}")
        return {
            "code": code,
            "answer": "I'm sorry, I couldn't retrieve information for this code.",
            "sources": [],
            "organized_context": {}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)
