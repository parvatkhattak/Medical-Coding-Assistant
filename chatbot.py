import os
import logging
from typing import List, Dict, Any, Optional
import json

from fastapi import FastAPI
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
        import re
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
        "collection":"Medical_Coder",
        "files": ["RAG1.pdf", "RAG1_1.xlsx"],
        "description": "ICD-10 Guidelines",
        "priority": 1
    },
    "Group 2": {
        "collection":"Medical_Coder",
        "files": ["RAG2.xlsx", "RAG2_1.pdf", "RAG2_2.pdf", "RAG2_3.pdf"],
        "description": "ICD-10 Index",
        "priority": 1
    },
    "Group 3": {
        "collection":"Medical_Coder",
        "files": ["RAG3.csv"],
        "description": "ICD-10 Tabular List",
        "priority": 1
    }
}


RAG_CONFIG = {
    "MAX_RESULTS": 9,           # Maximum results to retrieve
    "MIN_TEXT_LENGTH": 33,      # Minimum text length for quality filtering
    "SIMILARITY_THRESHOLD": 0.6, # Minimum similarity score
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
    import re
    code_patterns = [
        r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b',  # ICD-10 codes (e.g., J15.212, E11.9)
        r'\b\d{5}(?:-\d{2})?\b',          # CPT codes
        r'\b[A-Z]\d{2}-[A-Z]\d{2}\b'     # ICD-10 ranges
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
                import re
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
    """Enhanced query enhancement with better medical term mapping"""
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
        
        # Add ICD-10 context keywords for better retrieval
        icd_context_terms = ['icd-10', 'code', 'diagnosis', 'classification']
        
        # If the query doesn't contain coding-specific terms, add them
        if not any(term in enhanced_query for term in icd_context_terms):
            # Always add context for medical queries in a consistent way
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
    import re
    
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
        'code first': ['Code First']
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
    import re
    # Convert to lowercase, remove extra spaces, normalize punctuation
    normalized = re.sub(r'\s+', ' ', query.strip().lower())
    normalized = re.sub(r'[^\w\s\-\.]', ' ', normalized)
    return normalized.strip()


def search_single_collection_with_filtering(rephrased_query: str, limit: int = None) -> List[Dict[str, Any]]:
    """Enhanced search with improved filtering and relevance scoring"""
    if limit is None:
        limit = RAG_CONFIG["MAX_RESULTS"]
    
    try:
        # Generate embedding for the rephrased query
        query_embedding = get_gemini_embedding(rephrased_query)
        
        # Search with higher initial limit for better filtering
        # Make initial_limit consistent:
        initial_limit = 30  # Fixed value instead of min(50, limit * 10)
        
        search_result = qdrant_client.search(
            collection_name="Medical_Coder",
            query_vector=query_embedding,
            limit=initial_limit,
            score_threshold=RAG_CONFIG["SIMILARITY_THRESHOLD"]  # Use config value
        )
        
        # In search_single_collection_with_filtering(), after the try block:
        rephrased_query = normalize_query(rephrased_query)
        # Process and score results
        processed_results = []
        seen_texts = set()
        
        for result in search_result:
            text_content = result.payload.get("text", "").strip()
            
            # Skip empty or very short content using config
            if len(text_content) < RAG_CONFIG["MIN_TEXT_LENGTH"]:
                continue

            import hashlib
            original_text = result.payload.get("text", "").strip()
            text_hash = hashlib.md5(original_text.encode()).hexdigest()

            if text_hash not in seen_texts and len(original_text) >= RAG_CONFIG["MIN_TEXT_LENGTH"]:
                seen_texts.add(text_hash)
                
            # For structured data, preserve more content
            text_content = original_text
            max_length = RAG_CONFIG["MAX_TEXT_LENGTH"]

            # Check if this looks like structured data (CSV format)
            if any(indicator in text_content for indicator in ['|', 'Row ', 'Lookup Code:', 'ICD Code:']):
                max_length = RAG_CONFIG["MAX_TEXT_LENGTH"] * 3  # Allow 3x length for structured data

            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "..."
                
                # Calculate enhanced relevance score
                relevance_score = calculate_relevance_score(
                    rephrased_query, 
                    text_content, 
                    result.score
                )
                
                # Determine source group
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
        
        # Filter by content quality using config
        quality_filtered = filter_by_content_quality(processed_results)
        
        # Sort by relevance score (combination of similarity and keyword matching)
        quality_filtered.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply diversity filtering to avoid too many results from same source
        diverse_results = apply_source_diversity(quality_filtered, limit)
        
        logger.info(f"Retrieved {len(diverse_results)} high-quality results from {len(search_result)} initial results")
        
        return diverse_results[:limit]  # Return only the requested number
        
    except Exception as e:
        logger.error(f"Error in enhanced collection search: {e}")
        return []


def search_structured_data(query_intent: Dict[str, Any], rephrased_query: str, limit: int = None) -> List[Dict[str, Any]]:
    """Search structured data (CSV/Excel) with improved code matching - FIXED VERSION"""
    if limit is None:
        limit = RAG_CONFIG["MAX_RESULTS"]
    
    try:
        # Get initial results from vector search with lower threshold for structured data
        query_embedding = get_gemini_embedding(rephrased_query)
        
        # Use a much lower threshold for structured data to ensure we get results
        search_result = qdrant_client.search(
            collection_name="Medical_Coder",
            query_vector=query_embedding,
            limit=200,  # Increased to get more results for better filtering
            score_threshold=0.3  # Even lower threshold to ensure we get all potential matches
        )
        
        structured_results = []
        
        # Extract target codes from query for better matching
        target_codes = set()
        if query_intent.get("search_terms"):
            target_codes.update(code.upper() for code in query_intent["search_terms"])
        
        # Also extract codes from original query
        import re
        original_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', query_intent.get("original_query", "").upper())
        target_codes.update(original_codes)
        # Remove base codes if more specific codes exist
        more_specific = {code for code in target_codes if '.' in code}
        if more_specific:
            target_codes = {code for code in target_codes if '.' in code or all(not code.startswith(ms.split('.')[0]) for ms in more_specific)}

        
        logger.info(f"Looking for codes: {target_codes}")
        
        for result in search_result:
            metadata = result.payload.get("metadata", {})
            file_name = metadata.get("file_name", "")
            text_content = result.payload.get("text", "").strip()
            
            # Focus on CSV and Excel files
            if not (file_name.endswith('.csv') or file_name.endswith('.xlsx')):
                continue
            
            # Check if any target codes are in this result - IMPROVED MATCHING
            text_upper = text_content.upper()
            code_match = False
            matched_code = None
            
            import re

            # Try matching target codes
            for code in target_codes:
                # PRIORITY 1: Exact regex match using ICD CODE or LOOKUP CODE label
                pattern = rf'\b(?:ICD CODE|LOOKUP CODE):\s*{re.escape(code)}\b'
                if re.search(pattern, text_upper):
                    code_match = True
                    matched_code = code
                    logger.info(f"✅ Regex exact match found for {code}")
                    break

                # PRIORITY 2: Exact match in Lookup Code field (fallback)
                if f"LOOKUP CODE: {code}" in text_upper:
                    code_match = True
                    matched_code = code
                    logger.info(f"✅ Found exact LOOKUP CODE match for {code}")
                    break

                # PRIORITY 3: Exact match in ICD Code field
                elif f"ICD CODE: {code}" in text_upper:
                    code_match = True
                    matched_code = code
                    logger.info(f"✅ Found exact ICD CODE match for {code}")
                    break

                # PRIORITY 4: Pipe-delimited match (e.g., | M25.649 |)
                elif f"| {code} |" in text_upper:
                    code_match = True
                    matched_code = code
                    logger.info(f"✅ Found pipe-separated match for {code}")
                    break

            # ✅ Reject partial matches (e.g., M25 when M25.649 was requested)
            if matched_code and matched_code not in target_codes:
                logger.info(f"❌ Rejected partial match: {matched_code} not in {target_codes}")
                continue


            # If looking for excludes and this result has excludes info, include it
            excludes_match = False
            if query_intent.get("lookup_type") == "excludes_lookup":
                if "EXCLUDES1 CODE(S):" in text_upper or "EXCLUDES2 CODE(S):" in text_upper:
                    excludes_match = True
            
            # Calculate enhanced scoring
            relevance_score = calculate_structured_relevance(
                query_intent, text_content, result.score
            )
            
            # Boost score significantly for exact code matches
            if code_match:
                relevance_score = min(relevance_score + 1.0, 1.0)

            
            # Include if we have a good match
            if code_match or excludes_match or relevance_score > 0.6:  # Raised threshold
                source_group, source_description = get_source_info(file_name)
                
                structured_results.append({
                    "text": text_content,
                    "metadata": metadata,
                    "score": result.score,
                    "relevance_score": relevance_score,
                    "source_group": source_group,
                    "source_priority": 1,
                    "source_description": source_description,
                    "is_structured": True,
                    "code_match": code_match,
                    "matched_code": matched_code  # Track which code was matched
                })
        
        # Sort by relevance and code match priority
        structured_results.sort(key=lambda x: (x["code_match"], x["relevance_score"]), reverse=True)
        
        logger.info(f"Structured search found {len(structured_results)} results")

        # ✅ Final strict filter: if exact code match (like M25.649) exists, drop partial ones like M25
        strict_matches = [r for r in structured_results if r.get("matched_code") in target_codes]
        if any(r.get("matched_code") and '.' in r.get("matched_code") for r in strict_matches):
            structured_results = strict_matches
            logger.info("✅ Filtered out base code matches because more specific matches exist")

        return structured_results[:limit]
        
    except Exception as e:
        logger.error(f"Error in structured data search: {e}")
        return []


def calculate_structured_relevance(query_intent: Dict[str, Any], text_content: str, base_score: float) -> float:
    """Calculate relevance score for structured data with improved code matching - FIXED VERSION"""
    try:
        relevance_score = base_score
        
        # Extract ICD codes from query using regex
        import re
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
    """Generate response using the new RAG processing prompt"""
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

        return generate_gemini_response(messages, temperature=0.3, max_tokens=1536)

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
        supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlsbm53aHNrdHh0dXdoa2NiYXVwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU4MDkwMDEsImV4cCI6MjA2MTM4NTAwMX0.tL6-RiUQJykGwzss_mZ5-LUB6XbqeTu4ihs89jd7OKs")
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
        # Replace this section in the chat endpoint:
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

                                        
                    if structured_match:
                        import re
                        text = structured_match.get("text", "").strip()

                        # Try to extract Excludes1 codes
                        excludes1_match = re.search(r"Lookup Code:\s*{}\s*\|.*?Excludes1 Code\(s\):\s*([^|]*)".format(re.escape(matched_code)), text, re.IGNORECASE | re.DOTALL)

                        if excludes1_match and excludes1_match.group(1).strip():
                            excludes1_value = excludes1_match.group(1).strip()
                        else:
                            excludes1_value = "No Excludes1 codes listed for this code."


                        # Format answer
                        formatted_answer = "\n\n".join([
                            f"**Answer**: Excludes1 codes for {matched_code} are: {excludes1_value}",
                            f"**Rationale**: Extracted directly from the structured ICD-10 tabular entry for {matched_code}.",
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
                answer = generate_rag_response_with_context(
                    request.question,
                    rephrased_query,
                    rag_results,
                    conversation_history,
                    conversation_context
                )
            else:
                answer = "I'm sorry, I couldn't find relevant ICD-10 information for your query. Please verify the code or try rephrasing."

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)
