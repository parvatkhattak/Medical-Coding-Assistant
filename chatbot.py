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
    "SIMILARITY_THRESHOLD": 0.544, # Minimum similarity score
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
        # Fallback: return a zero vector of appropriate dimension (768 for text-embedding-004)
        return [0.0] * 768

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
            # Check if it's asking for codes
            if any(phrase in enhanced_query for phrase in ['what are the', 'correct codes', 'icd codes']):
                enhanced_query += ' icd-10 code diagnosis'
        
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

        rephrased_query = generate_gemini_response(messages, temperature=0.3, max_tokens=512)
        
        return rephrased_query.strip()

    except Exception as e:
        logger.error(f"Error rephrasing user input: {e}")
        return question

def search_single_collection_with_filtering(rephrased_query: str, limit: int = None) -> List[Dict[str, Any]]:
    """Enhanced search with improved filtering and relevance scoring"""
    if limit is None:
        limit = RAG_CONFIG["MAX_RESULTS"]
    
    try:
        # Generate embedding for the rephrased query
        query_embedding = get_gemini_embedding(rephrased_query)
        
        # Search with higher initial limit for better filtering
        initial_limit = min(50, limit * 10)  # Get more results initially
        
        search_result = qdrant_client.search(
            collection_name="Medical_Coder",
            query_vector=query_embedding,
            limit=initial_limit,
            score_threshold=RAG_CONFIG["SIMILARITY_THRESHOLD"]  # Use config value
        )
        
        # Process and score results
        processed_results = []
        seen_texts = set()
        
        for result in search_result:
            text_content = result.payload.get("text", "").strip()
            
            # Skip empty or very short content using config
            if len(text_content) < RAG_CONFIG["MIN_TEXT_LENGTH"]:
                continue
            
            # Truncate text if it exceeds max length
            if len(text_content) > RAG_CONFIG["MAX_TEXT_LENGTH"]:
                text_content = text_content[:RAG_CONFIG["MAX_TEXT_LENGTH"]] + "..."
            
            # Create hash for deduplication
            import hashlib
            text_hash = hashlib.md5(text_content.encode()).hexdigest()
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                
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
    sources = list(source_groups.keys())
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


def organize_rag_results_by_source(rag_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """Organize RAG results by source type with length limits"""
    organized = {
        "guideline_context": "",
        "index_context": "",
        "tabular_context": ""
    }
    
    # Use config value for max section length
    MAX_LENGTH_PER_SECTION = RAG_CONFIG["MAX_SECTION_LENGTH"]
    
    for result in rag_results:
        source_group = result.get("source_group", "")
        text = result.get("text", "").strip()
        
        # Truncate very long texts using config
        if len(text) > RAG_CONFIG["MAX_TEXT_LENGTH"]:
            text = text[:RAG_CONFIG["MAX_TEXT_LENGTH"]] + "..."
        
        if source_group == "Group 1":  # Guidelines
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

**Response Format**:
- **Answer**: Provide a concise response with ICD-10 code(s) highlighted (e.g., **E11.9**) or relevant information.
- **Rationale**: Explain the response, referencing the Guideline, Include/Exclude notes, Code also/Code first/Use Additional Code instructions, and any relevant laterality, gender, or age specificity from the Tabular List.
- **Clarification (if needed)**: Include a single question if clarification is needed for specificity or missing context; otherwise omit this section.
- **Disclaimer**: Always include: "This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder."

**Adhere to ICD-10-CM Guidelines**: Follow official coding conventions, including sequencing rules and specificity requirements, as outlined in the Guideline dataset.

**Avoid Non-ICD-10 Content**: Do not include unrelated information (e.g., general health advice or CPT) unless supported by the datasets."""

        # Format conversation history
        conversation_history_text = format_conversation_history_for_prompt(conversation_history)
        
        # Organize RAG results by source
        organized_context = organize_rag_results_by_source(rag_results)
        
        # Create the user message with the template format
        user_message = f"""**Conversation History**: {conversation_history_text}

**Rephrased User Query**: {rephrased_query}

**Retrieved Context**:

- **Guideline**: {organized_context['guideline_context']}

- **Alphabetic Index**: {organized_context['index_context']}

- **Tabular List**: {organized_context['tabular_context']}"""

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
        else:
            # Rephrase the user input using the new preprocessing prompt
            rephrased_query = structure_user_input_with_context(request.question, conversation_context, conversation_history)
            logger.info(f"Rephrased query: {rephrased_query}")
            
            # Search using the rephrased query
            enhanced_query = enhance_query_for_retrieval(rephrased_query)
            
            # Search using the enhanced rephrased query with config limit
            rag_results = search_single_collection_with_filtering(enhanced_query)  # Uses RAG_CONFIG["MAX_RESULTS"]
            
            logger.info(f"Retrieved {len(rag_results)} high-quality results from RAG sources")
            
            # If we don't have enough high-quality results, try with original query
            if len(rag_results) < 2:
                logger.info("Trying with original rephrased query...")
                rag_results = search_single_collection_with_filtering(rephrased_query, limit=3)
            
            # Generate response with the new RAG processing prompt
            answer = generate_rag_response_with_context(
                request.question, 
                rephrased_query, 
                rag_results, 
                conversation_history, 
                conversation_context
            )
        
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
