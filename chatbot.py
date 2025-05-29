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
    """Determine if the question is related to medical coding"""
    medical_keywords = [
        'icd', 'code', 'diagnosis', 'medical', 'condition', 'disease', 'symptom',
        'guideline', 'documentation', 'requirement', 'coding', 'clinical', 'health',
        'patient', 'treatment', 'procedure', 'assessment', 'record', 'chart'
    ]
    
    question_lower = question.lower()
    # Check for common greetings or general chat
    general_chat_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']
    
    if any(pattern in question_lower for pattern in general_chat_patterns):
        return False
    
    return any(keyword in question_lower for keyword in medical_keywords)

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

7. Output only the rephrased query as a single sentence, avoiding explanations or additional text."""

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

def search_single_collection_with_filtering(rephrased_query: str, limit: int = 9) -> List[Dict[str, Any]]:
    """Search single collection with filtering to avoid duplicates"""
    try:
        # Generate embedding for the rephrased query
        query_embedding = get_gemini_embedding(rephrased_query)
        
        # Search single collection with higher limit
        search_result = qdrant_client.search(
            collection_name="Medical_Coder",  # Your actual collection name
            query_vector=query_embedding,
            limit=limit * 3  # Get more results for deduplication
        )
        
        # Deduplicate results
        seen_texts = set()
        unique_results = []
        
        for result in search_result:
            text_content = result.payload.get("text", "")
            
            # Create hash for deduplication
            import hashlib
            text_hash = hashlib.md5(text_content.encode()).hexdigest()
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                
                # Determine source group based on file name
                file_name = result.payload.get("metadata", {}).get("file_name", "")
                source_group = "Unknown"
                source_description = "Unknown"
                
                if file_name in ["RAG1.pdf", "RAG1_1.xlsx"]:
                    source_group = "Group 1"
                    source_description = "ICD-10 Guidelines"
                elif file_name in ["RAG2.xlsx", "RAG2_1.pdf", "RAG2_2.pdf", "RAG2_3.pdf"]:
                    source_group = "Group 2" 
                    source_description = "ICD-10 Index"
                elif file_name in ["RAG3.csv"]:
                    source_group = "Group 3"
                    source_description = "ICD-10 Tabular List"
                
                unique_results.append({
                    "text": text_content,
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score,
                    "source_group": source_group,
                    "source_priority": 1,
                    "source_description": source_description
                })
                
                if len(unique_results) >= limit:
                    break
        
        return unique_results
        
    except Exception as e:
        logger.error(f"Error in single collection search: {e}")
        return []

def organize_rag_results_by_source(rag_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """Organize RAG results by source type for the new prompt format"""
    organized = {
        "guideline_context": "",
        "index_context": "",
        "tabular_context": ""
    }
    
    for result in rag_results:
        source_group = result.get("source_group", "")
        text = result.get("text", "")
        
        if source_group == "Group 1":  # Guidelines
            organized["guideline_context"] += f"{text}\n\n"
        elif source_group == "Group 2":  # Index
            organized["index_context"] += f"{text}\n\n"
        elif source_group == "Group 3":  # Tabular List
            organized["tabular_context"] += f"{text}\n\n"
    
    # Clean up trailing newlines
    for key in organized:
        organized[key] = organized[key].strip()
        if not organized[key]:
            organized[key] = "No relevant information found."
    
    return organized

def generate_rag_response_with_context(user_question: str, rephrased_query: str, rag_results: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None, conversation_context: Dict[str, Any] = None) -> str:
    """Generate response using the new RAG processing prompt"""
    try:
        # Use the new RAG processing prompt from the document
        system_prompt = """You are an expert medical coding assistant specializing in ICD-10-CM. Your task is to generate a response to a user's query based on the provided ICD-10 dataset context (Guideline, Alphabetic Index, and Tabular List) and conversation history, strictly adhering to ICD-10-CM guidelines and formatting the response as specified below. Follow these guidelines:

1. **Use Retrieved Context**: Base the response primarily on the retrieved documents from the Guideline, Alphabetic Index, and Tabular List, ensuring accuracy and relevance.

2. **Incorporate Conversation History**: Reference prior interactions to maintain continuity in multi-turn conversations, addressing follow-up questions appropriately.

3. **Focus on Tabular List Details**: When processing the Tabular List, explicitly consider and reference:
   - **Include Notes**: Conditions covered by the code (e.g., **J45** includes allergic asthma).
   - **Exclude 1 Notes**: Conditions not coded here (e.g., **J12.9** excludes SARS pneumonia, **U07.1**).
   - **Exclude 2 Notes**: Conditions that can be coded separately if present (e.g., **J44.9** excludes **J60-J70**, but both can apply).
   - **Code Also**: Additional codes needed for full description (e.g., **M32.14** may require **N03.-**).
   - **Code First**: Underlying condition to code before the manifestation (e.g., **E11.22** requires **I12.-** first).
   - **Use Additional Code**: Codes for cause or severity (e.g., **E11.621** needs **L97.4-** for ulcer site).
   - **Laterality**: Left, right, or bilateral specifications (e.g., **H26.11-** for left eye cataract).
   - **Gender Specificity**: Male or female-specific codes (e.g., **C61** for prostate cancer).
   - **Age Specificity**: Pediatric or geriatric codes (e.g., **P07.1-** for short gestation).

4. **Apply Combination Code Guidelines**: If two or more diagnoses are related by "due to," "with," or "associated with" in the query or medical context, check for a combination code in ICD-10-CM. If present, assign only the combination code.

5. **Respect Hierarchy and Laterality**: Choose the most specific code available, including laterality, severity, or type (acute/chronic) when applicable. Only default to an unspecified code if the query lacks the necessary detail, and then prompt for clarification.

6. **Handle Query Types**:
   - For code lookup queries (e.g., "What is the ICD-10 code for diabetes?"), provide the exact code(s) and a brief description.
   - For guideline queries (e.g., "How do I code a fractured arm?"), include relevant coding rules or instructions from the Guideline.
   - For general medical inquiries (e.g., "What does E11.9 mean?"), explain the code or concept using the Tabular List and Guideline.

7. **Handle Missing Specificity/Context**: If the query or context lacks specificity (e.g., laterality, type, gender, or age), use the unspecified or default ICD-10-CM code (if applicable) as per the Alphabetic Index or Tabular List, and include a single clarifying question in the "Clarification (if needed)" section to prompt for specific details.

8. **Response Format**:
   - **Answer**: Provide a concise response with ICD-10 code(s) highlighted (e.g., **E11.9**) or relevant information.
   - **Rationale**: Explain the response, referencing the Guideline, Include/Exclude notes, Code also/Code first/Use Additional Code instructions, and any relevant laterality, gender, or age specificity from the Tabular List.
   - **Clarification (if needed)**: Include a single question if clarification is needed for specificity or missing context; otherwise omit this section.
   - **Disclaimer**: Always include: "This answer is for informational purposes only. Please confirm with the latest ICD-10-CM guidelines or a certified medical coder."

9. **Adhere to ICD-10-CM Guidelines**: Follow official coding conventions, including sequencing rules and specificity requirements, as outlined in the Guideline dataset.

10. **Avoid Non-ICD-10 Content**: Do not include unrelated information (e.g., general health advice or CPT) unless supported by the datasets."""

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
            rag_results = search_single_collection_with_filtering(rephrased_query, limit=9)
            
            logger.info(f"Retrieved {len(rag_results)} unique results from RAG sources")
            
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
