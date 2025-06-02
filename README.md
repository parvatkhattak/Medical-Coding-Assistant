# 🏥 Medical Coding Assistant

An AI-powered medical coding chatbot that provides intelligent guidance on ICD-10 and CPT coding for medical professionals. Built with FastAPI backend and Streamlit frontend, utilizing RAG (Retrieval-Augmented Generation) architecture with Google's Gemini AI.

## ✨ Features

- **🤖 Intelligent Medical Coding Assistance**: Get instant answers to ICD-10 and CPT coding questions
- **📚 RAG-Powered Knowledge Base**: Retrieves relevant information from comprehensive medical coding datasets
- **💬 Conversational Interface**: Maintains context across multiple questions in a chat session
- **🔍 Advanced Query Processing**: Automatically rephrases and optimizes user queries for better results
- **📊 Debug Mode**: Detailed insights into query processing and source retrieval
- **📱 Responsive Design**: Modern, mobile-friendly interface with dark theme
- **🔄 Real-time Processing**: Fast response times with conversation history tracking

## 🏗️ Architecture

### Backend (FastAPI)
- **RAG Pipeline**: Query preprocessing, vector search, and response generation
- **Vector Database**: Qdrant for semantic search across medical coding documents
- **AI Integration**: Google Gemini 2.0 Flash for embeddings and text generation
- **Conversation Management**: Supabase for chat history storage
- **Multi-source Knowledge**: ICD-10 Guidelines, Alphabetic Index, and Tabular List

### Frontend (Streamlit)
- **Modern UI**: Clean, professional interface with dark theme
- **Real-time Chat**: Interactive chat interface with message history
- **Debug Tools**: Advanced debugging capabilities for development
- **Responsive Design**: Mobile-optimized with touch-friendly controls
- **Quick Actions**: Sample questions for easy getting started

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Qdrant vector database
- Google AI API key
- Supabase account (for chat history)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/parvatkhattak/Medical_chatbot.git
cd Medical_chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Google AI Configuration
GEMINI_API_KEY=your_gemini_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_TABLE_NAME=chathistory
```

4. **Start the backend server**
```bash
python chatbot.py
```

5. **Launch the frontend (in a new terminal)**
```bash
streamlit run streamlit_app.py
```

6. **Access the application**
Open your browser and navigate to `http://localhost:8501`


## 🗄️ Data Setup

The system expects three main document groups in your Qdrant collection:

### Group 1: ICD-10 Guidelines
- `RAG1.pdf` - Official ICD-10 coding guidelines
- `RAG1_1.xlsx` - Supplementary guideline data

### Group 2: ICD-10 Alphabetic Index
- `RAG2.xlsx` - Main alphabetic index
- `RAG2_1.pdf`, `RAG2_2.pdf`, `RAG2_3.pdf` - Supporting index documents

### Group 3: ICD-10 Tabular List
- `RAG3.csv` - Complete tabular list with codes and descriptions

## 🔧 Configuration

### Backend Configuration
The system uses several configurable parameters:

- **Collection Name**: `Medical_Coder` (default Qdrant collection)
- **Embedding Model**: `text-embedding-004` (Google)
- **Generation Model**: `gemini-2.0-flash-exp`
- **Search Limit**: 9 results per query (configurable)
- **Temperature**: 0.3 for medical queries, 0.7 for general chat

### Frontend Configuration
- **Server URL**: `http://localhost:8000` (FastAPI backend)
- **Debug Mode**: Toggle for development insights
- **Chat History**: Automatic session management

## 🎯 Usage Examples

### Basic Coding Questions
```
User: "What is the ICD-10 code for diabetes?"
Assistant: The ICD-10 code for diabetes mellitus is **E11.9** (Type 2 diabetes mellitus without complications)...
```

### Complex Coding Scenarios
```
User: "How do I code pneumonia with sepsis?"
Assistant: For pneumonia with sepsis, you'll need to consider combination coding rules...
```

### Follow-up Questions
```
User: "What about the same patient with kidney complications?"
Assistant: Building on the previous case, for diabetes with kidney complications...
```

## 🔍 Debug Mode

Enable debug mode to see:
- **Query Preprocessing**: How user input is optimized
- **Source Retrieval**: Which documents were searched
- **Response Generation**: AI reasoning process
- **Conversation Context**: How chat history influences responses

## 🏥 Medical Coding Features

### ICD-10 Specific Capabilities
- **Code Lookup**: Find specific codes for conditions
- **Guideline Interpretation**: Understand coding rules and conventions
- **Sequencing Rules**: Proper order for multiple diagnoses
- **Combination Codes**: When to use single vs. multiple codes
- **Specificity Requirements**: Most detailed code available

### Advanced Features
- **Include/Exclude Notes**: Comprehensive code usage rules
- **Laterality Coding**: Left, right, bilateral specifications
- **Gender/Age Specificity**: Demographic-specific codes
- **Code Also/Code First**: Proper sequencing instructions

## 🛠️ Development

### Project Structure
```
Medical_chatbot/
├── chatbot.py              # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── nltk_data/             # NLTK data directory
└── README.md              # This file
```

### API Endpoints
- `POST /api/chat` - Main chat endpoint
- `POST /api/new-chat` - Create new chat session
- `GET /api/chat-history/{chat_id}` - Retrieve chat history
- `GET /api/health` - Health check

### Key Functions
- **Query Preprocessing**: `structure_user_input_with_context()`
- **Vector Search**: `search_single_collection_with_filtering()`
- **Response Generation**: `generate_rag_response_with_context()`
- **Conversation Management**: `get_conversation_history()`

## 🔒 Security & Privacy

- **Data Privacy**: No sensitive medical data is stored permanently
- **API Security**: Environment variables for sensitive keys
- **Session Management**: Secure chat session handling
- **Rate Limiting**: Prevents API abuse

## 📊 Performance

- **Response Time**: Typically 2-5 seconds for complex queries
- **Accuracy**: High precision for standard ICD-10 codes
- **Scalability**: Supports multiple concurrent users
- **Memory Usage**: Optimized for production deployment

## 🚨 Important Disclaimers

⚠️ **For Educational and Informational Purposes Only**

This chatbot is designed to assist with medical coding education and provide general guidance. It should NOT be used as the sole source for clinical coding decisions. Always:

- Consult official ICD-10-CM guidelines
- Verify codes with certified medical coders
- Follow your organization's coding policies
- Review codes with healthcare providers when needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Backend not starting?**
- Check your `.env` file configuration
- Ensure Qdrant and Supabase are accessible
- Verify API keys are valid

**Frontend connection errors?**
- Confirm FastAPI server is running on port 8000
- Check firewall settings
- Verify CORS configuration

**Empty responses?**
- Check if Qdrant collection has data
- Verify embedding model is working
- Enable debug mode for detailed insights

### Getting Help

- 📧 Email: your-parvatkhattak03@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/parvatkhattak/Medical_chatbot/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/parvatkhattak/Medical_chatbot/discussions)

## 🙏 Acknowledgments

- **Google AI**: For Gemini API and embedding models
- **Qdrant**: For vector database capabilities
- **Streamlit**: For the excellent web framework
- **FastAPI**: For the robust backend framework
- **Medical Coding Community**: For domain expertise and feedback

## 📈 Roadmap

- [ ] Add CPT coding support
- [ ] Implement user authentication
- [ ] Add batch query processing
- [ ] Mobile app development
- [ ] Integration with EHR systems
- [ ] Multi-language support

---

**Built with ❤️ for the medical coding community**
