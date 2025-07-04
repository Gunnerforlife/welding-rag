# Welding RAG System

A comprehensive welding procedure recommendation system that integrates ASME material standards with welding procedure specifications (WPS) using RAG (Retrieval-Augmented Generation) technology.

## Features

- **ASME Material Analysis**: Uses RAG to search ASME standards for material P/G numbers
- **WPS Matching**: Finds compatible welding procedures based on material specifications
- **Compatibility Checking**: Validates material compatibility using ASME standards
- **Intelligent Fallback**: Text-based matching when RAG data is unavailable
- **Web Interface**: User-friendly Streamlit interface

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   # OpenAI API Configuration (Required for RAG vector search)
   OPENAI_API_KEY = "your_openai_api_key_here"
   
   # Optional: Custom file paths
   # ASME_PDF_PATH = "data/ASME BPVC 2023 Section IX_removed.pdf"
   # WPS_JSON_PATH = "welding_procedures.json"
   ```

3. **Run the App**
   ```bash
   streamlit run streamlit_app.py
   ```

### Streamlit Cloud Deployment

1. **Upload Project** to GitHub
2. **Deploy on Streamlit Cloud**
3. **Add Secrets** in App Settings:
   - Go to your app settings
   - Navigate to "Secrets" section
   - Add your secrets in TOML format:
     ```toml
     OPENAI_API_KEY = "your_openai_api_key_here"
     ```

## System Components

### Core Files

- `streamlit_app.py` - Main web interface
- `welding_api.py` - API for welding recommendations
- `rag_core.py` - RAG system for ASME material lookup
- `welding_procedures.json` - Database of welding procedures

### Configuration

- `.streamlit/secrets.toml` - Secrets configuration (not committed)
- `requirements.txt` - Python dependencies

## Usage

### Basic Material Lookup

1. Enter base material (e.g., "A36", "SA-516 Grade 70")
2. Enter filler material (e.g., "E7018", "ER70S-2")
3. Specify thickness range
4. Select PWHT requirements
5. Get recommendations with ASME compatibility analysis

### Advanced Options

- **Enable/Disable RAG**: Toggle between RAG and text-based search
- **Max Results**: Control number of WPS recommendations
- **System Status**: Monitor RAG and OpenAI availability

## System Requirements

- Python 3.8+
- OpenAI API key (optional, for enhanced RAG performance)
- ASME PDF document (optional, for full RAG functionality)

## Error Handling

The system gracefully handles missing components:
- **No OpenAI Key**: Falls back to text-based search
- **No PDF**: Uses cached RAG data if available
- **No RAG Data**: Performs text-based material matching

## Security

- Secrets are stored in `.streamlit/secrets.toml` (local) or Streamlit Cloud secrets
- The secrets file is automatically excluded from version control
- No sensitive data is committed to the repository

## Support

For questions or issues:
1. Check the system status in the app
2. Verify your secrets configuration
3. Ensure all required files are present 