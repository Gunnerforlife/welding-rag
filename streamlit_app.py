import streamlit as st
import json
import os
from welding_api import WeldingRecommendationAPI

# Load secrets and set environment variables
def load_secrets():
    """Load secrets from Streamlit secrets and set as environment variables"""
    try:
        # Get OpenAI API key from secrets
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
            
        # Optional: Load other secrets if available
        if hasattr(st, 'secrets'):
            if 'ASME_PDF_PATH' in st.secrets:
                os.environ['ASME_PDF_PATH'] = st.secrets['ASME_PDF_PATH']
            if 'WPS_JSON_PATH' in st.secrets:
                os.environ['WPS_JSON_PATH'] = st.secrets['WPS_JSON_PATH']
            if 'AUTH_USERNAME' in st.secrets:
                os.environ['AUTH_USERNAME'] = st.secrets['AUTH_USERNAME']
            if 'AUTH_PASSWORD' in st.secrets:
                os.environ['AUTH_PASSWORD'] = st.secrets['AUTH_PASSWORD']
                
    except Exception as e:
        st.warning(f"Could not load secrets: {e}")

# Load secrets at startup
load_secrets()

# Initialize the API
@st.cache_resource
def init_welding_api():
    """Initialize the welding recommendation API"""
    return WeldingRecommendationAPI()

# Authentication function
def authenticate_user():
    """Simple authentication system using secrets"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Welding RAG Authentication")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Please enter your credentials")
            
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            if st.button("Login", type="primary", use_container_width=True):
                # Get credentials from secrets
                try:
                    valid_username = st.secrets.get('AUTH_USERNAME', 'admin')
                    valid_password = st.secrets.get('AUTH_PASSWORD', 'welding2024')
                except (AttributeError, FileNotFoundError):
                    # Fallback to environment variables or defaults
                    valid_username = os.environ.get('AUTH_USERNAME', 'admin')
                    valid_password = os.environ.get('AUTH_PASSWORD', 'welding2024')
                
                if username == valid_username and password == valid_password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
                    st.info("Check your .streamlit/secrets.toml for valid credentials")
        
        return False
    
    return True

def main():
    """Main application function"""
    
    # Check authentication
    if not authenticate_user():
        return
    
    # Page configuration
    st.set_page_config(
        page_title="Welding RAG Assistant",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize API
    try:
        api = init_welding_api()
    except Exception as e:
        st.error(f"Failed to initialize Welding API: {str(e)}")
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Welding RAG Assistant</h1>
        <p>AI-Powered Welding Procedure Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"👤 Welcome, {st.session_state.username}!")
        
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 🔧 System Status")
        
        # Check if RAG system is available
        pdf_available = os.path.exists("data/ASME BPVC 2023 Section IX_removed.pdf")
        cache_available = os.path.exists("cache/asme_rag_index/material_spec.json")
        
        # Check for OpenAI API key in secrets
        openai_key_available = False
        try:
            openai_key_available = bool(st.secrets.get('OPENAI_API_KEY') and 
                                       st.secrets.get('OPENAI_API_KEY') != 'your_openai_api_key_here')
        except (AttributeError, FileNotFoundError):
            # Fallback to environment variable
            openai_key_available = bool(os.environ.get('OPENAI_API_KEY'))
        
        # RAG System Status
        st.markdown("**RAG System:**")
        if pdf_available:
            st.success("✅ ASME PDF Available")
        elif cache_available:
            st.warning("⚠️ Using Cached RAG Data")
        else:
            st.error("❌ No RAG Data Available")
        
        # OpenAI API Status
        st.markdown("**Search Capability:**")
        if openai_key_available:
            st.success("✅ Vector Search (OpenAI)")
        else:
            st.warning("⚠️ Text Search Only (No OpenAI Key)")
        
        # WPS Database Status
        st.markdown("**WPS Database:**")
        if os.path.exists("welding_procedures.json"):
            st.success("✅ WPS Database Loaded")
        else:
            st.error("❌ WPS Database Missing")
        
        # Show recommendations for missing components
        if not openai_key_available:
            st.info("💡 Add OPENAI_API_KEY to .streamlit/secrets.toml for better RAG performance")
        
        if not pdf_available and not cache_available:
            st.warning("⚠️ No ASME material data available. RAG lookups will fail.")
        
        st.markdown("---")
        
        st.markdown("### 📚 About")
        st.markdown("""
        This application uses **RAG (Retrieval-Augmented Generation)** to provide 
        AI-powered welding procedure recommendations based on:
        
        - **ASME BPVC Section IX** standards
        - **P/G number compatibility** analysis
        - **Material specification** matching
        - **Thickness optimization**
        """)
        
        st.markdown("### 🎯 Features")
        st.markdown("""
        - 🔍 **RAG-powered material lookup**
        - 📊 **ASME P/G number matching**
        - 🔧 **Compatibility analysis**
        - 📏 **Wall thickness validation (mm)**
        - 🔥 **PWHT requirement filtering**
        - ⚡ **Process recommendations**
        - 📋 **Detailed WPS matching**
        """)
        
        st.markdown("### 💡 How It Works")
        st.markdown("""
        1. **Material Analysis**: RAG system searches ASME standards for both materials
        2. **P/G Number Extraction**: Identifies P/G numbers for from/to materials
        3. **Compatibility Check**: Validates P/G number compatibility (direct or reverse match)
        4. **WPS Matching**: Finds procedures matching both material P/G numbers
        5. **Thickness Check**: Validates wall thickness against WPS qualified ranges (mm)
        6. **Recommendation**: Provides ranked recommendations with process details
        """)
        
        st.markdown("### 🔍 Search Methods")
        st.markdown("""
        - **RAG + P/G**: Primary method using ASME standards
        - **Fallback Matching**: Text-based material matching
        - **Cached Lookup**: Uses pre-processed data
        """)
        
        # Add setup instructions
        st.markdown("### ⚙️ Setup Instructions")
        st.markdown("""
        **For Local Development:**
        1. Add your OpenAI API key to `.streamlit/secrets.toml`
        2. File should contain: `OPENAI_API_KEY = "your_key"`
        
        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Add secrets in the "Secrets" section
        3. Use the same TOML format
        """)
        
        # Add example inputs
        st.markdown("### 📝 Example Inputs")
        st.markdown("""
        **Material Specifications:**
        - `A36` (Carbon Steel)
        - `SA-516 Grade 70` (Pressure Vessel Steel)
        - `304 Stainless Steel` (Austenitic SS)
        - `API 5L X52` (Pipeline Steel)
        - `SA 106 GR. B` (Seamless Carbon Steel Pipe)
        
        **Common Combinations:**
        - A36 to A36 (similar materials)
        - SA-516 GR.70 to SA-516 GR.70 (pressure vessels)
        - 304 SS to 316L SS (stainless combinations)
        """)
        
        st.markdown("---")
        st.markdown("*RAG-powered welding recommendations*")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Input Parameters")
        
        # Input form
        with st.form("welding_params"):
            st.markdown("**Enter material specifications as they appear in ASME standards:**")
            
            from_material = st.text_input(
                "From Material",
                placeholder="e.g., A36, SA-516 Grade 70, 304 Stainless Steel",
                help="Enter the first material specification (ASME format preferred for better RAG matching)"
            )
            
            to_material = st.text_input(
                "To Material",
                placeholder="e.g., A36, SA-516 Grade 70, 304 Stainless Steel",
                help="Enter the second material specification (ASME format preferred for better RAG matching)"
            )
            
            thickness = st.number_input(
                "Wall Thickness (mm)",
                min_value=0.1,
                max_value=200.0,
                value=12.7,
                step=0.1,
                help="Enter material wall thickness in millimeters"
            )
            
            col_proc, col_pwht = st.columns(2)
            
            with col_proc:
                preferred_process = st.selectbox(
                    "Preferred Process (Optional)",
                    ["Any", "SMAW", "GMAW", "GTAW", "FCAW", "SAW"],
                    help="Select preferred welding process or leave as 'Any'"
                )
            
            with col_pwht:
                pwht_required = st.selectbox(
                    "PWHT Required",
                    ["Any", "Yes", "No"],
                    help="Post Weld Heat Treatment requirement"
                )
            
            # Add advanced options
            with st.expander("⚙️ Advanced Options"):
                enable_rag = st.checkbox("Enable RAG Analysis", value=True, help="Use RAG system for P/G number lookup")
                max_wps = st.slider("Max WPS to Display", min_value=1, max_value=10, value=3, help="Maximum number of WPS to show")
            
            st.info("💡 Note: This system matches WPS based on material-to-material welding (not base+filler). Input wall thickness in mm matches WPS qualified thickness ranges. Joint type filtering is not available as the WPS database doesn't contain joint type information. PWHT (Post Weld Heat Treatment) filtering is available.")
            
            submitted = st.form_submit_button("🔍 Get Recommendations", type="primary")
    
    with col2:
        st.markdown("### 📊 Recommendations")
        
        if submitted:
            if not from_material or not to_material:
                st.error("Please enter both from material and to material.")
            else:
                # Prepare input data
                input_data = {
                    "from_material": from_material,
                    "to_material": to_material,
                    "thickness": thickness,
                    "joint_type": "Any", # Joint type is not used for WPS filtering
                    "pwht_required": pwht_required,
                    "enable_rag": enable_rag,
                    "max_wps": max_wps
                }
                
                if preferred_process != "Any":
                    input_data["preferred_process"] = preferred_process
                
                # Get recommendations
                with st.spinner("Analyzing materials and generating recommendations..."):
                    try:
                        recommendations = api.get_recommendations(input_data)
                        
                        # Show RAG analysis first
                        st.markdown("### 🔍 RAG Analysis Results")
                        search_details = recommendations.get("search_details", {})
                        
                        if search_details.get("error"):
                            st.error(f"RAG Analysis Error: {search_details['error']}")
                        else:
                            # Show search method used
                            matching_method = search_details.get("matching_method", "Unknown")
                            if matching_method == "RAG + P/G":
                                st.success("✅ Using ASME RAG + P/G Number matching")
                            else:
                                st.warning("⚠️ Using fallback material name matching")
                            
                            # Show RAG details for both materials
                            col_rag1, col_rag2 = st.columns(2)
                            
                            with col_rag1:
                                st.markdown("**From Material RAG:**")
                                from_rag = search_details.get("from_material_rag")
                                if from_rag:
                                    st.write(f"- P-No: {from_rag.get('p_no', 'Not found')}")
                                    st.write(f"- G-No: {from_rag.get('g_no', 'Not found')}")
                                    st.write(f"- Spec: {from_rag.get('spec_no', 'Not found')}")
                                    st.write(f"- Grade: {from_rag.get('grade', 'Not found')}")
                                    st.write(f"- Method: {from_rag.get('search_method', 'Unknown')}")
                                else:
                                    st.write("❌ No RAG data found")
                            
                            with col_rag2:
                                st.markdown("**To Material RAG:**")
                                to_rag = search_details.get("to_material_rag")
                                if to_rag:
                                    st.write(f"- P-No: {to_rag.get('p_no', 'Not found')}")
                                    st.write(f"- G-No: {to_rag.get('g_no', 'Not found')}")
                                    st.write(f"- Spec: {to_rag.get('spec_no', 'Not found')}")
                                    st.write(f"- Grade: {to_rag.get('grade', 'Not found')}")
                                    st.write(f"- Method: {to_rag.get('search_method', 'Unknown')}")
                                else:
                                    st.write("❌ No RAG data found")
                        
                        # Show ASME compatibility
                        st.markdown("### 🔧 ASME Compatibility")
                        compatibility = recommendations.get("asme_compatibility", {})
                        
                        if compatibility.get("error"):
                            st.error(f"Compatibility Error: {compatibility['error']}")
                        else:
                            if compatibility.get("compatible"):
                                st.success("✅ Materials are ASME compatible!")
                            elif compatibility.get("has_rag_data"):
                                st.error("❌ Materials are NOT compatible according to ASME P/G numbers")
                            else:
                                st.warning("⚠️ ASME compatibility could not be determined (no RAG data)")
                            
                            # Show detailed compatibility info
                            with st.expander("📋 Detailed Compatibility Info"):
                                col_comp1, col_comp2 = st.columns(2)
                                
                                with col_comp1:
                                    st.markdown("**Base Material:**")
                                    st.write(f"- P-Number: {compatibility.get('base_p_number', 'Not found')}")
                                    st.write(f"- G-Number: {compatibility.get('base_g_number', 'Not found')}")
                                    st.write(f"- Spec: {compatibility.get('base_spec', 'Not found')}")
                                    st.write(f"- Grade: {compatibility.get('base_grade', 'Not found')}")
                                    st.write(f"- Search Method: {compatibility.get('base_search_method', 'Unknown')}")
                                
                                with col_comp2:
                                    st.markdown("**Filler Material:**")
                                    st.write(f"- P-Number: {compatibility.get('filler_p_number', 'Not found')}")
                                    st.write(f"- G-Number: {compatibility.get('filler_g_number', 'Not found')}")
                                    st.write(f"- Spec: {compatibility.get('filler_spec', 'Not found')}")
                                    st.write(f"- Grade: {compatibility.get('filler_grade', 'Not found')}")
                                    st.write(f"- Search Method: {compatibility.get('filler_search_method', 'Unknown')}")
                        
                        # Show WPS recommendations
                        st.markdown("### 📊 WPS Recommendations")
                        
                        if recommendations.get("error"):
                            st.error(f"Recommendation Error: {recommendations['error']}")
                        elif recommendations.get("compatible_wps"):
                            st.success(f"Found {len(recommendations['compatible_wps'])} compatible procedures!")
                            
                            # Display summary metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{len(recommendations['compatible_wps'])}</h3>
                                    <p>Compatible WPS</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_b:
                                total_matches = recommendations.get("total_matches", 0)
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{total_matches}</h3>
                                    <p>Total Matches</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_c:
                                status = "✅" if compatibility.get("compatible", False) else "⚠️"
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{status}</h3>
                                    <p>ASME Compatible</p>
                                </div>
                                """, unsafe_allow_html=True)
                            

                            
                            # Show compatible WPS
                            st.markdown("**Compatible WPS Details:**")
                            max_display = input_data.get("max_wps", 3)
                            for i, wps in enumerate(recommendations["compatible_wps"][:max_display]):
                                with st.expander(f"📄 WPS {i+1}: {wps.get('wps_no', 'N/A')}", expanded=i==0):
                                    col_p, col_q = st.columns(2)
                                    
                                    with col_p:
                                        st.write("**Specification:**")
                                        st.write(f"- WPS Number: {wps.get('wps_no', 'N/A')}")
                                        st.write(f"- PQR Number: {wps.get('pqr_no', 'N/A')}")
                                        st.write(f"- Reference Code: {wps.get('reference_code', 'N/A')}")
                                        st.write(f"- Base Material: {wps.get('material_from', 'N/A')}")
                                        st.write(f"- Filler Material: {wps.get('material_to', 'N/A')}")
                                        st.write(f"- Electrode: {wps.get('electrode', 'N/A')}")
                                        
                                        # Show welding process for this WPS
                                        welding_processes = wps.get('welding_process', [])
                                        if welding_processes:
                                            process_str = ", ".join(welding_processes)
                                            st.write(f"- Welding Process: {process_str}")
                                        else:
                                            st.write("- Welding Process: Not determined")
                                    
                                    with col_q:
                                        st.write("**Parameters:**")
                                        st.write(f"- Qualified Thickness: {wps.get('qualified_thick', 'N/A')}")
                                        st.write(f"- Qualified Diameter: {wps.get('qualified_dia', 'N/A')}")
                                        st.write(f"- P-Numbers: {wps.get('p_from', 'N/A')} → {wps.get('p_to', 'N/A')}")
                                        st.write(f"- G-Numbers: {wps.get('g_from', 'N/A')} → {wps.get('g_to', 'N/A')}")
                                        
                                        pwht_status = "Yes" if wps.get('pwht', False) else "No"
                                        st.write(f"- PWHT Required: {pwht_status}")
                                        
                                        st.write(f"- Welder: {wps.get('welder_name', 'N/A')} ({wps.get('welder_no', 'N/A')})")
                        else:
                            st.warning("No compatible welding procedures found for the specified materials.")
                            
                            # Show what was attempted
                            st.markdown("### 🔍 Search Details")
                            st.write("**Search Parameters:**")
                            st.json(input_data)
                            
                            if search_details:
                                st.write("**RAG Search Results:**")
                                st.json(search_details)
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        st.write("Please check your input parameters and try again.")
                        
                        # Show debug info
                        with st.expander("🐛 Debug Information"):
                            st.write("**Input Data:**")
                            st.json(input_data)
                            st.write("**Error Details:**")
                            st.text(str(e))
        else:
            st.info("👆 Enter your welding parameters above to get AI-powered recommendations!")

if __name__ == "__main__":
    main()
