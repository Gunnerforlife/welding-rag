"""
Enhanced JSON Welding Recommendation API with ASME RAG Integration

This module provides welding procedure recommendations by:
1. Loading welding procedures from JSON
2. Querying P/G numbers from ASME RAG
3. Normalizing and checking P/G compatibility
4. Fallback matching by material names and thickness
5. Extracting welding processes from electrodes
6. Finding matching WPS with limits
"""

import json
import os
import logging
from typing import Dict, List, Optional, Union, Any
import re
from rag_core import ASMEMaterialRAG

# Load secrets from Streamlit if available, otherwise use environment variables
def get_secret(key: str, default: str = None) -> Optional[str]:
    """Get secret from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except ImportError:
        pass
    
    # Fallback to environment variables
    return os.environ.get(key, default)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG instance
_rag_instance = None

class WeldingRecommendationAPI:
    """Enhanced welding recommendation API with ASME RAG integration"""
    
    def __init__(self, json_file_path: str = None, pdf_path: str = None):
        """Initialize the API with welding procedures data"""
        # Use secrets if available, otherwise use provided or default paths
        self.json_file_path = json_file_path or get_secret('WPS_JSON_PATH', 'welding_procedures.json')
        self.pdf_path = pdf_path or get_secret('ASME_PDF_PATH', 'data/ASME BPVC 2023 Section IX_removed.pdf')
        self.welding_procedures = self._load_welding_procedures()
        
    def _load_welding_procedures(self) -> List[Dict]:
        """Load welding procedures from JSON file"""
        try:
            if os.path.exists(self.json_file_path):
                with open(self.json_file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} welding procedures from {self.json_file_path}")
                    return data
            else:
                logger.warning(f"JSON file not found: {self.json_file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading welding procedures: {e}")
            return []
    
    def _query_asme_rag_pg_numbers(self, material: str) -> Optional[Dict[str, str]]:
        """Query ASME RAG for P/G numbers"""
        global _rag_instance
        
        try:
            # Initialize RAG system only once
            if _rag_instance is None:
                # Check if PDF exists
                if not os.path.exists(self.pdf_path):
                    logger.warning(f"PDF file not found: {self.pdf_path}")
                    logger.info("RAG system will use cached data if available")
                
                # Check for OpenAI API key
                openai_api_key = get_secret('OPENAI_API_KEY')
                if not openai_api_key:
                    logger.warning("No OpenAI API key found. RAG will use text-based search only.")
                    use_vector_search = False
                else:
                    logger.info("OpenAI API key found. RAG will use vector search.")
                    use_vector_search = True
                
                _rag_instance = ASMEMaterialRAG(self.pdf_path, use_vector_search=use_vector_search)
                logger.info("Initializing ASME material lookup system...")
                _rag_instance.build_index()
                logger.info("RAG system ready!")
            
            # Query the material
            result = _rag_instance.query_material(material)
            
            if result.get("error"):
                logger.error(f"RAG query error: {result['error']}")
                return None
            
            return {
                "p_no": result.get("p_no", ""),
                "g_no": result.get("g_no", ""),
                "spec_no": result.get("spec_no", ""),
                "grade": result.get("grade", ""),
                "composition": result.get("composition", ""),
                "thickness_limits": result.get("thickness_limits", ""),
                "search_method": result.get("search_method", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error querying ASME RAG for material '{material}': {e}")
            return None

    
    def _normalize_pg_number(self, pg_str: str) -> Optional[int]:
        """Normalize P/G number string to integer"""
        if not pg_str:
            return None
        
        # Handle various formats: "P-1", "P1", "1", etc.
        normalized = str(pg_str).strip().upper()
        normalized = re.sub(r'[P\-G\-]', '', normalized)
        
        try:
            return int(normalized)
        except ValueError:
            logger.warning(f"Could not normalize P/G number: {pg_str}")
            return None
    
    def _check_pg_compatibility(self, base_pg: Optional[int], filler_pg: Optional[int]) -> bool:
        """Check if P/G numbers are compatible"""
        if base_pg is None or filler_pg is None:
            return False
        
        # Basic compatibility logic - can be enhanced based on ASME standards
        if base_pg == filler_pg:
            return True
        
        # Some common compatibility rules (simplified)
        compatible_groups = [
            {1, 2, 3},  # Low carbon steels
            {4, 5, 6},  # Low alloy steels
            {8, 9, 10}, # Stainless steels
        ]
        
        for group in compatible_groups:
            if base_pg in group and filler_pg in group:
                return True
        
        return False
    
    def _fallback_material_match(self, base_material: str, filler_material: str) -> bool:
        """Fallback matching by material name similarity"""
        base_lower = base_material.lower()
        filler_lower = filler_material.lower()
        
        # Simple keyword matching
        keywords = ['carbon', 'steel', 'stainless', 'aluminum', 'alloy', 'mild']
        
        for keyword in keywords:
            if keyword in base_lower and keyword in filler_lower:
                return True
        
        return False
    
    def _check_thickness_compatibility(self, base_thickness: float, wps_thickness_range: str) -> bool:
        """Check if base thickness falls within WPS thickness range"""
        try:
            # Handle different formats in qualified_thick field
            thickness_str = wps_thickness_range.strip()
            
            # Handle special cases
            if not thickness_str or thickness_str.upper() in ['NA', 'ALL', 'VN']:
                return True  # Accept if no restrictions
            
            # Convert input thickness from inches to mm for comparison (assuming JSON is in mm)
            base_thickness_mm = base_thickness * 25.4
            
            # Handle ">" format (e.g., ">25")
            if thickness_str.startswith('>'):
                min_thickness = float(thickness_str[1:])
                return base_thickness_mm >= min_thickness
            
            # Handle "<" format (e.g., "<50")
            if thickness_str.startswith('<'):
                max_thickness = float(thickness_str[1:])
                return base_thickness_mm <= max_thickness
            
            # Handle range format (e.g., "5-72", "1.5-11.12")
            if '-' in thickness_str:
                parts = thickness_str.split('-')
                if len(parts) == 2:
                    min_thickness = float(parts[0].strip())
                    max_thickness = float(parts[1].strip())
                    return min_thickness <= base_thickness_mm <= max_thickness
            else:
                # Single thickness value
                target_thickness = float(thickness_str)
                # Allow some tolerance (±20% or ±2mm, whichever is smaller)
                tolerance = min(target_thickness * 0.2, 2.0)
                return abs(base_thickness_mm - target_thickness) <= tolerance
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse thickness range: {wps_thickness_range} - {e}")
            return True  # Accept if we can't parse (conservative approach)
        
        return False
    
    def _extract_welding_processes(self, electrode_info: str) -> List[str]:
        """Extract welding processes from electrode information"""
        processes = []
        electrode_lower = electrode_info.lower()
        
        # Map electrode types to welding processes
        process_mapping = {
            'e60': 'SMAW',
            'e70': 'SMAW',
            'e80': 'SMAW',
            'e90': 'SMAW',
            'er70': 'GMAW',
            'er80': 'GMAW',
            'er90': 'GMAW',
            'er316': 'GTAW',
            'er308': 'GTAW',
            'flux': 'SAW',
            'mig': 'GMAW',
            'tig': 'GTAW',
            'stick': 'SMAW',
            'arc': 'SMAW'
        }
        
        for electrode_type, process in process_mapping.items():
            if electrode_type in electrode_lower:
                if process not in processes:
                    processes.append(process)
        
        return processes if processes else ['SMAW']  # Default to SMAW
    
    def _find_matching_wps(self, base_material: str, filler_material: str, 
                          base_thickness: float, joint_type: str = "Butt", pwht_required: str = "Any") -> List[Dict]:
        """Find matching WPS based on materials, thickness, and PWHT requirements"""
        matching_wps = []
        
        # First try ASME RAG for P/G numbers
        base_pg_info = self._query_asme_rag_pg_numbers(base_material)
        filler_pg_info = self._query_asme_rag_pg_numbers(filler_material)
        
        for wps in self.welding_procedures:
            try:
                # Note: Joint type filtering removed since welding_procedures.json doesn't contain joint_type field
                # All WPS are considered regardless of joint type
                
                # Check PWHT compatibility
                if pwht_required != "Any":
                    wps_pwht = wps.get('pwht', False)
                    required_pwht = pwht_required == "Yes"
                    if wps_pwht != required_pwht:
                        continue
                
                # Material compatibility check
                material_compatible = False
                
                if base_pg_info and filler_pg_info:
                    # Use P/G number matching
                    base_pg = self._normalize_pg_number(base_pg_info.get('p_no'))
                    filler_pg = self._normalize_pg_number(filler_pg_info.get('p_no'))
                    material_compatible = self._check_pg_compatibility(base_pg, filler_pg)
                else:
                    # Fallback to material name matching
                    wps_base_material = wps.get('material_from', '')  # Updated field name
                    wps_filler_material = wps.get('material_to', '')    # Updated field name
                    
                    material_compatible = (
                        self._fallback_material_match(base_material, wps_base_material) and
                        self._fallback_material_match(filler_material, wps_filler_material)
                    )
                
                if not material_compatible:
                    continue
                
                # Check thickness compatibility using qualified_thick field
                wps_thickness = wps.get('qualified_thick', '')
                if wps_thickness and not self._check_thickness_compatibility(base_thickness, wps_thickness):
                    continue
                
                # If we get here, the WPS is compatible
                matching_wps.append(wps)
                
            except Exception as e:
                logger.warning(f"Error processing WPS {wps.get('wps_no', 'Unknown')}: {e}")
                continue
        
        return matching_wps
    
    def get_welding_recommendations(self, base_material: str, filler_material: str,
                                  base_thickness: float, joint_type: str = "Butt", pwht_required: str = "Any") -> str:
        """Get welding recommendations as JSON string"""
        try:
            matching_wps = self._find_matching_wps(base_material, filler_material, base_thickness, joint_type, pwht_required)
            
            # Limit to 5 WPS
            limited_wps = matching_wps[:5]
            
            # Extract recommended processes
            recommended_processes = set()
            for wps in limited_wps:
                electrode_info = wps.get('electrode', '')
                processes = self._extract_welding_processes(electrode_info)
                recommended_processes.update(processes)
            
            result = {
                "status": "success",
                "input_parameters": {
                    "base_material": base_material,
                    "filler_material": filler_material,
                    "base_thickness": base_thickness,
                    "joint_type": joint_type,
                    "pwht_required": pwht_required
                },
                "total_matches": len(matching_wps),
                "returned_wps_count": len(limited_wps),
                "recommended_processes": list(recommended_processes),
                "welding_procedures": limited_wps
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error getting welding recommendations: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e),
                "welding_procedures": []
            }, indent=2)
    
    def get_welding_recommendations_dict(self, base_material: str, filler_material: str,
                                       base_thickness: float, joint_type: str = "Butt", pwht_required: str = "Any") -> Dict:
        """Get welding recommendations as dictionary"""
        try:
            result_json = self.get_welding_recommendations(base_material, filler_material, base_thickness, joint_type, pwht_required)
            return json.loads(result_json)
        except Exception as e:
            logger.error(f"Error getting welding recommendations as dict: {e}")
            return {
                "status": "error",
                "message": str(e),
                "welding_procedures": []
            }
    
    def get_recommendations(self, input_data: Dict) -> Dict:
        """Get recommendations in the format expected by streamlit app"""
        try:
            base_material = input_data.get("base_material", "")
            filler_material = input_data.get("filler_material", "")
            thickness = input_data.get("thickness", 0.25)
            joint_type = input_data.get("joint_type", "Butt")
            pwht_required = input_data.get("pwht_required", "Any")
            
            # Get RAG information for both materials
            base_rag_info = self._query_asme_rag_pg_numbers(base_material)
            filler_rag_info = self._query_asme_rag_pg_numbers(filler_material)
            
            # Determine ASME compatibility
            asme_compatibility = self._build_asme_compatibility_info(
                base_material, filler_material, base_rag_info, filler_rag_info
            )
            
            result = self.get_welding_recommendations_dict(
                base_material, filler_material, thickness, joint_type, pwht_required
            )
            
            if result.get("status") == "success":
                return {
                    "compatible_wps": result.get("welding_procedures", []),
                    "recommended_processes": result.get("recommended_processes", []),
                    "total_matches": result.get("total_matches", 0),
                    "input_parameters": result.get("input_parameters", {}),
                    "asme_compatibility": asme_compatibility,
                    "search_details": {
                        "base_material_rag": base_rag_info,
                        "filler_material_rag": filler_rag_info,
                        "matching_method": "RAG + P/G" if base_rag_info and filler_rag_info else "Fallback matching"
                    }
                }
            else:
                return {
                    "compatible_wps": [],
                    "recommended_processes": [],
                    "total_matches": 0,
                    "error": result.get("message", "Unknown error"),
                    "asme_compatibility": asme_compatibility,
                    "search_details": {
                        "base_material_rag": base_rag_info,
                        "filler_material_rag": filler_rag_info,
                        "matching_method": "RAG + P/G" if base_rag_info and filler_rag_info else "Fallback matching"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in get_recommendations: {e}")
            return {
                "compatible_wps": [],
                "recommended_processes": [],
                "total_matches": 0,
                "error": str(e),
                "asme_compatibility": {"compatible": False, "error": str(e)},
                "search_details": {"error": str(e)}
            }
    
    def _build_asme_compatibility_info(self, base_material: str, filler_material: str, 
                                     base_rag_info: Dict, filler_rag_info: Dict) -> Dict:
        """Build ASME compatibility information structure"""
        try:
            # Extract P/G numbers
            base_p_no = base_rag_info.get("p_no", "") if base_rag_info else ""
            base_g_no = base_rag_info.get("g_no", "") if base_rag_info else ""
            filler_p_no = filler_rag_info.get("p_no", "") if filler_rag_info else ""
            filler_g_no = filler_rag_info.get("g_no", "") if filler_rag_info else ""
            
            # Check compatibility
            compatible = False
            if base_p_no and filler_p_no:
                base_pg = self._normalize_pg_number(base_p_no)
                filler_pg = self._normalize_pg_number(filler_p_no)
                compatible = self._check_pg_compatibility(base_pg, filler_pg)
            
            return {
                "compatible": compatible,
                "base_p_number": base_p_no,
                "base_g_number": base_g_no,
                "filler_p_number": filler_p_no,
                "filler_g_number": filler_g_no,
                "base_spec": base_rag_info.get("spec_no", "") if base_rag_info else "",
                "base_grade": base_rag_info.get("grade", "") if base_rag_info else "",
                "filler_spec": filler_rag_info.get("spec_no", "") if filler_rag_info else "",
                "filler_grade": filler_rag_info.get("grade", "") if filler_rag_info else "",
                "base_search_method": base_rag_info.get("search_method", "Not found") if base_rag_info else "Not found",
                "filler_search_method": filler_rag_info.get("search_method", "Not found") if filler_rag_info else "Not found",
                "has_rag_data": bool(base_rag_info and filler_rag_info)
            }
            
        except Exception as e:
            logger.error(f"Error building ASME compatibility info: {e}")
            return {
                "compatible": False,
                "error": str(e),
                "has_rag_data": False
            }

# Example usage and testing functions
def main():
    """Example usage of the welding recommendation API"""
    # Initialize the API
    api = WeldingRecommendationAPI("welding_procedures.json")
    
    # Example 1: Basic recommendation
    print("=== Example 1: Basic Recommendation ===")
    result = api.get_welding_recommendations(
        base_material="A36 Carbon Steel",
        filler_material="E7018",
        base_thickness=0.5,
        pwht_required="Any"
    )
    print(result)
    
    # Example 2: Dictionary result with PWHT Required
    print("\n=== Example 2: Dictionary Result with PWHT Required ===")
    result_dict = api.get_welding_recommendations_dict(
        base_material="SA 516 GR. 70",
        filler_material="E7018",
        base_thickness=0.5,
        pwht_required="Yes"
    )
    print(f"Status: {result_dict.get('status')}")
    print(f"Total matches: {result_dict.get('total_matches')}")
    print(f"Recommended processes: {result_dict.get('recommended_processes')}")
    
    # Example 3: Using the streamlit-compatible method
    print("\n=== Example 3: Streamlit-compatible Method ===")
    input_data = {
        "base_material": "SA 516 GR. 70",
        "filler_material": "E7018",
        "thickness": 0.5,
        "pwht_required": "Yes",
        "enable_rag": True,
        "max_wps": 3
    }
    
    recommendations = api.get_recommendations(input_data)
    print(f"Compatible WPS found: {len(recommendations.get('compatible_wps', []))}")
    print(f"ASME compatible: {recommendations.get('asme_compatibility', {}).get('compatible', False)}")
    
    if recommendations.get('compatible_wps'):
        print("\nFirst WPS details:")
        wps = recommendations['compatible_wps'][0]
        print(f"- WPS: {wps.get('wps_no')}")
        print(f"- Materials: {wps.get('material_from')} → {wps.get('material_to')}")
        print(f"- Electrode: {wps.get('electrode')}")
        print(f"- Thickness: {wps.get('qualified_thick')}")
        print(f"- PWHT: {'Yes' if wps.get('pwht') else 'No'}")


if __name__ == "__main__":
    main()
