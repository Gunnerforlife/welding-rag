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
    
    def _check_pg_compatibility(self, from_p: Optional[int], from_g: Optional[int], 
                              to_p: Optional[int], to_g: Optional[int],
                              wps_from_p: Optional[int], wps_from_g: Optional[int],
                              wps_to_p: Optional[int], wps_to_g: Optional[int]) -> bool:
        """Check if P/G numbers are compatible between input materials and WPS materials"""
        # If any critical P number is missing, we can't verify compatibility
        if None in [from_p, to_p, wps_from_p, wps_to_p]:
            return False
        
        # Direct match - both materials should match their respective P AND G numbers
        direct_match = (
            from_p == wps_from_p and to_p == wps_to_p and
            (from_g == wps_from_g or from_g is None or wps_from_g is None) and
            (to_g == wps_to_g or to_g is None or wps_to_g is None)
        )
        
        # Reverse match - materials could be swapped (from->to, to->from)
        reverse_match = (
            from_p == wps_to_p and to_p == wps_from_p and
            (from_g == wps_to_g or from_g is None or wps_to_g is None) and
            (to_g == wps_from_g or to_g is None or wps_from_g is None)
        )
        
        return direct_match or reverse_match
    
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
    
    def _check_thickness_compatibility(self, input_thickness: float, wps_thickness_range: str) -> bool:
        """Check if input thickness falls within WPS thickness range (both in mm)"""
        try:
            # Handle different formats in qualified_thick field
            thickness_str = wps_thickness_range.strip()
            
            # Handle special cases
            if not thickness_str or thickness_str.upper() in ['NA', 'ALL', 'VN']:
                return True  # Accept if no restrictions
            
            # Both input and WPS thickness are in mm, no conversion needed
            
            # Handle ">" format (e.g., ">25")
            if thickness_str.startswith('>'):
                min_thickness = float(thickness_str[1:])
                return input_thickness >= min_thickness
            
            # Handle "<" format (e.g., "<50")
            if thickness_str.startswith('<'):
                max_thickness = float(thickness_str[1:])
                return input_thickness <= max_thickness
            
            # Handle range format (e.g., "5-72", "1.5-11.12")
            if '-' in thickness_str:
                parts = thickness_str.split('-')
                if len(parts) == 2:
                    min_thickness = float(parts[0].strip())
                    max_thickness = float(parts[1].strip())
                    return min_thickness <= input_thickness <= max_thickness
            else:
                # Single thickness value
                target_thickness = float(thickness_str)
                # Allow some tolerance (±20% or ±2mm, whichever is smaller)
                tolerance = min(target_thickness * 0.2, 2.0)
                return abs(input_thickness - target_thickness) <= tolerance
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse thickness range: {wps_thickness_range} - {e}")
            return True  # Accept if we can't parse (conservative approach)
        
        return False


    
    def _extract_welding_processes(self, electrode_info: str) -> List[str]:
        """Extract welding processes from electrode information"""
        processes = []
        electrode_lower = electrode_info.lower().replace(' ', '').replace('-', '')
        
        # Define process patterns (order matters - more specific first)
        process_patterns = [
            # FCAW (Flux Cored Arc Welding) - check first as it's specific
            ('eh10k', 'FCAW'),
            ('eh14', 'FCAW'),
            ('e71t', 'FCAW'),
            ('e81t', 'FCAW'),
            
            # SMAW (Shielded Metal Arc Welding) - E prefix electrodes
            ('e7018', 'SMAW'),
            ('e6010', 'SMAW'),
            ('e6013', 'SMAW'),
            ('e316l', 'SMAW'),  # E316L is SMAW
            ('e308l', 'SMAW'),  # E308L is SMAW
            ('e309l', 'SMAW'),  # E309L is SMAW
            ('e316', 'SMAW'),
            ('e308', 'SMAW'),
            ('e309', 'SMAW'),
            ('e60', 'SMAW'),
            ('e70', 'SMAW'),
            ('e80', 'SMAW'),
            ('e90', 'SMAW'),
            
            # GTAW (Gas Tungsten Arc Welding) - ER stainless steel filler rods
            ('er316l', 'GTAW'),
            ('er308l', 'GTAW'),
            ('er309l', 'GTAW'),
            ('er321', 'GTAW'),
            ('er347', 'GTAW'),
            ('er316', 'GTAW'),
            ('er308', 'GTAW'),
            ('er309', 'GTAW'),
            
            # GMAW (Gas Metal Arc Welding) - ER carbon steel solid wires
            ('er70s', 'GMAW'),
            ('er80s', 'GMAW'),
            ('er90s', 'GMAW'),
            ('er70', 'GMAW'),
            ('er80', 'GMAW'),
            ('er90', 'GMAW'),
            
            # SAW (Submerged Arc Welding)
            ('flux', 'SAW'),
            ('saw', 'SAW'),
            ('submerged', 'SAW'),
            
            # Process keywords
            ('mig', 'GMAW'),
            ('tig', 'GTAW'),
            ('stick', 'SMAW'),
            ('arc', 'SMAW')
        ]
        
        # Split by common delimiters to handle multiple electrodes
        parts = electrode_lower.replace('&', ',').replace(';', ',').split(',')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check each pattern against this part
            for pattern, process in process_patterns:
                if pattern in part:
                    if process not in processes:
                        processes.append(process)
                    break  # Found a match, don't check other patterns for this part
        
        return processes if processes else ['SMAW']  # Default to SMAW
    
    def _find_matching_wps(self, from_material: str, to_material: str, 
                          thickness: float, joint_type: str = "Butt", pwht_required: str = "Any") -> List[Dict]:
        """Find matching WPS based on materials, thickness, and PWHT requirements"""
        matching_wps = []
        
        # Get ASME RAG info for both materials
        from_material_rag_info = self._query_asme_rag_pg_numbers(from_material)
        to_material_rag_info = self._query_asme_rag_pg_numbers(to_material)
        
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
                
                if from_material_rag_info and to_material_rag_info:
                    # Use P/G number matching for both materials
                    from_p = self._normalize_pg_number(from_material_rag_info.get('p_no'))
                    from_g = self._normalize_pg_number(from_material_rag_info.get('g_no'))
                    to_p = self._normalize_pg_number(to_material_rag_info.get('p_no'))
                    to_g = self._normalize_pg_number(to_material_rag_info.get('g_no'))
                    wps_from_p = self._normalize_pg_number(wps.get('p_from'))
                    wps_from_g = self._normalize_pg_number(wps.get('g_from'))
                    wps_to_p = self._normalize_pg_number(wps.get('p_to'))
                    wps_to_g = self._normalize_pg_number(wps.get('g_to'))
                    
                    material_compatible = self._check_pg_compatibility(
                        from_p, from_g, to_p, to_g, 
                        wps_from_p, wps_from_g, wps_to_p, wps_to_g
                    )
                else:
                    # Fallback to material name matching
                    wps_from_material = wps.get('material_from', '')
                    wps_to_material = wps.get('material_to', '')
                    
                    # Check both direct and reverse material matching
                    direct_match = (
                        self._fallback_material_match(from_material, wps_from_material) and
                        self._fallback_material_match(to_material, wps_to_material)
                    )
                    reverse_match = (
                        self._fallback_material_match(from_material, wps_to_material) and
                        self._fallback_material_match(to_material, wps_from_material)
                    )
                    
                    material_compatible = direct_match or reverse_match
                
                if not material_compatible:
                    continue
                
                # Check thickness compatibility using qualified_thick field
                wps_thickness = wps.get('qualified_thick', '')
                if wps_thickness and not self._check_thickness_compatibility(thickness, wps_thickness):
                    continue
                
                # If we get here, the WPS is compatible
                # Use the actual welding process field from JSON instead of inferring from electrode
                wps_with_process = wps.copy()
                # The welding process is now directly available in the JSON
                welding_process = wps.get('welding_process', '')
                if welding_process:
                    # Convert single process string to list for consistency
                    if isinstance(welding_process, str):
                        # Split on + to handle combined processes like "GTAW + SMAW"
                        processes = [p.strip() for p in welding_process.split('+')]
                    else:
                        processes = welding_process
                    wps_with_process['welding_process'] = processes
                else:
                    # Fallback to inferring from electrode if welding_process is empty
                    electrode_info = wps.get('electrode', '')
                    processes = self._extract_welding_processes(electrode_info)
                    wps_with_process['welding_process'] = processes
                matching_wps.append(wps_with_process)
                
            except Exception as e:
                logger.warning(f"Error processing WPS {wps.get('wps_no', 'Unknown')}: {e}")
                continue
        
        return matching_wps
    
    def get_welding_recommendations(self, from_material: str, to_material: str,
                                  thickness: float, joint_type: str = "Butt", pwht_required: str = "Any") -> str:
        """Get welding recommendations as JSON string"""
        try:
            matching_wps = self._find_matching_wps(from_material, to_material, thickness, joint_type, pwht_required)
            
            # Limit to 5 WPS
            limited_wps = matching_wps[:5]
            
            result = {
                "status": "success",
                "input_parameters": {
                    "from_material": from_material,
                    "to_material": to_material,
                    "thickness": thickness,
                    "joint_type": joint_type,
                    "pwht_required": pwht_required
                },
                "total_matches": len(matching_wps),
                "returned_wps_count": len(limited_wps),
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
    
    def get_welding_recommendations_dict(self, from_material: str, to_material: str,
                                       thickness: float, joint_type: str = "Butt", pwht_required: str = "Any") -> Dict:
        """Get welding recommendations as dictionary"""
        try:
            result_json = self.get_welding_recommendations(from_material, to_material, thickness, joint_type, pwht_required)
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
            from_material = input_data.get("from_material", input_data.get("base_material", ""))
            to_material = input_data.get("to_material", input_data.get("filler_material", ""))
            thickness = input_data.get("thickness", 12.7)  # Default 12.7mm (0.5 inch equivalent)
            joint_type = input_data.get("joint_type", "Butt")
            pwht_required = input_data.get("pwht_required", "Any")
            
            # Get RAG information for both materials
            from_material_rag_info = self._query_asme_rag_pg_numbers(from_material)
            to_material_rag_info = self._query_asme_rag_pg_numbers(to_material)
            
            # Determine ASME compatibility
            asme_compatibility = self._build_asme_compatibility_info(
                from_material, to_material, from_material_rag_info, to_material_rag_info
            )
            
            result = self.get_welding_recommendations_dict(
                from_material, to_material, thickness, joint_type, pwht_required
            )
            
            if result.get("status") == "success":
                return {
                    "compatible_wps": result.get("welding_procedures", []),
                    "total_matches": result.get("total_matches", 0),
                    "input_parameters": result.get("input_parameters", {}),
                    "asme_compatibility": asme_compatibility,
                    "search_details": {
                        "from_material_rag": from_material_rag_info,
                        "to_material_rag": to_material_rag_info,
                        "matching_method": "RAG + P/G" if from_material_rag_info and to_material_rag_info else "Fallback matching"
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
                        "from_material_rag": from_material_rag_info,
                        "to_material_rag": to_material_rag_info,
                        "matching_method": "RAG + P/G" if from_material_rag_info and to_material_rag_info else "Fallback matching"
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
    
    def _build_asme_compatibility_info(self, from_material: str, to_material: str, 
                                     from_rag_info: Dict, to_rag_info: Dict) -> Dict:
        """Build ASME compatibility information structure"""
        try:
            # Extract P/G numbers
            from_p_no = from_rag_info.get("p_no", "") if from_rag_info else ""
            from_g_no = from_rag_info.get("g_no", "") if from_rag_info else ""
            to_p_no = to_rag_info.get("p_no", "") if to_rag_info else ""
            to_g_no = to_rag_info.get("g_no", "") if to_rag_info else ""
            
            # Check compatibility - now we need to check if we can find a WPS that matches both materials
            compatible = False
            if from_p_no and to_p_no:
                from_pg = self._normalize_pg_number(from_p_no)
                to_pg = self._normalize_pg_number(to_p_no)
                # For compatibility info, we just check if materials are in compatible groups
                # The actual WPS matching will be done in the matching function
                compatible_groups = [
                    {1, 2, 3},  # Low carbon steels
                    {4, 5, 6},  # Low alloy steels
                    {8, 9, 10}, # Stainless steels
                ]
                
                # Check if both materials are in the same compatible group
                for group in compatible_groups:
                    if from_pg in group and to_pg in group:
                        compatible = True
                        break
            
            return {
                "compatible": compatible,
                "from_p_number": from_p_no,
                "from_g_number": from_g_no,
                "to_p_number": to_p_no,
                "to_g_number": to_g_no,
                "from_spec": from_rag_info.get("spec_no", "") if from_rag_info else "",
                "from_grade": from_rag_info.get("grade", "") if from_rag_info else "",
                "to_spec": to_rag_info.get("spec_no", "") if to_rag_info else "",
                "to_grade": to_rag_info.get("grade", "") if to_rag_info else "",
                "from_search_method": from_rag_info.get("search_method", "Not found") if from_rag_info else "Not found",
                "to_search_method": to_rag_info.get("search_method", "Not found") if to_rag_info else "Not found",
                "has_rag_data": bool(from_rag_info and to_rag_info)
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
        from_material="A36",
        to_material="A36",
        thickness=12.7,  # 12.7mm thickness
        pwht_required="Any"
    )
    print(result)
    
    # Example 2: Dictionary result with PWHT Required
    print("\n=== Example 2: Dictionary Result with PWHT Required ===")
    result_dict = api.get_welding_recommendations_dict(
        from_material="SA 516 GR. 70",
        to_material="SA 516 GR. 70",
        thickness=25.0,  # 25mm thickness
        pwht_required="Yes"
    )
    print(f"Status: {result_dict.get('status')}")
    print(f"Total matches: {result_dict.get('total_matches')}")
    print(f"Recommended processes: {result_dict.get('recommended_processes')}")
    
    # Example 3: Using the streamlit-compatible method
    print("\n=== Example 3: Streamlit-compatible Method ===")
    input_data = {
        "from_material": "SA 516 GR. 70",
        "to_material": "SA 516 GR. 70",
        "thickness": 36.0,  # 36mm thickness
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
