import os
import pandas as pd
from typing import Dict, List, Optional
import json
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.readers.file import PDFReader
import pymupdf as fitz
import re
from dataclasses import dataclass
import hashlib

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


@dataclass
class MaterialSpec:
    """Data class for material specification with P and G numbers"""
    spec_no: str
    grade: str
    p_no: str
    g_no: str
    uns_no: str = ""
    tensile_strength: str = ""
    composition: str = ""
    product_form: str = ""
    thickness_limits: str = ""
    additional_numeric: str = ""


class ASMETableExtractor:
    """Extract tabular data from ASME BPVC Section IX PDF pages 151-239"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.start_page = 150  # 0-indexed (page 151)
        self.end_page = 238    # 0-indexed (page 239)
    
    def extract_tables_from_pages(self) -> List[MaterialSpec]:
        """Extract table data from specified pages"""
        doc = fitz.open(self.pdf_path)
        all_specs = []
        
        for page_num in range(self.start_page, self.end_page + 1):
            try:
                page = doc.load_page(page_num)
                text = page.get_text()
                specs = self._parse_table_from_text(text, page_num)
                all_specs.extend(specs)
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")
                continue
        
        doc.close()
        return all_specs
    
    def _parse_table_from_text(self, text: str, page_num: int) -> List[MaterialSpec]:
        """Parse table data from extracted text using the observed structure"""
        specs = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for various specification patterns
            spec_patterns = [
                # ASME specifications
                (r'^(A\s+or\s+)?S?A[–-]\d+', 'ASME'),
                (r'^A\d+', 'ASME'),
                # API specifications
                (r'^API 5L', 'API'),
                # JIS specifications  
                (r'^JIS or SA/JIS G\d+', 'JIS'),
                (r'^SA/JIS G\d+', 'JIS'),
                # MSS specifications
                (r'^MSS SP[–-]\d+', 'MSS'),
                # CSA specifications
                (r'^CSA Z\d+\.\d+', 'CSA'),
                # A860 type specifications
                (r'^A860', 'A860'),
            ]
            
            spec_type = None
            for pattern, stype in spec_patterns:
                if re.match(pattern, line):
                    spec_type = stype
                    break
            
            if spec_type:
                try:
                    spec_no = line
                    
                    # The next line should be the grade/designation
                    if i + 1 < len(lines):
                        grade = lines[i + 1]
                        
                        # Look ahead for P-No and G-No
                        j = i + 2
                        uns_no = ""
                        tensile_strength = ""
                        p_no = ""
                        g_no = ""
                        composition = ""
                        product_form = ""
                        thickness_limits = ""
                        
                        # Look ahead window varies by spec type
                        look_ahead = 15 if spec_type == 'ASME' else 10
                        
                        # Extract all columns systematically based on position
                        # Standard table structure after spec_no and grade:
                        # +2: UNS number (or similar identifier)
                        # +3: Tensile strength (format: "XX (YY)")  
                        # +4: P-No
                        # +5: G-No
                        # +6: Thickness limit
                        # +7: Additional numeric value
                        # +8: Composition (contains chemical symbols)
                        # +9: Product form
                        
                        if j < len(lines):
                            uns_no = lines[j] if j < len(lines) else ""  # +2
                        if j + 1 < len(lines):
                            tensile_strength = lines[j + 1] if re.match(r'^\d+\s*\(\d+\)', lines[j + 1]) else ""  # +3
                        if j + 2 < len(lines):
                            p_no = lines[j + 2] if re.match(r'^\d{1,2}[A-Z]?$', lines[j + 2]) else ""  # +4
                        if j + 3 < len(lines):
                            g_no = lines[j + 3] if re.match(r'^\d+(\.\d+)?$', lines[j + 3]) else ""  # +5
                        if j + 4 < len(lines):
                            thickness_limits = lines[j + 4] if re.match(r'^\d+(\.\d+)?$', lines[j + 4]) else ""  # +6
                        
                        # Additional columns - extract everything else
                        additional_numeric = ""
                        if j + 5 < len(lines):
                            additional_numeric = lines[j + 5] if re.match(r'^\d+(\.\d+)?$', lines[j + 5]) else ""  # +7
                        
                        # Find composition (contains chemical symbols)
                        for k in range(j + 6, min(len(lines), j + 10)):
                            if k < len(lines) and re.search(r'(C|Mn|Si|Cr|Mo|Ni)', lines[k]):
                                composition = lines[k]
                                break
                        
                        # Find product form (next line after composition or scan area)
                        for k in range(j + 6, min(len(lines), j + 12)):
                            if k < len(lines) and any(keyword in lines[k].lower() for keyword in 
                                   ['plate', 'pipe', 'tube', 'bar', 'forgings', 'flanges', 'fittings', 
                                    'welded', 'smls', 'seamless', 'bars', 'shapes']):
                                product_form = lines[k]
                                break
                        
                        # Only add if we found both P-No and G-No
                        if p_no and g_no:
                            specs.append(MaterialSpec(
                                spec_no=spec_no,
                                grade=grade,
                                p_no=p_no,
                                g_no=g_no,
                                uns_no=uns_no,
                                tensile_strength=tensile_strength,
                                composition=composition,
                                product_form=product_form,
                                thickness_limits=thickness_limits,
                                additional_numeric=additional_numeric
                            ))
                
                except Exception as e:
                    print(f"Error parsing {spec_type} spec starting at line '{line}' on page {page_num + 1}: {e}")
            
            i += 1
        
        return specs


class ASMEMaterialRAG:
    """RAG system for ASME material P and G number lookup"""
    
    def __init__(self, pdf_path: str, use_vector_search: bool = True):
        self.pdf_path = pdf_path
        self.specs: List[MaterialSpec] = []
        self.index: Optional[VectorStoreIndex] = None
        self.persist_dir = "cache/asme_rag_index"
        self.use_vector_search = use_vector_search
        self.pdf_available = os.path.exists(pdf_path) if pdf_path else False
        
        # Configure LlamaIndex settings only if vector search is requested
        if self.use_vector_search:
            try:
                Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
                Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI components: {e}")
                print("Falling back to text-based search only")
                self.use_vector_search = False
    
    def build_index(self):
        """Build or load the vector index using enhanced document creation"""
        if not self.pdf_available:
            print(f"Warning: PDF file not found at {self.pdf_path}")
            print("RAG system will use cached data or return errors for new queries")
            # Try to load cached specs
            specs_cache_file = os.path.join(self.persist_dir, "material_specs.json")
            if os.path.exists(specs_cache_file):
                print("Loading cached material specifications...")
                with open(specs_cache_file, 'r') as f:
                    specs_data = json.load(f)
                    self.specs = [MaterialSpec(**spec) for spec in specs_data]
                print(f"Loaded {len(self.specs)} cached material specifications")
            else:
                print("No cached data available. System will return errors for queries.")
            return
        
        # Check if we have cached specs first
        specs_cache_file = os.path.join(self.persist_dir, "material_specs.json")
        
        if os.path.exists(specs_cache_file) and not self.specs:
            try:
                print("Loading cached material specifications...")
                with open(specs_cache_file, 'r') as f:
                    specs_data = json.load(f)
                self.specs = [MaterialSpec(**spec) for spec in specs_data]
                print(f"Loaded {len(self.specs)} cached material specifications")
            except Exception as e:
                print(f"Failed to load cached specs: {e}")
                self._extract_table_data()
        elif not self.specs:
            # Extract specs if not cached
            self._extract_table_data()
        
        # If vector search is not available, skip vector index building
        if not self.use_vector_search:
            print("Vector search not available - using text-based search only")
            return
        
        # Check if persisted index exists
        if os.path.exists(self.persist_dir):
            try:
                print("Loading existing index from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self.index = load_index_from_storage(storage_context)
                print("Successfully loaded existing index!")
                return
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                print("Building new index...")
        
        # Build new index if no existing one found
        print("Building new vector index with enhanced documents...")
        try:
            # Create enhanced documents for each spec
            documents = []
            for spec in self.specs:
                # Create multiple searchable variations for better RAG matching
                variations = self._create_spec_variations(spec)
                searchable_text = " | ".join(variations)
                
                # Enhanced metadata with normalized fields for post-processing
                metadata = {
                    "spec_no": spec.spec_no,
                    "spec_no_normalized": spec.spec_no.upper().replace(" ", "").replace("–", "").replace("—", ""),
                    "grade": spec.grade,
                    "grade_normalized": spec.grade.upper().replace(" ", "").replace(".", ""),
                    "p_no": spec.p_no,
                    "g_no": spec.g_no,
                    "uns_no": spec.uns_no,
                    "tensile_strength": spec.tensile_strength,
                    "composition": spec.composition,
                    "product_form": spec.product_form,
                    "thickness_limits": spec.thickness_limits,
                    "additional_numeric": spec.additional_numeric
                }
                
                doc = Document(
                    text=searchable_text,
                    metadata=metadata
                )
                documents.append(doc)
            
            # Build vector store index
            self.index = VectorStoreIndex.from_documents(documents)
            
            # Persist the index
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"Enhanced vector index built and persisted to {self.persist_dir}")
            
        except Exception as e:
            print(f"Failed to build vector index: {e}")
            print("Continuing with text-based search only")
            self.use_vector_search = False
    
    def _create_spec_variations(self, spec: MaterialSpec) -> List[str]:
        """Create multiple searchable variations of a spec for better RAG matching"""
        variations = []
        
        # Original spec
        variations.append(f"{spec.spec_no} {spec.grade}")
        
        # Extract spec number for variations
        spec_num_match = re.search(r'(\d+)', spec.spec_no)
        if spec_num_match:
            spec_num = spec_num_match.group(1)
            
            # Common variations
            variations.extend([
                f"SA {spec_num} {spec.grade}",           # SA 516 70
                f"SA-{spec_num} {spec.grade}",           # SA-516 70
                f"ASTM A {spec_num} {spec.grade}",       # ASTM A 516 70
                f"A {spec_num} {spec.grade}",            # A 516 70
                f"SA {spec_num} Grade {spec.grade}",     # SA 516 Grade 70
                f"SA {spec_num} GR {spec.grade}",        # SA 516 GR 70
            ])
        
        # Add composition for semantic matching
        if spec.composition:
            variations.append(f"{spec.composition} steel {spec.spec_no}")
        
        return variations
    
    def _extract_table_data(self):
        """Extract table data from PDF"""
        print("Extracting table data from PDF...")
        extractor = ASMETableExtractor(self.pdf_path)
        self.specs = extractor.extract_tables_from_pages()
        print(f"Extracted {len(self.specs)} material specifications")
        
        if not self.specs:
            raise ValueError("No material specifications found in the PDF")
        
        # Show first few specs for verification
        print("First few extracted specs:")
        for spec in self.specs[:5]:
            print(f"  {spec.spec_no} {spec.grade} - P:{spec.p_no}, G:{spec.g_no}")
        
        # Cache the specs
        self._cache_specs()
    
    def _cache_specs(self):
        """Cache extracted specs to JSON for faster loading"""
        os.makedirs(self.persist_dir, exist_ok=True)
        specs_cache_file = os.path.join(self.persist_dir, "material_specs.json")
        
        try:
            specs_data = []
            for spec in self.specs:
                specs_data.append({
                    "spec_no": spec.spec_no,
                    "grade": spec.grade,
                    "p_no": spec.p_no,
                    "g_no": spec.g_no,
                    "uns_no": spec.uns_no,
                    "tensile_strength": spec.tensile_strength,
                    "composition": spec.composition,
                    "product_form": spec.product_form,
                    "thickness_limits": spec.thickness_limits,
                    "additional_numeric": spec.additional_numeric
                })
            
            with open(specs_cache_file, 'w') as f:
                json.dump(specs_data, f, indent=2)
            print(f"Material specifications cached to {specs_cache_file}")
        except Exception as e:
            print(f"Failed to cache material specs: {e}")
    
    def query_material(self, material: str, top_k: int = 10) -> Dict:
        """Enhanced RAG query with intelligent preprocessing and post-processing"""
        # Try vector search first if available
        if self.use_vector_search and self.index:
            try:
                return self._enhanced_rag_search(material, top_k)
            except Exception as e:
                print(f"Vector search failed: {e}")
                print("Falling back to text search")
        
        # Fallback to text-based search
        return self._text_based_search(material)
    
    def _enhanced_rag_search(self, material: str, top_k: int) -> Dict:
        """Enhanced RAG search with text search intelligence"""
        print(f"RAG searching for: {material}")
        
        # Removed non-ASME filtering - let the system find what it can from the ASME database
        
        # Extract spec and grade using same logic as text search
        spec_info = self._extract_spec_and_grade(material)
        
        # Create multiple query variations
        queries = [material]  # Original query
        
        if spec_info["spec_num"]:
            # Add variations based on extracted info
            spec_num = spec_info["spec_num"]
            grade_info = spec_info["grade_info"]
            
            queries.extend([
                f"SA {spec_num} {grade_info}",
                f"A or SA {spec_num} {grade_info}",
                f"ASTM A {spec_num} {grade_info}",
                f"specification {spec_num} grade {grade_info}",
            ])
        
        # Query with multiple variations and collect results
        all_results = []
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        
        for query in queries[:3]:  # Use top 3 query variations
            if query.strip():
                try:
                    response = query_engine.query(f"Find material specification: {query}")
                    if response.source_nodes:
                        for node in response.source_nodes:
                            all_results.append((node, query))
                except Exception as e:
                    print(f"Query '{query}' failed: {e}")
                    continue
        
        if not all_results:
            return {"error": "No matching material found"}
        
        # Apply intelligent scoring using text search logic
        scored_results = []
        for node, query_used in all_results:
            metadata = node.metadata
            score = self._calculate_rag_score(spec_info, metadata, node.score if hasattr(node, 'score') else 0.5)
            scored_results.append((score, metadata, query_used))
        
        # Sort by score and return best match
        scored_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_metadata, best_query = scored_results[0]
        
        # Apply confidence threshold - if score is too low, treat as not found
        confidence_threshold = 1.5  # Lowered threshold since we removed filtering
        if best_score < confidence_threshold:
            return {"error": f"No confident match found for '{material}'. Best match had low confidence score: {best_score:.2f}"}
        
        print(f"Best RAG match (score={best_score:.2f}) from query '{best_query}': {best_metadata['spec_no']} {best_metadata['grade']}")
        
        # Show top matches for debugging
        print("Top RAG matches:")
        for i, (score, metadata, query_used) in enumerate(scored_results[:3]):
            print(f"  {i+1}. {metadata['spec_no']} {metadata['grade']} - P:{metadata['p_no']}, G:{metadata['g_no']} (score={score:.2f})")
        
        return {
            "p_no": best_metadata.get("p_no", ""),
            "g_no": best_metadata.get("g_no", ""),
            "spec_no": best_metadata.get("spec_no", ""),
            "grade": best_metadata.get("grade", ""),
            "composition": best_metadata.get("composition", ""),
            "thickness_limits": best_metadata.get("thickness_limits", ""),
            "search_method": "enhanced_rag",
            "rag_score": best_score
        }
    
    def _extract_spec_and_grade(self, material: str) -> Dict:
        """Extract spec number and grade using same logic as text search"""
        material_clean = material.upper().replace(".", "").replace(",", "").strip()
        
        # Enhanced patterns to handle various specification types
        spec_patterns = [
            # API patterns (check first to avoid conflicts)
            r'API\s+5L\s+(?:GR\.?\s+|GRADE\s+)?(?:X\s*)?(\w+)',  # API 5L X60, API 5L grade B, API 5L GR. B
            # ASME patterns
            r'(?:A\s+OR\s+)?S?A[-–]?\s*(\d+)',     # SA-516, A or SA-516, SA 516
            r'ASTM\s+A\s*(\d+)',                   # ASTM A 105 -> match A 105
            r'A\s*(\d+)',                          # A 516, A 105
            # JIS patterns  
            r'JIS\s+(?:OR\s+SA/)?JIS\s+G(\d+)',    # JIS or SA/JIS G4303
            r'SA/JIS\s+G(\d+)',                    # SA/JIS G4303
            # MSS patterns
            r'MSS\s+SP[-–](\d+)',                  # MSS SP-75
            # CSA patterns
            r'CSA\s+Z(\d+)',                       # CSA Z245 (simplified)
            # A860 patterns
            r'A(860)',                             # A860
        ]
        
        grade_patterns = [
            r'X(\d+)',                                 # X60, X65 (for API 5L) - check first
            r'GR\.?\s*([A-Z0-9\-]+)',                  # GR. 70, GR. X-7
            r'GRADE\s+([A-Z0-9]+)',                    # GRADE B, GRADE 70
            r'CLASS\s+([A-Z0-9\-]+)',                  # CLASS 1
            r'TYPE\s+([A-Z0-9\-]+)',                   # TYPE A
            r'\s([A-Z])\s*$',                          # Single letter at end: "SA 106 B"
            r'WPHY[-\s]*(\d+)',                        # WPHY 42, WPHY-46
            r'(\d+)',                                  # Plain numbers like 290, 304
        ]
        
        spec_num = ""
        grade_info = ""
        spec_type = ""
        
        # Try to extract spec number and identify type
        for pattern in spec_patterns:
            match = re.search(pattern, material_clean)
            if match:
                if 'API' in pattern:
                    spec_type = 'API'
                    # For API 5L, the grade is extracted from the pattern match
                    grade_info = match.group(1)  # Can be letters (B, A) or numbers (60, 65, etc.)
                    # Remove X prefix if present
                    if grade_info.startswith('X'):
                        grade_info = grade_info[1:]
                elif 'CSA' in pattern:
                    spec_num = match.group(1)  # The number part
                    spec_type = 'CSA'
                elif 'JIS' in pattern:
                    spec_num = match.group(1)  # The number part
                    spec_type = 'JIS'
                elif 'MSS' in pattern:
                    spec_num = match.group(1)  # The number part
                    spec_type = 'MSS'
                elif 'A860' in pattern:
                    spec_num = match.group(1)  # The number part
                    spec_type = 'A860'
                else:
                    spec_num = match.group(1)  # The number part
                    spec_type = 'ASME'
                
                print(f"Pattern '{pattern}' matched, spec_num='{spec_num}', type='{spec_type}'")
                break
        
        # Try to extract grade if not already found
        if not grade_info:
            for pattern in grade_patterns:
                match = re.search(pattern, material_clean)
                if match:
                    grade_info = match.group(1)
                    print(f"Grade pattern '{pattern}' matched, grade_info='{grade_info}'")
                    break
        
        return {"spec_num": spec_num, "grade_info": grade_info, "spec_type": spec_type}
    
    def _calculate_rag_score(self, spec_info: Dict, metadata: Dict, similarity_score: float) -> float:
        """Calculate enhanced score combining similarity and text search logic"""
        score = similarity_score  # Base semantic similarity
        
        # Strong bonus for exact spec number match
        if spec_info["spec_num"]:
            spec_no_normalized = metadata.get("spec_no_normalized", "")
            
            # Check for exact match (e.g., "36" matches "A–36" but not "1036" or "369")
            if f"–{spec_info['spec_num']}" in spec_no_normalized or f"A{spec_info['spec_num']}" in spec_no_normalized:
                score += 3.0  # Higher bonus for exact spec match
            elif spec_info["spec_num"] in spec_no_normalized:
                # Partial match, but check if it's at word boundary
                if re.search(rf'\b{spec_info["spec_num"]}\b', spec_no_normalized):
                    score += 2.0  # Good bonus for word boundary match
                else:
                    score += 0.5  # Small bonus for partial match
        
        # Bonus for grade match (same as text search)
        if spec_info["grade_info"] and spec_info["grade_info"] in metadata.get("grade_normalized", ""):
            score += 2.0
        elif spec_info["grade_info"] and any(part in metadata.get("grade_normalized", "") for part in spec_info["grade_info"].split("-")):
            score += 1.0
        
        return score
    
    def _text_based_search(self, material: str) -> Dict:
        """Enhanced text-based search for ASME material specifications"""
        print(f"Searching for: {material}")
        
        # Clean and normalize input
        material_clean = material.upper().replace(".", "").replace(",", "").strip()
        
        # Enhanced spec extraction patterns for all types
        spec_patterns = [
            # API patterns (check first to avoid conflicts)
            r'API\s+5L\s+(?:GR\.?\s+|GRADE\s+)?(?:X\s*)?(\w+)',  # API 5L X60, API 5L grade B, API 5L GR. B
            # ASME patterns
            r'(?:A\s+OR\s+)?S?A[-–]?\s*(\d+)',      # SA-516, A or SA-516, SA 516
            r'ASTM\s+A\s*(\d+)',                    # ASTM A 105 -> match A 105
            r'A\s*(\d+)',                           # A 516, A 105
            # JIS patterns  
            r'JIS\s+(?:OR\s+SA/)?JIS\s+G(\d+)',     # JIS or SA/JIS G4303
            r'SA/JIS\s+G(\d+)',                     # SA/JIS G4303
            # MSS patterns
            r'MSS\s+SP[-–](\d+)',                   # MSS SP-75
            # CSA patterns
            r'CSA\s+Z(\d+)',                        # CSA Z245
            # A860 patterns
            r'A(860)',                              # A860
        ]
        
        # Extract grade patterns
        grade_patterns = [
            r'(?:X\s*)?(\w+)',                      # X60, X 60, A, B, etc. (for API 5L)
            r'GR\.?\s*([A-Z0-9\-]+)',               # GR. 70, GR. X-7
            r'GRADE\s+([A-Z0-9]+)',                 # GRADE B, GRADE 70
            r'WPHY[-\s]*(\d+)',                     # WPHY 42, WPHY-46
            r'SUS\s*(\d+)',                         # SUS 304, SUS 302
        ]
        
        spec_num = ""
        grade_info = ""
        spec_type = ""
        
        # Try to extract spec number and identify type
        for pattern in spec_patterns:
            match = re.search(pattern, material_clean)
            if match:
                if 'API' in pattern:
                    spec_type = 'API'
                    spec_num = '5L'  # API 5L is the spec
                    grade_info = match.group(1)  # X60, A, B, etc.
                    # Remove X prefix if present
                    if grade_info.startswith('X'):
                        grade_info = grade_info[1:]
                    print(f"API 5L pattern matched, grade_info='{grade_info}'")
                else:
                    spec_num = match.group(1)  # The number part
                    if 'CSA' in pattern:
                        spec_type = 'CSA'
                    elif 'JIS' in pattern:
                        spec_type = 'JIS'
                    elif 'MSS' in pattern:
                        spec_type = 'MSS'
                    elif 'A860' in pattern:
                        spec_type = 'A860'
                    else:
                        spec_type = 'ASME'
                    print(f"Pattern '{pattern}' matched, spec_num='{spec_num}', type='{spec_type}'")
                break
        
        if not spec_num and spec_type != 'API':
            return {"error": f"Could not extract specification number from: {material}"}
        
        print(f"Extracted: spec_type='{spec_type}', spec_num='{spec_num}', grade_info='{grade_info}'")
        
        # Search through cached specs with type-specific logic
        matches = []
        for spec in self.specs:
            if spec_type == 'API':
                # For API 5L, match both spec and grade
                if 'API 5L' in spec.spec_no and grade_info:
                    spec_grade_clean = spec.grade.upper().replace(' ', '')
                    grade_clean = grade_info.upper().replace(' ', '')
                    
                    if grade_clean == spec_grade_clean or f'X{grade_clean}' == spec_grade_clean:
                        score = 3.0  # High score for exact match
                        matches.append((score, spec))
                        print(f"API 5L match: {spec.spec_no} {spec.grade}")
            else:
                # Original ASME/other logic
                spec_normalized = spec.spec_no.upper().replace(" ", "").replace("–", "").replace("—", "")
                grade_normalized = spec.grade.upper().strip()
                
                # Check if spec number matches
                if spec_num in spec_normalized:
                    score = 1.0
                    
                    # Bonus for grade match
                    if grade_info and grade_info in grade_normalized:
                        score += 2.0
                    elif grade_info and any(part in grade_normalized for part in grade_info.split("-")):
                        score += 1.0
                    
                    matches.append((score, spec))
        
        if not matches:
            return {"error": f"No matching specifications found for: {material}"}
        
        # Sort by score and return best match
        matches.sort(key=lambda x: x[0], reverse=True)
        best_score, best_spec = matches[0]
        
        print(f"Best match (score={best_score}): {best_spec.spec_no} {best_spec.grade}")
        
        # Show top 3 matches for debugging
        print("Top matches:")
        for i, (score, spec) in enumerate(matches[:3]):
            print(f"  {i+1}. {spec.spec_no} {spec.grade} - P:{spec.p_no}, G:{spec.g_no} (score={score})")
        
        return {
            "p_no": best_spec.p_no,
            "g_no": best_spec.g_no,
            "spec_no": best_spec.spec_no,
            "grade": best_spec.grade,
            "composition": best_spec.composition,
            "thickness_limits": best_spec.thickness_limits,
            "search_method": "enhanced_text_search"
        }
    
    def search_by_spec_and_grade(self, spec_no: str, grade: str = "") -> Dict:
        """Direct search by specification number and grade"""
        # Normalize for comparison
        normalized_spec = spec_no.replace(" ", "").upper().replace("–", "-").replace("—", "-")
        normalized_grade = grade.upper().strip()
        
        for spec in self.specs:
            spec_normalized = spec.spec_no.replace(" ", "").upper().replace("–", "-").replace("—", "-")
            grade_normalized = spec.grade.upper().strip()
            
            if spec_normalized.endswith(normalized_spec.split("-")[-1]):  # Match the number part
                if not grade or normalized_grade in grade_normalized or grade_normalized in normalized_grade:
                    return {
                        "p_no": spec.p_no,
                        "g_no": spec.g_no,
                        "spec_no": spec.spec_no,
                        "grade": spec.grade,
                        "composition": spec.composition
                    }
        
        return {"error": f"Specification {spec_no} {grade} not found"}
    
    def list_all_sa516_specs(self) -> List[Dict]:
        """List all SA 516 specifications for debugging"""
        sa516_specs = []
        for spec in self.specs:
            if "516" in spec.spec_no:
                sa516_specs.append({
                    "spec_no": spec.spec_no,
                    "grade": spec.grade,
                    "p_no": spec.p_no,
                    "g_no": spec.g_no,
                    "composition": spec.composition
                })
        return sa516_specs


# Global RAG instance
_rag_instance = None

def set_pdf_path(pdf_path: str):
    """Set the PDF path for the RAG system"""
    global _rag_instance
    _rag_instance = None  # Force re-initialization with new path
    return pdf_path

def get_p_and_g_numbers(material: str, use_vector_search: bool = True, pdf_path: str = None) -> Dict:
    """
    Simple function to get P and G numbers for a given material using enhanced RAG
    
    Args:
        material (str): Material specification (e.g., "SA 516 GR. 70")
        use_vector_search (bool): Whether to use vector search (requires OpenAI API key)
        pdf_path (str): Path to ASME PDF file (optional)
        
    Returns:
        Dict: Contains p_no, g_no and other material info
    """
    global _rag_instance
    
    # Initialize RAG system only once
    if _rag_instance is None:
        default_pdf_path = pdf_path or "data/ASME BPVC 2023 Section IX_removed.pdf"
        _rag_instance = ASMEMaterialRAG(default_pdf_path, use_vector_search=use_vector_search)
        print("Initializing ASME material lookup system...")
        _rag_instance.build_index()
        print("System ready!")
    
    # Query the material
    result = _rag_instance.query_material(material)
    
    if result.get("error"):
        return {"error": result["error"]}
    
    return {
        "p_no": result.get("p_no", ""),
        "g_no": result.get("g_no", ""),
        "spec_no": result.get("spec_no", ""),
        "grade": result.get("grade", ""),
        "composition": result.get("composition", ""),
        "thickness_limits": result.get("thickness_limits", ""),
        "search_method": result.get("search_method", "unknown")
    }


def main():
    """Main function to test the RAG system"""
    # OpenAI API key will be loaded from Streamlit secrets or environment
    openai_key = get_secret('OPENAI_API_KEY')
    if openai_key and openai_key != 'your_openai_api_key_here':
        os.environ["OPENAI_API_KEY"] = openai_key
        print(f"Using OpenAI API key: {openai_key[:10]}...")
    else:
        print("No valid OpenAI API key found. Using text-based search only.")
    
    print("ASME Material P & G Number Lookup System")
    print("=" * 50)
    
    # Test materials
    test_materials = [
        "ASTM A 333 GR. 6"
    ]
    
    for i, material in enumerate(test_materials):
        print(f"\nMaterial: {material}")
        try:
            result = get_p_and_g_numbers(material)
            if result.get("error"):
                print(f"  Error: {result['error']}")
            elif result.get("p_no"):
                print(f"  P-No: {result['p_no']}")
                print(f"  G-No: {result['g_no']}")
                print(f"  Spec: {result['spec_no']}")
                print(f"  Grade: {result['grade']}")
                print(f"  Composition: {result['composition']}")
            else:
                print(f"  Not found or error in lookup")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n{'='*50}")
    print("Example usage:")
    print("result = get_p_and_g_numbers('SA 516 GR. 70')")
    print(f"# Returns: {{'p_no': '1', 'g_no': '2', ...}}")


# NEW WELDING PROCESS RECOMMENDATION SYSTEM
# Note: The welding recommendation system is now handled by welding_api.py
# This section is kept for reference but the actual implementation has moved

def main_welding_demo():
    """Demo function for the new welding process recommendation system"""
    print("WELDING PROCESS RECOMMENDATION SYSTEM DEMO")
    print("=" * 60)
    print("Note: The welding recommendation system is now available through welding_api.py")
    print("Please use the Streamlit app or import from welding_api.py directly")

# Joint types available in the system
AVAILABLE_JOINT_TYPES = [
    "full penetration butt weld",
    "full penetration", 
    "fillet weld",
    "partial penetration",
    "corner joint",
    "lap joint"
]

def list_available_joint_types():
    """List all available joint types"""
    print("Available Joint Types:")
    for i, joint_type in enumerate(AVAILABLE_JOINT_TYPES, 1):
        print(f"  {i}. {joint_type}")
    return AVAILABLE_JOINT_TYPES

# Example usage function that can be called from other modules
def example_welding_recommendation():
    """Example of how to use the welding recommendation system"""
    print("Example: Welding SA 516 GR. 70 to SA 516 GR. 70")
    print("Joint: Full penetration butt weld, Thickness: 20mm, PWHT: No")
    print("Note: Use welding_api.py for actual recommendations")
    
    return {"note": "Use welding_api.py for actual recommendations"}

# Fallback function
def get_welding_recommendation(*args, **kwargs):
    return {"error": "Please use welding_api.py for welding recommendations"}

# Add to main function for testing
if __name__ == "__main__":
    main()  # Original ASME material lookup demo
    
    print("\n" + "="*80)
    print("WELDING PROCESS RECOMMENDATION DEMO")
    print("="*80)
    
    main_welding_demo()  # New welding recommendation demo
    
    print("\n" + "="*80)
    print("AVAILABLE JOINT TYPES")
    print("="*80)
    list_available_joint_types()