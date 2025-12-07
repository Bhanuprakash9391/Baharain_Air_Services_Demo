import streamlit as st
import base64
import time
import fitz  # PyMuPDF
import pickle
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional


import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import io

from openai import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from typing import List, Optional, Dict, Any
import requests
import json
from dotenv import load_dotenv

# import re
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# import json

# import nltk
# nltk.download("punkt")
# nltk.download("stopwords")

# # Initialize NLTK (add this to main function or initialization)
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('averaged_perceptron_tagger', quiet=True)
# except:
#     pass



st.set_page_config(
    page_title="BAS App",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# DeepSeek configuration
DEEPSEEK_CONFIG = {
    "DEEPSEEK_API_KEY": "sk-59bc78c167244d94bc105dfd72e32d59",
    "DEEPSEEK_API_URL": "https://api.deepseek.com/v1/chat/completions"
    
}

class DualDatabaseRetriever:
    """Enhanced retriever that searches both visual and text databases intelligently"""
    
    def __init__(self, visual_vectorstore, text_vectorstore, visual_weight: float = 0.7):
        self.visual_vectorstore = visual_vectorstore
        self.text_vectorstore = text_vectorstore
        self.visual_weight = visual_weight
    
    def get_relevant_documents(self, query: str, k: int = 15) -> List[Document]:
        """Retrieve documents from both databases based on query intent"""
        intent_type, is_visual_question = detect_question_intent(query)
        
        documents = []
        
        # Determine search strategy based on intent
        if is_visual_question and self.visual_vectorstore:
            # Visual question: prioritize visual database
            visual_k = max(1, int(k * self.visual_weight))
            text_k = k - visual_k
            
            # Search visual database first
            try:
                visual_docs = self.visual_vectorstore.similarity_search(query, k=visual_k)
                documents.extend(visual_docs)
                print(f"Retrieved {len(visual_docs)} documents from visual database")
            except Exception as e:
                print(f"Error searching visual database: {e}")
            
            # Search text database for additional context
            if self.text_vectorstore and text_k > 0:
                try:
                    text_docs = self.text_vectorstore.similarity_search(query, k=text_k)
                    documents.extend(text_docs)
                    print(f"Retrieved {len(text_docs)} documents from text database")
                except Exception as e:
                    print(f"Error searching text database: {e}")
                    
        else:
            # Text question: prioritize text database
            text_k = max(1, int(k * 0.8))
            visual_k = k - text_k
            
            # Search text database first
            if self.text_vectorstore:
                try:
                    text_docs = self.text_vectorstore.similarity_search(query, k=text_k)
                    documents.extend(text_docs)
                    print(f"Retrieved {len(text_docs)} documents from text database")
                except Exception as e:
                    print(f"Error searching text database: {e}")
            
            # Search visual database for additional context
            if self.visual_vectorstore and visual_k > 0:
                try:
                    visual_docs = self.visual_vectorstore.similarity_search(query, k=visual_k)
                    documents.extend(visual_docs)
                    print(f"Retrieved {len(visual_docs)} documents from visual database")
                except Exception as e:
                    print(f"Error searching visual database: {e}")
        
        # Add metadata about search strategy
        for doc in documents:
            doc.metadata['search_intent'] = intent_type
            doc.metadata['is_visual_query'] = is_visual_question
        
        return documents[:k]  # Ensure we don't exceed k documents

# Custom DeepSeek Reranker implementation
class EnhancedDeepSeekReranker(BaseDocumentCompressor):
    """Enhanced reranker that considers visual content relevance."""
    
    def __init__(self, top_n: int = 3):
        self.top_n = top_n
    
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Enhanced reranking that considers visual content."""
        if not documents or not query:
            return documents[:self.top_n]
        
        print(f"Enhanced DeepSeek reranking {len(documents)} documents...")
        
        try:
            # Detect if query is asking about visual content
            visual_keywords = [
                "diagram", "chart", "graph", "image", "figure", "table", 
                "illustration", "picture", "visual", "flowchart", "schema",
                "plot", "drawing", "photo", "map", "layout"
            ]
            is_visual_query = any(keyword in query.lower() for keyword in visual_keywords)
            
            # Create enhanced prompt for relevance scoring
            relevance_prompt = f"""
            You are evaluating document relevance for this question: {query}
            
            {"This query appears to be asking about visual content (diagrams, charts, images, etc.). Give higher scores to documents that contain visual descriptions." if is_visual_query else "This query is asking about textual content."}
            
            Documents to evaluate:
            {self._format_documents_for_evaluation(documents)}

            Evaluate each document's relevance on a scale of 0.0 to 1.0:
            - 1.0: Perfectly relevant, directly answers the question
            - 0.7-0.9: Highly relevant with good supporting information
            - 0.4-0.6: Somewhat relevant, contains related information
            - 0.0-0.3: Low relevance or unrelated to the question
            
            {"For visual queries: Prioritize documents marked as having visual content or containing descriptions of diagrams, charts, images, etc." if is_visual_query else ""}

            Return JSON format: {{"0": score, "1": score, "2": score, ...}}
            """
            
            response = call_deepseek_api(relevance_prompt, max_tokens=2000)
            
            if response:
                relevance_scores = self._parse_relevance_scores(response, len(documents))
                
                # Apply visual content boost if it's a visual query
                if is_visual_query:
                    for i, doc in enumerate(documents):
                        if doc.metadata.get('has_visuals', False):
                            # Boost visual content documents for visual queries
                            current_score = relevance_scores.get(str(i), 0.0)
                            relevance_scores[str(i)] = min(1.0, current_score + 0.1)
                
                # Add scores to documents
                for i, doc in enumerate(documents):
                    doc.metadata['relevance_score'] = relevance_scores.get(str(i), 0.0)
                    doc.metadata['original_score'] = i
                
                sorted_docs = sorted(documents, key=lambda x: x.metadata.get('relevance_score', 0.0), reverse=True)
                
                print(f"Enhanced reranking completed. Visual query: {is_visual_query}")
                print(f"Top scores: {[doc.metadata.get('relevance_score', 0.0) for doc in sorted_docs[:3]]}")
                
                return sorted_docs[:self.top_n]
            
        except Exception as e:
            print(f"Enhanced reranking failed: {e}")
        
        return documents[:self.top_n]
    
    
    def _format_documents_for_evaluation(self, documents: List[Document]) -> str:
        """Format documents for the evaluation prompt."""
        formatted = []
        for i, doc in enumerate(documents):
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            formatted.append(f"Document {i}:\n{content_preview}\n")
        return "\n".join(formatted)
    
    def _parse_relevance_scores(self, response: str, num_documents: int) -> Dict[str, float]:
        """Parse relevance scores from DeepSeek response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                # Validate and normalize scores
                validated_scores = {}
                for key, value in scores.items():
                    if key.isdigit() and int(key) < num_documents:
                        validated_scores[key] = max(0.0, min(1.0, float(value)))
                return validated_scores
        except Exception as e:
            print(f"Failed to parse relevance scores: {e}")
        
        # Fallback: return equal scores
        return {str(i): 0.5 for i in range(num_documents)}
    
    async def acompress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Async version of compress_documents (not implemented)."""
        return self.compress_documents(documents, query)

# Reranker availability flag
DEEPSEEK_RERANKER_AVAILABLE = True

load_dotenv()  # take variables from .env

AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
}

# Directory for storing vector databases and RAGAS cache
VECTOR_DB_DIR = "vector_databases"
METADATA_FILE = "db_metadata.json"
RAGAS_CACHE_DIR = "ragas_cache"
VISION_CACHE_DIR = "vision_cache"

# CSS Styling (kept same as original)
st.markdown("""
<style>
.header-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.main-title {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}
.subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
}
.welcome-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}
.welcome-title {
    color: #2c3e50;
    margin-bottom: 1rem;
}
.welcome-text {
    color: #34495e;
    line-height: 1.6;
}
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.feature-item {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.feature-title {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}
.feature-desc {
    color: #7f8c8d;
    font-size: 0.9rem;
}
.stats-container {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.stat-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    flex: 1;
}
.stat-number {
    font-size: 2rem;
    font-weight: bold;
}
.stat-label {
    font-size: 0.8rem;
    opacity: 0.9;
}
.sidebar-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.section-title {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #667eea;
}
.main-content {
    min-height: 400px;
}
.cost-savings {
    background: #e8f5e8;
    border: 1px solid #4caf50;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)



def ensure_directories():
    """Ensure necessary directories exist"""
    Path(VECTOR_DB_DIR).mkdir(exist_ok=True)
    Path(RAGAS_CACHE_DIR).mkdir(exist_ok=True)
    Path(VISION_CACHE_DIR).mkdir(exist_ok=True)


def validate_azure_config():
    """Validate Azure configuration"""
    required_fields = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
    ]
    
    for field in required_fields:
        if not AZURE_CONFIG.get(field):
            st.error(f"Missing Azure configuration: {field}")
            return False
    return True

def categorize_visual_content(enhanced_description: str, basic_text: str) -> str:
    """Categorize the type of visual content based on description and text"""
    description_lower = enhanced_description.lower()
    text_lower = basic_text.lower()
    combined = f"{description_lower} {text_lower}"
    
    # Define categories with keywords
    categories = {
        'diagram': ['diagram', 'flowchart', 'schematic', 'workflow', 'process flow', 'architecture', 'system diagram'],
        'chart_graph': ['chart', 'graph', 'plot', 'histogram', 'bar chart', 'line graph', 'pie chart', 'scatter plot'],
        'table': ['table', 'rows', 'columns', 'tabular', 'matrix', 'grid'],
        'photograph': ['photograph', 'photo', 'picture', 'image of people', 'portrait', 'landscape'],
        'technical_drawing': ['blueprint', 'engineering drawing', 'cad', 'technical drawing', 'specifications'],
        'map': ['map', 'geographical', 'location', 'coordinates', 'regions'],
        'mathematical': ['equation', 'formula', 'mathematical', 'calculation', 'theorem'],
        'infographic': ['infographic', 'visual representation', 'data visualization', 'statistics']
    }
    
    # Score each category
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in combined)
        if score > 0:
            category_scores[category] = score
    
    # Return the highest scoring category, or 'mixed' if tie, or 'general' if none
    if not category_scores:
        return 'general'
    
    max_score = max(category_scores.values())
    top_categories = [cat for cat, score in category_scores.items() if score == max_score]
    
    if len(top_categories) == 1:
        return top_categories[0]
    else:
        return 'mixed'



def detect_question_intent(question: str) -> Tuple[str, bool]:
    """
    Detect if question is asking about visual content and return intent type
    Returns: (intent_type, is_visual_question)
    """
    question_lower = question.lower()
    
    # Visual question indicators
    visual_keywords = [
        'diagram', 'chart', 'graph', 'image', 'figure', 'table', 'picture',
        'illustration', 'flowchart', 'plot', 'drawing', 'photo', 'map',
        'visual', 'show', 'display', 'layout', 'structure', 'design'
    ]
    
    is_visual = any(keyword in question_lower for keyword in visual_keywords)
    
    # Determine intent type
    if any(word in question_lower for word in ['what', 'define', 'definition', 'meaning']):
        intent_type = 'definition'
    elif any(word in question_lower for word in ['how', 'process', 'step', 'procedure']):
        intent_type = 'process'
    elif any(word in question_lower for word in ['why', 'reason', 'cause']):
        intent_type = 'explanation'
    elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
        intent_type = 'comparison'
    elif any(word in question_lower for word in ['list', 'types', 'categories']):
        intent_type = 'categorical'
    elif any(word in question_lower for word in ['number', 'amount', 'how many', 'quantity']):
        intent_type = 'quantitative'
    elif any(word in question_lower for word in ['example', 'instance', 'case']):
        intent_type = 'examples'
    elif is_visual:
        intent_type = 'visual_content'
    else:
        intent_type = 'general'
    
    return intent_type, is_visual

def get_cache_key(question: str, answer: str, contexts: List[str]) -> str:
    """Generate a cache key for RAGAS evaluation"""
    content = f"{question}|{answer}|{'|'.join(contexts)}"
    return hashlib.md5(content.encode()).hexdigest()

def get_page_hash(page):
    """Create hash of page content for vision caching"""
    try:
        pix = page.get_pixmap(dpi=72)  # Low res for hashing
        return hashlib.md5(pix.tobytes()).hexdigest()
    except:
        # Fallback to text-based hash
        text = page.get_text()
        return hashlib.md5(text.encode()).hexdigest()

def setup_processing_log():
    """Setup simple log file for processing tracking."""
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/processing_{timestamp}.log"
    
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='w'
    )
    
    logging.info("PDF Processing Started")
    return log_filename

def load_ragas_cache(cache_key: str) -> Optional[Dict]:
    """Load RAGAS metrics from cache"""
    cache_file = os.path.join(RAGAS_CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load RAGAS cache: {e}")
    return None

def load_vision_cache(page_hash: str) -> Optional[Dict]:
    """Load vision analysis from cache"""
    cache_file = os.path.join(VISION_CACHE_DIR, f"{page_hash}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load vision cache: {e}")
    return None

def save_vision_cache(page_hash: str, result: Dict):
    """Save vision analysis to cache"""
    ensure_directories()
    cache_file = os.path.join(VISION_CACHE_DIR, f"{page_hash}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Failed to save vision cache: {e}")

# Smart Visual Content Detection Functions
def detect_images_in_pdf_page(page):
    """Detect images using PyMuPDF's built-in capabilities"""
    try:
        image_list = page.get_images()
        drawings = page.get_drawings()  # Vector graphics, shapes
        
        return {
            'has_images': len(image_list) > 0,
            'has_drawings': len(drawings) > 0,
            'image_count': len(image_list),
            'drawing_count': len(drawings),
            'total_visual_elements': len(image_list) + len(drawings)
        }
    except Exception as e:
        print(f"Error detecting images: {e}")
        return {
            'has_images': False,
            'has_drawings': False,
            'image_count': 0,
            'drawing_count': 0,
            'total_visual_elements': 0
        }

def analyze_text_density(page):
    """Determine if page is text-heavy or visual-heavy"""
    try:
        text = page.get_text()
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        # Calculate text coverage using text blocks
        text_blocks = page.get_text("dict")["blocks"]
        text_area = 0
        text_block_count = 0
        
        for block in text_blocks:
            if "lines" in block:  # Text block
                text_block_count += 1
                bbox = block["bbox"]
                text_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        text_coverage_ratio = text_area / page_area if page_area > 0 else 0
        
        return {
            'text_length': len(text),
            'text_coverage_ratio': text_coverage_ratio,
            'text_block_count': text_block_count,
            'likely_visual': text_coverage_ratio < 0.3 or len(text) < 100,
            'very_sparse_text': len(text) < 50
        }
    except Exception as e:
        print(f"Error analyzing text density: {e}")
        text = page.get_text()
        return {
            'text_length': len(text),
            'text_coverage_ratio': 0.5,
            'text_block_count': 1,
            'likely_visual': len(text) < 100,
            'very_sparse_text': len(text) < 50
        }

def contains_table_indicators(text: str) -> bool:
    """
    A more robust heuristic to detect table-like structures in text.
    It checks for multiple indicators and returns True if at least two are found.
    """
    # 1. Guard Clause: Ignore very short or effectively empty text.
    if not text or len(text.strip()) < 20:
        return False

    try:
        lines = text.split('\n')

        # --- Indicator 1: Presence of common table delimiters ---
        has_tabs_or_pipes = '\t' in text or '|' in text

        # --- Indicator 2: High ratio of spaces, suggesting columnar layout ---
        # A high number of spaces can indicate columns separated by whitespace.
        high_space_ratio = text.count(' ') / len(text) > 0.4

        # --- Indicator 3 (Corrected): Consistent number of columns across multiple lines ---
        non_empty_lines = [line.split() for line in lines if line.strip()]
        has_consistent_cols = False
        # Check only if there are enough lines to establish a pattern.
        if len(non_empty_lines) > 2:
            # Use the first non-empty line as a reference for column count.
            # A meaningful table must have more than one column.
            ref_col_count = len(non_empty_lines[0])
            if ref_col_count > 1:
                # Count how many lines have the same number of columns as the reference.
                matching_lines_count = sum(1 for parts in non_empty_lines if len(parts) == ref_col_count)
                # If at least 3 lines (or more than half) match, it's a strong signal.
                if matching_lines_count >= 3 and matching_lines_count > len(non_empty_lines) / 2:
                    has_consistent_cols = True

        # --- Indicator 4: Multiple lines with a significant number of "columns" ---
        # This catches tables that may not have perfectly consistent column counts.
        lines_with_many_cols = sum(1 for line in lines if len(line.split()) >= 4)
        has_multiple_columnar_lines = lines_with_many_cols > 2
        
        # --- Indicator 5: Presence of financial or numeric symbols ---
        # Often indicates data tables.
        first_few_lines = "".join(lines[:5])
        has_financial_symbols = any(char in first_few_lines for char in '$%€')

        # --- Final Decision ---
        # Tally the boolean indicators. True counts as 1, False as 0.
        indicators_found = sum([
            has_tabs_or_pipes,
            high_space_ratio,
            has_consistent_cols,
            has_multiple_columnar_lines,
            has_financial_symbols
        ])
        
        # Require at least two indicators to be confident it's a table.
        return indicators_found >= 2

    except Exception as e:
        print(f"Error detecting table indicators: {e}")
        return False


def should_use_vision_ai(page, file_name: str, page_num: int):
    """
    Decide whether to use expensive vision AI based on quick checks,
    now with a threshold for vector graphics to avoid false positives from simple lines.
    """
    # --- TUNABLE PARAMETER ---
    # Adjust this value based on your documents. Higher values are stricter.
    VECTOR_GRAPHICS_THRESHOLD = 300

    try:
        # 1. Perform quick, inexpensive checks on the page
        basic_text = page.get_text().strip()
        image_info = detect_images_in_pdf_page(page)
        density_info = analyze_text_density(page)

        # 2. Define the set of conditions that would require Vision AI
        conditions = {
            # Condition 1: Are there actual embedded images (like photos)?
            'has_embedded_images': image_info['has_images'],

            # Condition 2 (MODIFIED): Is there a *significant* number of vector drawings?
            # This now ignores pages with just a few lines (e.g., underlines).
            'has_significant_vector_graphics': image_info['drawing_count'] > VECTOR_GRAPHICS_THRESHOLD,

            # Condition 3: Is the page sparsely populated with text (like a title page or diagram)?
            'low_text_density': density_info['likely_visual'],

            # Condition 4: Is there almost no text on the page?
            'very_little_text': len(basic_text) < 50,

            # Condition 5: Does the text structure strongly suggest a table?
            'contains_tables': contains_table_indicators(basic_text)
        }

        # 3. Make the final decision: if any condition is met, use Vision AI
        use_vision = any(conditions.values())

        # 4. Prepare a detailed log of the decision for debugging purposes
        analysis_info = {
            'conditions': conditions,
            'image_count': image_info['image_count'],
            'drawing_count': image_info['drawing_count'],
            'text_length': len(basic_text),
            'text_coverage': density_info['text_coverage_ratio'],
            'reasoning': []
        }

        # Build the human-readable reasoning string
        if conditions['has_embedded_images']:
            analysis_info['reasoning'].append(f"Found {image_info['image_count']} embedded images")
        if conditions['has_significant_vector_graphics']:
            analysis_info['reasoning'].append(f"Found {image_info['drawing_count']} vector graphics (above threshold of {VECTOR_GRAPHICS_THRESHOLD})")
        if conditions['low_text_density']:
            analysis_info['reasoning'].append(f"Low text coverage ratio: {density_info['text_coverage_ratio']:.2f}")
        if conditions['very_little_text']:
            analysis_info['reasoning'].append(f"Very sparse text: {len(basic_text)} characters")
        if conditions['contains_tables']:
            analysis_info['reasoning'].append("Likely contains tables")

        if not use_vision:
            reasoning_text = "Sufficient extractable text"
            if image_info['drawing_count'] > 0:
                reasoning_text += f", and vector graphics count ({image_info['drawing_count']}) is below threshold ({VECTOR_GRAPHICS_THRESHOLD})"
            analysis_info['reasoning'].append(reasoning_text)

        return use_vision, analysis_info

    except Exception as e:
        print(f"Error in should_use_vision_ai: {e}")
        # Conservative fallback: use vision AI if analysis fails
        return True, {'error': str(e), 'reasoning': ['Error in analysis, using vision AI as fallback']}

def get_metric_color(score: float) -> str:
    """Get color based on metric score"""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "orange"
    else:
        return "red"

def call_deepseek_api(prompt: str, max_tokens: int = 1000) -> str:
    """Call DeepSeek API with the given prompt"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_CONFIG['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        response = requests.post(
            DEEPSEEK_CONFIG["DEEPSEEK_API_URL"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"DeepSeek API error: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        return ""

def evaluate_with_deepseek(question: str, answer: str, contexts: List[str]) -> Dict:
    """Evaluate response using DeepSeek LLM for faithfulness and relevance."""
    ensure_directories()
    
    cache_key = get_cache_key(question, answer, contexts)
    cached_metrics = load_ragas_cache(cache_key)
    if cached_metrics:
        print(f"Using cached metrics: {cached_metrics}")
        return cached_metrics
    
    print("Starting DeepSeek evaluation...")
    
    try:
        # Prepare context for evaluation
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)]) if contexts else "No context available"
        
        # Faithfulness evaluation prompt
        faithfulness_prompt = f"""
        Evaluate the faithfulness of the answer to the provided context. Faithfulness measures whether the answer is factually consistent with the context.

        Question: {question}
        Answer: {answer}
        Context: {context_text}

        Please analyze and provide a score between 0.0 and 1.0 where:
        - 1.0: The answer is completely faithful to the context (all facts are supported)
        - 0.0: The answer contains information not present in the context or contradicts the context

        Provide your response in JSON format with a single "faithfulness_score" field containing the numeric score.
        """
        
        # Answer relevance evaluation prompt
        relevance_prompt = f"""
        Evaluate the relevance of the answer to the question. Relevance measures how well the answer addresses the specific question.

        Question: {question}
        Answer: {answer}
        Context: {context_text}

        Please analyze and provide a score between 0.0 and 1.0 where:
        - 1.0: The answer perfectly addresses the question and is highly relevant
        - 0.0: The answer is completely irrelevant to the question

        Provide your response in JSON format with a single "relevance_score" field containing the numeric score.
        """
        
        # Get faithfulness score
        faithfulness_response = call_deepseek_api(faithfulness_prompt)
        faithfulness_score = 0.0
        if faithfulness_response:
            try:
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{.*\}', faithfulness_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    faithfulness_score = float(result.get("faithfulness_score", 0.0))
                else:
                    # Fallback: look for numeric score in text
                    score_match = re.search(r'(\d+\.\d+)', faithfulness_response)
                    if score_match:
                        faithfulness_score = float(score_match.group(1))
            except:
                faithfulness_score = 0.0
        
        # Get relevance score
        relevance_response = call_deepseek_api(relevance_prompt)
        relevance_score = 0.0
        if relevance_response:
            try:
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    relevance_score = float(result.get("relevance_score", 0.0))
                else:
                    # Fallback: look for numeric score in text
                    score_match = re.search(r'(\d+\.\d+)', relevance_response)
                    if score_match:
                        relevance_score = float(score_match.group(1))
            except:
                relevance_score = 0.0
        
        metrics = {
            "faithfulness": max(0.0, min(1.0, faithfulness_score)),
            "answer_relevancy": max(0.0, min(1.0, relevance_score)),
        }
        
        print(f"FINAL DEEPSEEK METRICS: {metrics}")
        
        return metrics
        
    except Exception as e:
        error_msg = f"DeepSeek evaluation failed: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

def save_vector_db_metadata(db_name: str, file_names: List[str], stats: Dict):
    """Save metadata about the vector database"""
    ensure_directories()
    metadata_path = os.path.join(VECTOR_DB_DIR, METADATA_FILE)
    
    # Load existing metadata
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Failed to load metadata: {e}")
            metadata = {}
    
    # Add new entry
    metadata[db_name] = {
        "created_at": datetime.now().isoformat(),
        "file_names": file_names,
        "stats": stats,
        "total_files": len(file_names)
    }
    
    # Save updated metadata
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Failed to save metadata: {e}")

def load_vector_db_metadata() -> Dict:
    """Load vector database metadata"""
    metadata_path = os.path.join(VECTOR_DB_DIR, METADATA_FILE)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load metadata: {e}")
    return {}

def save_vector_db(vectorstore, db_name: str, document_sources: Dict, uploaded_files_data: Dict = None):
    """Save vector database with proper naming"""
    ensure_directories()
    
    db_path = os.path.join(VECTOR_DB_DIR, db_name)
    sources_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_sources.pkl")
    files_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_files.pkl")
    
    try:
        # Save vectorstore
        if vectorstore:
            vectorstore.save_local(db_path)
        
        # Save document sources mapping
        with open(sources_path, 'wb') as f:
            pickle.dump(document_sources, f)
        
        # Save uploaded files data for later retrieval
        if uploaded_files_data:
            with open(files_path, 'wb') as f:
                pickle.dump(uploaded_files_data, f)
                
        return True
    except Exception as e:
        st.error(f"Failed to save vector database: {e}")
        return False

def load_vector_db(db_name: str, embeddings) -> Tuple[Optional[FAISS], Optional[Dict], Optional[Dict]]:
    """Load vector database"""
    db_path = os.path.join(VECTOR_DB_DIR, db_name)
    sources_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_sources.pkl")
    files_path = os.path.join(VECTOR_DB_DIR, f"{db_name}_files.pkl")
    
    try:
        # Load vectorstore
        vectorstore = None
        if os.path.exists(db_path):
            vectorstore = FAISS.load_local(
                db_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        
        # Load document sources
        document_sources = {}
        if os.path.exists(sources_path):
            with open(sources_path, 'rb') as f:
                document_sources = pickle.load(f)
        
        # Load uploaded files data
        uploaded_files_data = {}
        if os.path.exists(files_path):
            with open(files_path, 'rb') as f:
                uploaded_files_data = pickle.load(f)
        
        return vectorstore, document_sources, uploaded_files_data
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None, None, None

def display_pdf_viewer(file_name: str, start_page: int, response_index: int):
    """PDF viewer with unique keys and navigation controls below the image."""
    if file_name not in st.session_state.get('uploaded_file_bytes', {}):
        st.error(f"File '{file_name}' not found in current session.")
        return
    
    file_bytes = st.session_state.uploaded_file_bytes[file_name]
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        
        # MODIFIED: Create a unique key for the page state
        page_state_key = f"current_page_{file_name}_{response_index}"

        st.markdown(f"**{file_name}** - Page {start_page} of {total_pages}")
        
        # MODIFIED: Display the PDF image first
        page = doc[start_page - 1]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, use_container_width=True)
        
        doc.close()

        # MODIFIED: Place navigation controls and columns below the image
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("◀ Previous", disabled=(start_page <= 1), key=f"prev_{file_name}_{response_index}"):
                st.session_state[page_state_key] = max(1, start_page - 1)
                st.rerun()
        
        with col2:
            if st.button("Next ▶", disabled=(start_page >= total_pages), key=f"next_{file_name}_{response_index}"):
                st.session_state[page_state_key] = min(total_pages, start_page + 1)
                st.rerun()
        
        with col3:
            target_page = st.number_input(
                f"Go to page (1-{total_pages})", 
                min_value=1, max_value=total_pages, value=start_page,
                key=f"page_input_{file_name}_{response_index}"
            )
            # Add a button to jump, or make it react on change
            if st.button("Jump", key=f"jump_{file_name}_{response_index}"):
                if target_page != start_page:
                    st.session_state[page_state_key] = target_page
                    st.rerun()

    except Exception as e:
        st.error(f"Error displaying PDF: {e}")


def display_header():
    """Display professional header"""
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Optimized PDF Document Assistant</h1>
        <p class="subtitle">Smart Vision AI usage with 60-80% cost reduction through intelligent pre-filtering</p>
    </div>
    """, unsafe_allow_html=True)

def display_welcome():
    """Display welcome section with optimization info"""
    st.markdown("""
    <div class="welcome-section">
        <h2 class="welcome-title">Welcome to Optimized PDF Document Assistant</h2>
        <p class="welcome-text">
            Upload your PDF documents and images to start analyzing and asking questions about their content. 
            The system now uses smart pre-filtering to reduce Vision AI costs by 60-80% while maintaining accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="features-grid">
        <div class="feature-item">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Vision Detection</div>
            <div class="feature-desc">Only uses expensive Vision AI when actually needed</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Cost Optimized</div>
            <div class="feature-desc">60-80% reduction in Vision API costs</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Intelligent Caching</div>
            <div class="feature-desc">Caches vision results to avoid reprocessing</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Processing Analytics</div>
            <div class="feature-desc">Detailed breakdown of processing decisions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_enhanced_source_info(source, response_index: int, source_index: int):
    """Enhanced source display with enriched metadata"""
    file_name = source.get('file_name', 'Unknown')
    page_num = source.get('page', 1)
    source_type = source.get('type', 'pdf')
    rank = source.get('rank', source_index + 1)
    relevance_score = source.get('relevance_score', None)
    has_visuals = source.get('has_visuals', False)
    content = source.get('content', '')
    processing_method = source.get('processing_method', 'unknown')
    
    # New enriched fields
    visual_category = source.get('visual_category', None)
    keywords = source.get('keywords', [])
    summary = source.get('summary', '')
    question_types = source.get('question_types', [])
    search_intent = source.get('search_intent', 'general')
    
    # Enhanced metadata display
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        visual_indicator = "🖼️" if has_visuals else "📄"
        method_indicator = "🤖" if processing_method == "vision" else "💾" if processing_method == "cached" else "📝"
        category_indicator = f" ({visual_category})" if visual_category and visual_category != 'general' else ""
        st.markdown(f"**{visual_indicator}{method_indicator} File:** {file_name}{category_indicator}")
    with col2:
        st.markdown(f"**Page:** {page_num}")
    with col3:
        if relevance_score is not None and relevance_score > 0:
            color = "green" if relevance_score > 0.7 else "orange" if relevance_score > 0.4 else "red"
            st.markdown(f"**Score:** <span style='color: {color}'>{relevance_score:.2f}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Rank:** #{rank}")
    with col4:
        intent_emoji = "🎨" if search_intent == "visual_content" else "📝"
        st.markdown(f"**{intent_emoji} {search_intent.title()}**")
    
    # Enhanced processing info
    if processing_method == "vision":
        if visual_category:
            st.info(f"🤖 Vision AI: Detected {visual_category.replace('_', ' ')} content")
        else:
            st.info("🤖 Processed with Vision AI (detected visual content)")
    elif processing_method == "text":
        st.success("📝 Text extraction used (cost-efficient)")
    elif processing_method == "cached":
        st.info("💾 Used cached Vision AI result (cost-efficient)")
    
    # Display enriched metadata
    if keywords:
        st.markdown(f"**🔑 Keywords:** {', '.join(keywords[:5])}")
    
    if question_types:
        st.markdown(f"**❓ Question Types:** {', '.join(question_types[:3])}")
    
    # Enhanced content preview with summary
    if summary or content:
        with st.expander("Content Preview", expanded=False):
            if summary:
                st.markdown("**📋 Summary:**")
                st.info(summary)
                
            if content:
                st.markdown("**📄 Full Content:**")
                # Split content into parts if it contains enhanced structure
                if "Summary:" in content and "Keywords:" in content:
                    # This is enhanced content, display structured
                    parts = content.split("Content:")
                    if len(parts) > 1:
                        metadata_part = parts[0]
                        actual_content = parts[1]
                        
                        # Show metadata in expandable section
                        with st.expander("Chunk Metadata", expanded=False):
                            st.text(metadata_part)
                        
                        # Show actual content
                        if "Visual Analysis:" in actual_content:
                            text_part, visual_part = actual_content.split("Visual Analysis:", 1)
                            st.markdown("**Text Content:**")
                            st.text(text_part.replace("Text Content:", "").strip())
                            st.markdown("**Visual Analysis:**")
                            st.text(visual_part.strip())
                        else:
                            st.text(actual_content.strip())
                    else:
                        st.text(content)
                else:
                    # Regular content display
                    if "Visual Analysis:" in content:
                        text_part, visual_part = content.split("Visual Analysis:", 1)
                        st.markdown("**Text Content:**")
                        st.text(text_part.replace("Text Content:", "").strip())
                        st.markdown("**Visual Analysis:**")
                        st.text(visual_part.strip())
                    else:
                        st.text(content)
    
    # Display file viewer if available
    if file_name in st.session_state.get('uploaded_file_bytes', {}):
        if source_type == 'pdf':
            # Create unique key for this source's page state
            page_state_key = f"current_page_{file_name}_{response_index}_{source_index}"
            if page_state_key not in st.session_state:
                st.session_state[page_state_key] = page_num
            
            current_page = st.session_state[page_state_key]
            
            # Create a button to view the PDF
            if st.button(f"View PDF", key=f"view_pdf_{response_index}_{source_index}"):
                st.session_state[f"show_pdf_{response_index}_{source_index}"] = True
            
            # Show PDF viewer if requested
            if st.session_state.get(f"show_pdf_{response_index}_{source_index}", False):
                display_pdf_viewer_for_source(file_name, current_page, response_index, source_index)
                
                # Add close button
                if st.button("Close PDF Viewer", key=f"close_pdf_{response_index}_{source_index}"):
                    st.session_state[f"show_pdf_{response_index}_{source_index}"] = False
                    st.rerun()
                    
        elif source_type == 'image':
            # Create a button to view the image
            if st.button(f"View Image", key=f"view_img_{response_index}_{source_index}"):
                st.session_state[f"show_img_{response_index}_{source_index}"] = True
            
            # Show image viewer if requested
            if st.session_state.get(f"show_img_{response_index}_{source_index}", False):
                file_bytes = st.session_state.uploaded_file_bytes[file_name]
                st.image(file_bytes, caption=f"Source Image: {file_name}", use_container_width=True)
                
                # Add close button
                if st.button("Close Image Viewer", key=f"close_img_{response_index}_{source_index}"):
                    st.session_state[f"show_img_{response_index}_{source_index}"] = False
                    st.rerun()
    else:
        st.warning(f"File '{file_name}' is not available for viewing in this session.")

def display_processing_analytics(stats):
    """Display detailed processing analytics with cost savings"""
    total_pages = stats.get('total_pages', 0)
    vision_pages = stats.get('vision_pages', 0)
    text_pages = stats.get('text_pages', 0)
    cached_pages = stats.get('cached_pages', 0)
    
    if total_pages > 0:
        vision_percentage = (vision_pages / total_pages) * 100
        cost_savings = ((total_pages - vision_pages) / total_pages) * 100 if total_pages > 0 else 0
        
        st.markdown(f"""
        <div class="cost-savings">
            <h4>💰 Cost Optimization Results</h4>
            <p><strong>Vision AI Usage:</strong> {vision_pages}/{total_pages} pages ({vision_percentage:.1f}%)</p>
            <p><strong>Cost Savings:</strong> ~{cost_savings:.1f}% reduction in Vision API calls</p>
            <p><strong>Cached Results:</strong> {cached_pages} pages reused from cache</p>
        </div>
        """, unsafe_allow_html=True)

def display_enhanced_stats(stats):
    """Display enhanced processing statistics with dual database metrics."""
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-box">
            <div class="stat-number">{stats.get('text_pages', 0)}</div>
            <div class="stat-label">Text Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats.get('vision_pages', 0)}</div>
            <div class="stat-label">Vision Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats.get('cached_pages', 0)}</div>
            <div class="stat-label">Cached Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats.get('visual_chunks', 0)}</div>
            <div class="stat-label">Visual Chunks</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{stats.get('text_chunks', 0)}</div>
            <div class="stat-label">Text Chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    display_processing_analytics(stats)

def display_reranker_status():
    """Display reranker configuration status in the sidebar."""
    st.markdown("---")
    st.markdown('<div class="section-title">🏆 Reranker Status</div>', unsafe_allow_html=True)
    
    if DEEPSEEK_RERANKER_AVAILABLE:
        st.success("✅ DeepSeek Reranker Available")
        st.info("📊 Reranking active:\n- Retrieves 15 documents\n- Reranks to top N using DeepSeek LLM\n- Shows ranking in results")
    else:
        st.error("❌ DeepSeek Reranker Not Available")
        st.warning("DeepSeek API configuration required")

def get_optimized_vision_analysis(client: AzureOpenAI, page: fitz.Page, page_num: int, file_name: str, use_cache: bool = True) -> dict:
    """Optimized vision analysis with visual categorization"""
    
    # Check cache first if enabled
    if use_cache:
        page_hash = get_page_hash(page)
        cached_result = load_vision_cache(page_hash)
        if cached_result:
            print(f"Using cached vision result for page {page_num}")
            cached_result['processing_method'] = 'cached'
            return cached_result
    
    try:
        pix = page.get_pixmap(dpi=200)
        img_data = pix.tobytes("png")
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        vision_prompt = """
        Analyze this document page comprehensively and categorize the visual content. Identify and describe:
        
        1. **Content Type Classification**: Is this primarily a diagram, chart/graph, table, photograph, technical drawing, map, or mixed content?
        
        2. **Text Content**: Extract all readable text with proper formatting
        
        3. **Visual Elements Analysis**: 
           - For DIAGRAMS: Describe components, connections, flow direction, labels
           - For CHARTS/GRAPHS: Describe data trends, axes, legends, data points, patterns
           - For TABLES: Describe structure, headers, key data relationships
           - For PHOTOGRAPHS: Describe subjects, setting, composition, key objects
           - For TECHNICAL DRAWINGS: Describe specifications, measurements, components
           - For MAPS: Describe regions, landmarks, scales, legends
        
        4. **Key Information Extraction**:
           - Main concepts or data points
           - Relationships between elements
           - Important labels, captions, or annotations
        
        Provide a comprehensive description that captures both textual and visual information, clearly indicating the content type.
        """

        response = client.chat.completions.create(
            model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.0
        )
        
        basic_text = page.get_text().strip()
        enhanced_description = response.choices[0].message.content
        
        # Determine visual category
        visual_category = categorize_visual_content(enhanced_description, basic_text)
        
        result = {
            "basic_text": basic_text,
            "enhanced_description": enhanced_description,
            "visual_category": visual_category,
            "has_visuals": any(keyword in enhanced_description.lower() 
                             for keyword in ["diagram", "chart", "image", "table", "graph", "photo", "drawing"]),
            "processing_method": "vision"
        }
        
        # Save to cache if enabled
        if use_cache:
            page_hash = get_page_hash(page)
            save_vision_cache(page_hash, result)
        
        return result
        
    except Exception as e:
        st.error(f"Error in optimized vision analysis: {e}")
        return {
            "basic_text": page.get_text().strip(),
            "enhanced_description": "",
            "visual_category": "general",
            "has_visuals": False,
            "processing_method": "error"
        }

def get_text_from_image(client: AzureOpenAI, page: fitz.Page, page_num: int, file_name: str) -> str:
    """Extracts text from a PDF page image using Azure's vision model."""
    try:
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        vision_prompt = "Extract all text from this image of a document page. Preserve the original formatting as accurately as possible."
        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"Rate limit hit during OCR. Retrying in {base_delay}s...")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    st.error(f"An unexpected error occurred during OCR: {e}")
                    return ""
        st.error("OCR failed after multiple retries.")
        return ""
    except Exception as e:
        st.error(f"Error in get_text_from_image: {e}")
        return ""

def get_text_from_uploaded_image(client: AzureOpenAI, image_file, file_name: str) -> str:
    """Extracts text from an uploaded image file using Azure's vision model."""
    try:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_file.seek(0)  # Reset file pointer

        vision_prompt = "Extract all text from this image of a document. Preserve the original formatting as accurately as possible."
        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"Rate limit hit during OCR. Retrying in {base_delay}s...")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    st.error(f"An unexpected error occurred during OCR: {e}")
                    return ""
        st.error("OCR failed after multiple retries.")
        return ""
    except Exception as e:
        st.error(f"Error in get_text_from_uploaded_image: {e}")
        return ""

def get_documents_text_with_optimized_vision(uploaded_files: list, client):
    """Updated to handle dual database structure with visual categorization"""
    
    # Setup logging
    log_filename = setup_processing_log()
    start_time = time.time()
    
    documents = []
    processing_stats = {
        "text_pages": 0, 
        "vision_pages": 0, 
        "cached_pages": 0,
        "image_files": 0, 
        "total_pages": 0,
        "cost_savings_percentage": 0,
        "visual_chunks": 0,
        "text_chunks": 0
    }
    document_sources = {}

    logging.info(f"Processing {len(uploaded_files)} files with optimized vision AI")
    
    for file in uploaded_files:
        file_start_time = time.time()
        logging.info(f"Started processing: {file.name}")
        
        try:
            if file.type == "application/pdf":
                file_content = file.read()
                file.seek(0)
                
                doc = fitz.open(stream=file_content, filetype="pdf")
                
                for i, page in enumerate(doc):
                    page_num = i + 1
                    page_start_time = time.time()
                    processing_stats["total_pages"] += 1
                    
                    # Smart decision: should we use vision AI?
                    use_vision, analysis_info = should_use_vision_ai(page, file.name, page_num)
                    
                    logging.info(f"  Page {page_num}: Vision AI decision: {use_vision}")
                    logging.info(f"  Page {page_num}: Reasoning: {', '.join(analysis_info.get('reasoning', []))}")
                    
                    if use_vision:
                        # Use optimized vision analysis with caching
                        vision_result = get_optimized_vision_analysis(client, page, page_num, file.name)
                        
                        if vision_result['processing_method'] == 'cached':
                            processing_stats["cached_pages"] += 1
                            process_type = "cached"
                        else:
                            processing_stats["vision_pages"] += 1
                            process_type = "vision"
                        
                        # Create content from vision analysis
                        basic_text = vision_result["basic_text"]
                        enhanced_description = vision_result["enhanced_description"]
                        has_visuals = vision_result["has_visuals"]
                        visual_category = vision_result.get("visual_category", "general")
                        
                        if enhanced_description:
                            text_content = f"Text Content:\n{basic_text}\n\nVisual Analysis:\n{enhanced_description}"
                        else:
                            text_content = basic_text
                            
                    else:
                        # Use simple text extraction (cost-efficient)
                        text_content = page.get_text().strip()
                        has_visuals = False
                        visual_category = "general"
                        processing_stats["text_pages"] += 1
                        process_type = "text"
                    
                    page_time = time.time() - page_start_time
                    logging.info(f"  Page {page_num}: {process_type} processing - {page_time:.2f}s")
                    
                    if text_content and text_content.strip():
                        document = Document(
                            page_content=text_content,
                            metadata={
                                "source": file.name,
                                "page": page_num,
                                "type": "pdf",
                                "has_visuals": has_visuals,
                                "visual_category": visual_category,
                                "content_type": "visual" if has_visuals else "text",
                                "processing_method": process_type
                            }
                        )
                        documents.append(document)
                        
                        doc_id = len(documents) - 1
                        document_sources[doc_id] = {
                            "file_name": file.name,
                            "page": page_num,
                            "type": "pdf",
                            "has_visuals": has_visuals,
                            "visual_category": visual_category,
                            "processing_method": process_type
                        }
                
                doc.close()
                
            else:
                # Handle image files - always use vision AI
                enhanced_result, visual_category = get_enhanced_image_analysis(client, file, file.name)
                
                if enhanced_result and enhanced_result.strip():
                    processing_stats["image_files"] += 1
                    
                    document = Document(
                        page_content=enhanced_result,
                        metadata={
                            "source": file.name,
                            "page": 1,
                            "type": "image",
                            "has_visuals": True,
                            "visual_category": visual_category,
                            "content_type": "visual",
                            "processing_method": "vision"
                        }
                    )
                    documents.append(document)
                    
                    doc_id = len(documents) - 1
                    document_sources[doc_id] = {
                        "file_name": file.name,
                        "page": 1,
                        "type": "image",
                        "has_visuals": True,
                        "visual_category": visual_category,
                        "processing_method": "vision"
                    }
            
            file_time = time.time() - file_start_time
            logging.info(f"Completed: {file.name} - {file_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Error processing {file.name}: {str(e)}")
            st.error(f"Error processing {file.name}: {e}")
    
    # Calculate cost savings
    total_pages = processing_stats["total_pages"]
    if total_pages > 0:
        vision_pages = processing_stats["vision_pages"]
        cost_savings = ((total_pages - vision_pages) / total_pages) * 100
        processing_stats["cost_savings_percentage"] = cost_savings
    
    # Log completion with optimization results
    total_time = time.time() - start_time
    logging.info(f"Processing completed - Total time: {total_time:.2f}s")
    logging.info(f"Files: {len(uploaded_files)}, Documents: {len(documents)}")
    logging.info(f"Optimization: {processing_stats['vision_pages']}/{total_pages} pages used Vision AI")
    logging.info(f"Cost savings: ~{processing_stats['cost_savings_percentage']:.1f}%")
    logging.info(f"Stats: {processing_stats}")
    
    st.success(f"Processing completed! Vision AI used on {processing_stats['vision_pages']}/{total_pages} pages ({processing_stats.get('cost_savings_percentage', 0):.1f}% cost savings)")
    st.info(f"Processing log saved: {log_filename}")
    
    return documents, processing_stats, document_sources

def get_enhanced_image_analysis(client: AzureOpenAI, image_file, file_name: str) -> tuple:
    """
    Enhanced analysis for uploaded images.
    Returns: (enhanced_description, visual_category)
    """
    try:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_file.seek(0)

        vision_prompt = """
        Provide a comprehensive analysis of this image:
        
        1. **Content Type**: Is this a diagram, chart, table, photograph, technical drawing, or other?
        2. **Content Description**: What is shown in the image?
        3. **Technical Details**: 
           - If it's a diagram: describe the components, flow, relationships
           - If it's a chart/graph: describe data, trends, axes, legends
           - If it's a table: describe structure and key data points
           - If it's a photograph: describe subjects, setting, context
        4. **Text Elements**: Extract any visible text, labels, captions
        5. **Key Information**: What are the main points this image conveys?
        
        Make your description detailed enough that someone could understand the image's content and purpose without seeing it.
        """

        response = client.chat.completions.create(
            model=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.0
        )
        
        enhanced_description = response.choices[0].message.content
        visual_category = categorize_visual_content(enhanced_description, "")
        
        return enhanced_description, visual_category
        
    except Exception as e:
        st.error(f"Error in image analysis: {e}")
        return "", "general"

def get_text_chunks_with_sources(documents: List[Document], db_name: str = None) -> Tuple[List[Document], List[Document]]:
    """
    Split documents into chunks and separate into visual/text databases
    Returns: (visual_documents, text_documents)
    """
    if not documents:
        return [], []
        
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1200, 
        chunk_overlap=200, 
        length_function=len
    )
    
    visual_documents = []
    text_documents = []
    
    for doc_idx, doc in enumerate(documents):
        if doc.page_content and doc.page_content.strip():
            try:
                # Determine if this is visual or text content
                has_visuals = doc.metadata.get('has_visuals', False)
                processing_method = doc.metadata.get('processing_method', 'text')
                
                # Split the document content
                chunks = text_splitter.split_text(doc.page_content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk.strip():
                        # Extract visual category if applicable
                        visual_category = None
                        if has_visuals:
                            visual_analysis = ""
                            if "Visual Analysis:" in chunk:
                                visual_analysis = chunk.split("Visual Analysis:", 1)[1]
                            visual_category = categorize_visual_content(visual_analysis, chunk)
                        
                        # Create metadata with categorization
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata.update({
                            'chunk_index': chunk_idx,
                            'doc_index': doc_idx,
                            'visual_category': visual_category
                        })
                        
                        chunked_doc = Document(
                            page_content=chunk,
                            metadata=chunk_metadata
                        )
                        
                        # Route to appropriate database
                        if has_visuals or processing_method == 'vision':
                            visual_documents.append(chunked_doc)
                        else:
                            text_documents.append(chunked_doc)
                            
            except Exception as e:
                print(f"Error processing document {doc_idx}: {e}")
    
    return visual_documents, text_documents

def get_vectorstore_from_documents(documents: List[Document]):
    """Creates a FAISS vector store from Document objects with improved error handling."""
    valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not valid_documents:
        st.error("Could not find any valid text content to process in the document(s).")
        return None

    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
            api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
        )

        st.write(f"Creating embeddings for {len(valid_documents)} document chunks...")
        vectorstore = None
        BATCH_SIZE = 16

        progress_bar = st.progress(0)
        
        try:
            for i in range(0, len(valid_documents), BATCH_SIZE):
                batch = valid_documents[i:i + BATCH_SIZE]
                
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
                else:
                    vectorstore.add_documents(documents=batch)
                
                progress = min(((i + BATCH_SIZE) / len(valid_documents)), 1.0)
                progress_bar.progress(progress)
                time.sleep(1)
            
            progress_bar.progress(1.0)

        except Exception as e:
            st.error(f"An error occurred during embedding creation: {e}")
            return None

    except Exception as e:
        st.error(f"An error occurred during embedding initialization: {e}")
        return None

    return vectorstore

def get_conversation_chain_with_sources(vectorstore, faithfulness_threshold: float = 0.5, reranker_top_n: int = 10):
    """Conversation chain with single vectorstore and reranking"""
    try:
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
            api_version=AZURE_CONFIG["OPENAI_API_VERSION"],
            temperature=0.0,
            model_version="latest"
        )
        
        # Create base retriever
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        
        # Use DeepSeek reranker
        try:
            compressor = EnhancedDeepSeekReranker(top_n=reranker_top_n)
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            print(f"DeepSeek reranker created! Will rerank top {reranker_top_n} from 15 retrieved documents.")
            
        except Exception as e:
            print(f"Failed to create DeepSeek reranker: {e}")
            print("Falling back to base retriever without reranking")
            st.warning(f"Reranking failed, using base retrieval: {e}")
            compression_retriever = base_retriever
        
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True,
            output_key='answer'
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=compression_retriever, 
            memory=memory,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def format_response_with_deepseek_metrics(response, question: str, faithfulness_threshold: float):
    """Format response with DeepSeek metrics and enhanced source information."""
    # Extract contexts and answer safely
    source_documents = response.get('source_documents', [])
    contexts = [doc.page_content for doc in source_documents if hasattr(doc, 'page_content')]
    answer = response.get('answer', '')
    
    # Run DeepSeek evaluation
    deepseek_metrics = evaluate_with_deepseek(question, answer, contexts)
    
    # Prepare enhanced source information with ranking details
    sources = []
    for i, doc in enumerate(source_documents):
        if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
            source_info = {
                'rank': i + 1,
                'file_name': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown'),
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'type': doc.metadata.get('type', 'pdf'),
                'relevance_score': doc.metadata.get('relevance_score', None),
                'original_score': doc.metadata.get('original_score', None),
                'has_visuals': doc.metadata.get('has_visuals', False),
                'processing_method': doc.metadata.get('processing_method', 'unknown')
            }
            sources.append(source_info)
    
    # Check if we should block the answer based on faithfulness threshold
    faithfulness_score = deepseek_metrics.get('faithfulness', 0.0) if deepseek_metrics else 0.0
    
    # Determine final answer to display
    if not source_documents:
        final_answer = "Could not find the Answer"
    elif faithfulness_score < faithfulness_threshold and faithfulness_score > 0.0:
        final_answer = "Could not find a reliable answer (faithfulness score too low)"
    else:
        final_answer = answer
    
    return final_answer, sources, deepseek_metrics

def display_all_sources_interactive(sources, response_index: int):
    """Display all retrieved sources with interactive viewing capabilities."""
    if not sources:
        st.info("No relevant sources were found for this response.")
        return
    
    st.markdown("##### Retrieved Sources")
    
    # Create tabs for each source
    if len(sources) == 1:
        # If only one source, display directly without tabs
        source = sources[0]
        display_single_source(source, response_index, 0)
    else:
        # Create tabs for multiple sources
        tab_names = []
        for i, source in enumerate(sources):
            file_name = source.get('file_name', 'Unknown')
            page_num = source.get('page', 1)
            rank = source.get('rank', i + 1)
            relevance_score = source.get('relevance_score', 0.0)
            processing_method = source.get('processing_method', 'unknown')
            
            # Create enhanced tab name with processing method indicator
            method_icon = "🤖" if processing_method == "vision" else "💾" if processing_method == "cached" else "📝"
            
            if relevance_score is not None and relevance_score > 0:
                tab_name = f"#{rank} {method_icon} {file_name[:12]}... (P{page_num}) [{relevance_score:.2f}]"
            else:
                tab_name = f"#{rank} {method_icon} {file_name[:12]}... (P{page_num})"
            tab_names.append(tab_name)
        
        tabs = st.tabs(tab_names)
        
        for i, (tab, source) in enumerate(zip(tabs, sources)):
            with tab:
                display_single_source(source, response_index, i)

def display_single_source(source, response_index: int, source_index: int):
    """Display a single source with viewing capabilities and processing method info."""
    file_name = source.get('file_name', 'Unknown')
    page_num = source.get('page', 1)
    source_type = source.get('type', 'pdf')
    rank = source.get('rank', source_index + 1)
    relevance_score = source.get('relevance_score', None)
    content = source.get('content', '')
    processing_method = source.get('processing_method', 'unknown')
    has_visuals = source.get('has_visuals', False)
    
    # Display source metadata with processing method
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        visual_indicator = "🖼️" if has_visuals else "📄"
        method_indicator = "🤖" if processing_method == "vision" else "💾" if processing_method == "cached" else "📝"
        st.markdown(f"**{visual_indicator}{method_indicator} File:** {file_name}")
    with col2:
        st.markdown(f"**Page:** {page_num}")
    with col3:
        if relevance_score is not None and relevance_score > 0:
            color = "green" if relevance_score > 0.7 else "orange" if relevance_score > 0.4 else "red"
            st.markdown(f"**Score:** <span style='color: {color}'>{relevance_score:.2f}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Rank:** #{rank}")
    with col4:
        if processing_method == "vision":
            st.markdown("🤖 **Vision**")
        elif processing_method == "cached":
            st.markdown("💾 **Cached**")
        else:
            st.markdown("📝 **Text**")
    
    # Processing method indicator
    if processing_method == "vision":
        st.info("🤖 Processed with Vision AI (detected visual content)")
    elif processing_method == "text":
        st.success("📝 Text extraction used (cost-efficient)")
    elif processing_method == "cached":
        st.info("💾 Used cached Vision AI result (cost-efficient)")
    
    # Display content preview
    if content:
        with st.expander("Content Preview", expanded=False):
            # Split content into text and visual analysis if available
            if "Visual Analysis:" in content:
                text_part, visual_part = content.split("Visual Analysis:", 1)
                st.markdown("**Text Content:**")
                st.text(text_part.replace("Text Content:", "").strip())
                st.markdown("**Visual Analysis:**")
                st.text(visual_part.strip())
            else:
                st.text(content)
    
    # Display file viewer if available
    if file_name in st.session_state.get('uploaded_file_bytes', {}):
        if source_type == 'pdf':
            # Create unique key for this source's page state
            page_state_key = f"current_page_{file_name}_{response_index}_{source_index}"
            if page_state_key not in st.session_state:
                st.session_state[page_state_key] = page_num
            
            current_page = st.session_state[page_state_key]
            
            # Create a button to view the PDF
            if st.button(f"View PDF", key=f"view_pdf_{response_index}_{source_index}"):
                st.session_state[f"show_pdf_{response_index}_{source_index}"] = True
            
            # Show PDF viewer if requested
            if st.session_state.get(f"show_pdf_{response_index}_{source_index}", False):
                display_pdf_viewer_for_source(file_name, current_page, response_index, source_index)
                
                # Add close button
                if st.button("Close PDF Viewer", key=f"close_pdf_{response_index}_{source_index}"):
                    st.session_state[f"show_pdf_{response_index}_{source_index}"] = False
                    st.rerun()
                    
        elif source_type == 'image':
            # Create a button to view the image
            if st.button(f"View Image", key=f"view_img_{response_index}_{source_index}"):
                st.session_state[f"show_img_{response_index}_{source_index}"] = True
            
            # Show image viewer if requested
            if st.session_state.get(f"show_img_{response_index}_{source_index}", False):
                file_bytes = st.session_state.uploaded_file_bytes[file_name]
                st.image(file_bytes, caption=f"Source Image: {file_name}", use_container_width=True)
                
                # Add close button
                if st.button("Close Image Viewer", key=f"close_img_{response_index}_{source_index}"):
                    st.session_state[f"show_img_{response_index}_{source_index}"] = False
                    st.rerun()
    else:
        st.warning(f"File '{file_name}' is not available for viewing in this session.")

def display_pdf_viewer_for_source(file_name: str, start_page: int, response_index: int, source_index: int):
    """PDF viewer specifically for individual sources with unique keys."""
    if file_name not in st.session_state.get('uploaded_file_bytes', {}):
        st.error(f"File '{file_name}' not found in current session.")
        return
    
    file_bytes = st.session_state.uploaded_file_bytes[file_name]
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        
        # Create unique key for the page state
        page_state_key = f"current_page_{file_name}_{response_index}_{source_index}"

        st.markdown(f"**{file_name}** - Page {start_page} of {total_pages}")
        
        # Display the PDF image
        page = doc[start_page - 1]
        pix = page.get_pixmap(dpi=150)  # Reduced DPI for faster loading in tabs
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, use_container_width=True)
        
        doc.close()

        # Navigation controls below the image
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("◀ Previous", disabled=(start_page <= 1), key=f"prev_{file_name}_{response_index}_{source_index}"):
                st.session_state[page_state_key] = max(1, start_page - 1)
                st.rerun()
        
        with col2:
            if st.button("Next ▶", disabled=(start_page >= total_pages), key=f"next_{file_name}_{response_index}_{source_index}"):
                st.session_state[page_state_key] = min(total_pages, start_page + 1)
                st.rerun()
        
        with col3:
            target_page = st.number_input(
                f"Go to page (1-{total_pages})", 
                min_value=1, max_value=total_pages, value=start_page,
                key=f"page_input_{file_name}_{response_index}_{source_index}"
            )
            if st.button("Jump", key=f"jump_{file_name}_{response_index}_{source_index}"):
                if target_page != start_page:
                    st.session_state[page_state_key] = target_page
                    st.rerun()

    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

def get_metric_color_neutral(score: float) -> str:
    """Get neutral color for metric scores."""
    return "#333333"  # Dark gray for all scores

def display_ragas_metrics(ragas_metrics):
    """Display RAGAS metrics - now hidden as requested."""
    # This function is intentionally empty to hide faithfulness and relevancy metrics
    pass

def handle_user_input_with_deepseek(user_question: str, faithfulness_threshold: float):
    """Enhanced handler with dual database awareness and intent detection."""
    if st.session_state.conversation:
        try:
            # Detect question intent for better logging
            intent_type, is_visual_question = detect_question_intent(user_question)
            print(f"Processing question: {user_question}")
            print(f"Detected intent: {intent_type}, Visual question: {is_visual_question}")
            
            response = st.session_state.conversation.invoke({'question': user_question})
            
            # Enhanced logging with dual database info
            source_docs = response.get('source_documents', [])
            print(f"Retrieved {len(source_docs)} documents after dual database search and reranking")
            
            visual_sources = 0
            text_sources = 0
            for i, doc in enumerate(source_docs):
                if hasattr(doc, 'metadata'):
                    has_visuals = doc.metadata.get('has_visuals', False)
                    visual_category = doc.metadata.get('visual_category', 'general')
                    processing_method = doc.metadata.get('processing_method', 'unknown')
                    
                    if has_visuals:
                        visual_sources += 1
                    else:
                        text_sources += 1
                    
                    print(f"  Rank {i+1}: {doc.metadata.get('source', 'Unknown')} "
                          f"(Page {doc.metadata.get('page', 'Unknown')}) - {processing_method} "
                          f"- Category: {visual_category}")
            
            print(f"Sources breakdown: {visual_sources} visual, {text_sources} text")
            
            answer, sources, ragas_metrics = format_response_with_deepseek_metrics(
                response, user_question, faithfulness_threshold
            )
            
            # Store enhanced information
            st.session_state.chat_history = response.get('chat_history', [])
            
            if 'response_sources' not in st.session_state:
                st.session_state.response_sources = []
            st.session_state.response_sources.append({
                'question': user_question,
                'sources': sources,
                'ragas_metrics': ragas_metrics,
                'intent_type': intent_type,
                'is_visual_question': is_visual_question,
                'visual_sources_count': visual_sources,
                'text_sources_count': text_sources
            })
            
            return answer, sources, ragas_metrics
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return "An error occurred while processing your question.", [], None
    else:
        st.warning("Please upload and process your files first or load a saved vector database.")
        return "", [], None

def main():
    """Main function to run the Streamlit application."""

    # Inject CSS for the vertical line and column styling
    st.markdown("""
    <style>
    /* Simple vertical line separator */
    .vertical-divider {
        border-left: 2px solid #ddd;
        padding-left: 20px;
        margin-left: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Validate configuration at startup
    if not validate_azure_config():
        st.error("Please configure Azure OpenAI settings before using the application.")
        st.stop()

    # Display header
    display_header()

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = None
    if "document_sources" not in st.session_state:
        st.session_state.document_sources = {}
    if "response_sources" not in st.session_state:
        st.session_state.response_sources = []
    if "current_db_name" not in st.session_state:
        st.session_state.current_db_name = None
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state.uploaded_file_bytes = {}

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

        # Configuration settings
        st.markdown('<div class="section-title">⚙️ Configuration Settings</div>', unsafe_allow_html=True)
        reranker_top_n = st.slider(
            "Reranker Top N", 1, 10, 10, 1,
            help="Number of top documents to keep after reranking (up to 10 sources will be displayed)"
        )
        # Faithfulness threshold removed as metrics are hidden
        faithfulness_threshold = 0.0  # Default value since metrics are hidden
        
        st.markdown('<div class="section-title">💰 Cost Optimization</div>', unsafe_allow_html=True)
        st.info("Smart Vision AI usage:\n- Pre-filtering reduces costs by 60-80%\n- Caching prevents duplicate processing\n- Only uses Vision AI when needed")

        # Display reranker status
        display_reranker_status()
        st.markdown("---")

        # Load existing vector database section
        st.markdown('<div class="section-title">📁 Load Existing Database</div>', unsafe_allow_html=True)
        metadata = load_vector_db_metadata()
        if metadata:
            db_options = []
            for db_name, info in metadata.items():
                try:
                    created_date = datetime.fromisoformat(info['created_at']).strftime("%Y-%m-%d %H:%M")
                    files_info = f"{info.get('total_files', 0)} files"
                    db_options.append(f"{db_name} ({created_date}, {files_info})")
                except Exception as e:
                    db_options.append(f"{db_name} (Invalid metadata)")

            selected_db = st.selectbox("Select a saved vector database:", ["None"] + db_options)

            if st.button("Load Selected Database") and selected_db != "None":
                db_name = selected_db.split(" (")[0]
                with st.spinner("Loading vector database..."):
                    try:
                        embeddings = AzureOpenAIEmbeddings(
                            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
                            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                            api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
                        )
                        vectorstore, document_sources, uploaded_files_data = load_vector_db(db_name, embeddings)
                        if vectorstore:
                            conversation = get_conversation_chain_with_sources(vectorstore, faithfulness_threshold, reranker_top_n)
                            if conversation:
                                st.session_state.conversation = conversation
                                st.session_state.document_sources = document_sources or {}
                                st.session_state.current_db_name = db_name
                                st.session_state.processing_stats = metadata[db_name].get('stats', {})
                                st.session_state.uploaded_file_bytes = uploaded_files_data or {}
                                success_msg = f"Successfully loaded database: {db_name}"
                                if uploaded_files_data:
                                    success_msg += f" (PDF viewing available for {len(uploaded_files_data)} files)"
                                else:
                                    success_msg += " (PDF viewing not available)"
                                st.success(success_msg)
                            else:
                                st.error("Failed to create conversation chain")
                        else:
                            st.error("Failed to load database")
                    except Exception as e:
                        st.error(f"Error loading database: {e}")
        else:
            st.info("No saved vector databases found.")

        st.markdown("---")

        # Document upload section
        st.markdown('<div class="section-title">📤 Document Upload</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose PDF files or images",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        db_name = st.text_input("Database Name (for saving):", value=f"opt_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if st.button("Process Documents (Optimized)"):
            if uploaded_files and db_name:
                with st.spinner("Processing documents with smart Vision AI optimization..."):
                    try:
                        uploaded_files_data = {file.name: file.read() for file in uploaded_files}
                        for file in uploaded_files:
                            file.seek(0)
                        st.session_state.uploaded_file_bytes = uploaded_files_data

                        client = AzureOpenAI(
                            azure_endpoint=AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                            api_key=AZURE_CONFIG["AZURE_OPENAI_API_KEY"],
                            api_version=AZURE_CONFIG["OPENAI_API_VERSION"]
                        )

                        # Use optimized processing function
                        documents, stats, document_sources = get_documents_text_with_optimized_vision(uploaded_files, client)
                        
                        if documents:
                            st.session_state.processing_stats = stats
                            st.session_state.document_sources = document_sources
                            
                            # Get chunks separated by visual/text
                            visual_chunks, text_chunks = get_text_chunks_with_sources(documents, db_name)
                            
                            # Update stats with chunk counts
                            stats['visual_chunks'] = len(visual_chunks)
                            stats['text_chunks'] = len(text_chunks)
                            
                            # Combine all chunks for vectorstore
                            all_chunks = visual_chunks + text_chunks
                            
                            if all_chunks:
                                vectorstore = get_vectorstore_from_documents(all_chunks)

                                if vectorstore:
                                    file_names = [file.name for file in uploaded_files]
                                    if save_vector_db(vectorstore, db_name, document_sources, uploaded_files_data):
                                        save_vector_db_metadata(db_name, file_names, stats)
                                        conversation = get_conversation_chain_with_sources(vectorstore, faithfulness_threshold, reranker_top_n)
                                        if conversation:
                                            st.session_state.conversation = conversation
                                            st.session_state.current_db_name = db_name
                                            st.success(f"Documents processed and saved as '{db_name}'!")
                                            st.info(f"Created {len(visual_chunks)} visual chunks and {len(text_chunks)} text chunks")
                                        else:
                                            st.error("Failed to create conversation chain.")
                                    else:
                                        st.error("Failed to save vector database.")
                                else:
                                    st.error("Failed to create vector database.")
                            else:
                                st.error("No valid chunks created from documents.")
                        else:
                            st.error("Could not extract any text from the uploaded files.")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
            else:
                st.warning("Please upload files and provide a database name.")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.processing_stats:
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Processing Summary</div>', unsafe_allow_html=True)
            display_enhanced_stats(st.session_state.processing_stats)
            if st.session_state.current_db_name:
                st.info(f"Current Database: {st.session_state.current_db_name}")

    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if not st.session_state.conversation:
        display_welcome()
    else:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if message.type == 'human' else "assistant"
            with st.chat_message(role):
                if role == "user":
                    st.markdown(message.content)
                else:
                    col1, col2 = st.columns([2, 3])
                    response_index = i // 2

                    with col1:
                        st.markdown(message.content)
                        if response_index < len(st.session_state.response_sources):
                            response_data = st.session_state.response_sources[response_index]
                            display_ragas_metrics(response_data.get('ragas_metrics'))

                    with col2:
                        # Wrap the content in a div to apply the CSS class for the vertical line
                        st.markdown('<div class="vertical-divider">', unsafe_allow_html=True)
                        if response_index < len(st.session_state.response_sources):
                            response_data = st.session_state.response_sources[response_index]
                            display_all_sources_interactive(response_data.get('sources', []), response_index)
                        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input and new response handling
        if user_question := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                col1, col2 = st.columns([2, 3])
                with col1:
                    with st.spinner("Searching and reranking documents..."):
                        answer, sources, ragas_metrics = handle_user_input_with_deepseek(user_question, faithfulness_threshold)
                        if answer:
                            st.markdown(answer)
                    display_ragas_metrics(ragas_metrics)

                with col2:
                    # Wrap the content in a div to apply the CSS class for the vertical line
                    st.markdown('<div class="vertical-divider">', unsafe_allow_html=True)
                    response_index = len(st.session_state.response_sources) - 1
                    display_all_sources_interactive(sources, response_index)
                    st.markdown('</div>', unsafe_allow_html=True)

            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
