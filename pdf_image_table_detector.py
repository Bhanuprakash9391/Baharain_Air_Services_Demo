"""
PDF Image and Table Detection Module

This module contains functionality extracted from app_v4.py to detect images and tables
from PDF pages using PyMuPDF (fitz) library.

Key Features:
1. Image Detection: Detects embedded images and vector graphics in PDF pages
2. Table Detection: Uses heuristics to identify table-like structures in text
3. Text Density Analysis: Determines if a page is text-heavy or visual-heavy
4. Vision AI Decision: Decides whether to use expensive Vision AI based on page content

Dependencies:
- PyMuPDF (fitz): For PDF processing
"""

import fitz  # PyMuPDF


# ============================================================================
# IMAGE DETECTION FUNCTIONS
# ============================================================================

def detect_images_in_pdf_page(page):
    """
    Detect images using PyMuPDF's built-in capabilities
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        dict: Dictionary containing:
            - has_images: Boolean indicating if embedded images exist
            - has_drawings: Boolean indicating if vector graphics exist
            - image_count: Number of embedded images
            - drawing_count: Number of vector graphics/shapes
            - total_visual_elements: Total count of images + drawings
    """
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


# ============================================================================
# TEXT DENSITY ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text_density(page):
    """
    Determine if page is text-heavy or visual-heavy
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        dict: Dictionary containing:
            - text_length: Length of extracted text
            - text_coverage_ratio: Ratio of text area to page area
            - text_block_count: Number of text blocks
            - likely_visual: Boolean indicating if page is likely visual-heavy
            - very_sparse_text: Boolean indicating if page has very little text
    """
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


# ============================================================================
# TABLE DETECTION FUNCTIONS
# ============================================================================

def contains_table_indicators(text: str) -> bool:
    """
    A more robust heuristic to detect table-like structures in text.
    It checks for multiple indicators and returns True if at least two are found.
    
    Args:
        text: Text content to analyze
        
    Returns:
        bool: True if text likely contains a table, False otherwise
        
    Detection Indicators:
        1. Presence of common table delimiters (tabs, pipes)
        2. High ratio of spaces (suggesting columnar layout)
        3. Consistent number of columns across multiple lines
        4. Multiple lines with significant number of columns
        5. Presence of financial or numeric symbols
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


# ============================================================================
# VISION AI DECISION FUNCTION
# ============================================================================

def should_use_vision_ai(page, file_name: str, page_num: int, vector_graphics_threshold: int = 300):
    """
    Decide whether to use expensive vision AI based on quick checks,
    now with a threshold for vector graphics to avoid false positives from simple lines.
    
    Args:
        page: PyMuPDF page object
        file_name: Name of the PDF file
        page_num: Page number being analyzed
        vector_graphics_threshold: Minimum number of vector graphics to trigger Vision AI (default: 300)
        
    Returns:
        tuple: (use_vision, analysis_info)
            - use_vision: Boolean indicating whether to use Vision AI
            - analysis_info: Dictionary with detailed analysis information
    """
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
            'has_significant_vector_graphics': image_info['drawing_count'] > vector_graphics_threshold,

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
            analysis_info['reasoning'].append(f"Found {image_info['drawing_count']} vector graphics (above threshold of {vector_graphics_threshold})")
        if conditions['low_text_density']:
            analysis_info['reasoning'].append(f"Low text coverage ratio: {density_info['text_coverage_ratio']:.2f}")
        if conditions['very_little_text']:
            analysis_info['reasoning'].append(f"Very sparse text: {len(basic_text)} characters")
        if conditions['contains_tables']:
            analysis_info['reasoning'].append("Likely contains tables")

        if not use_vision:
            reasoning_text = "Sufficient extractable text"
            if image_info['drawing_count'] > 0:
                reasoning_text += f", and vector graphics count ({image_info['drawing_count']}) is below threshold ({vector_graphics_threshold})"
            analysis_info['reasoning'].append(reasoning_text)

        return use_vision, analysis_info

    except Exception as e:
        print(f"Error in should_use_vision_ai: {e}")
        # Conservative fallback: use vision AI if analysis fails
        return True, {'error': str(e), 'reasoning': ['Error in analysis, using vision AI as fallback']}


# ============================================================================
# TESTING AND DEMO FUNCTIONS
# ============================================================================

def analyze_pdf_file(pdf_path: str, verbose: bool = True):
    """
    Analyze an entire PDF file and report on images and tables detected
    
    Args:
        pdf_path: Path to the PDF file
        verbose: If True, print detailed information for each page
        
    Returns:
        dict: Summary statistics for the entire PDF
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        summary = {
            'total_pages': total_pages,
            'pages_with_images': 0,
            'pages_with_tables': 0,
            'pages_requiring_vision_ai': 0,
            'total_images': 0,
            'total_drawings': 0,
            'page_details': []
        }
        
        print(f"\n{'='*80}")
        print(f"Analyzing PDF: {pdf_path}")
        print(f"Total Pages: {total_pages}")
        print(f"{'='*80}\n")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Detect images
            image_info = detect_images_in_pdf_page(page)
            
            # Detect tables
            text = page.get_text()
            has_table = contains_table_indicators(text)
            
            # Check if Vision AI is needed
            use_vision, analysis_info = should_use_vision_ai(page, pdf_path, page_num + 1)
            
            # Update summary
            if image_info['has_images']:
                summary['pages_with_images'] += 1
            if has_table:
                summary['pages_with_tables'] += 1
            if use_vision:
                summary['pages_requiring_vision_ai'] += 1
            
            summary['total_images'] += image_info['image_count']
            summary['total_drawings'] += image_info['drawing_count']
            
            page_detail = {
                'page_num': page_num + 1,
                'has_images': image_info['has_images'],
                'image_count': image_info['image_count'],
                'drawing_count': image_info['drawing_count'],
                'has_table': has_table,
                'use_vision_ai': use_vision,
                'reasoning': analysis_info.get('reasoning', [])
            }
            summary['page_details'].append(page_detail)
            
            if verbose:
                print(f"Page {page_num + 1}:")
                print(f"  Images: {image_info['image_count']} | Drawings: {image_info['drawing_count']}")
                print(f"  Has Table: {has_table}")
                print(f"  Use Vision AI: {use_vision}")
                if analysis_info.get('reasoning'):
                    print(f"  Reasoning: {', '.join(analysis_info['reasoning'])}")
                print()
        
        doc.close()
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total Pages: {summary['total_pages']}")
        print(f"Pages with Images: {summary['pages_with_images']}")
        print(f"Pages with Tables: {summary['pages_with_tables']}")
        print(f"Pages Requiring Vision AI: {summary['pages_requiring_vision_ai']}")
        print(f"Total Embedded Images: {summary['total_images']}")
        print(f"Total Vector Graphics: {summary['total_drawings']}")
        print(f"Vision AI Cost Savings: {100 * (1 - summary['pages_requiring_vision_ai'] / summary['total_pages']):.1f}%")
        print(f"{'='*80}\n")
        
        return summary
        
    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        return None


def test_single_page(pdf_path: str, page_num: int):
    """
    Test detection on a single page of a PDF
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to analyze (1-indexed)
    """
    try:
        doc = fitz.open(pdf_path)
        
        if page_num < 1 or page_num > len(doc):
            print(f"Error: Page {page_num} is out of range (1-{len(doc)})")
            return
        
        page = doc[page_num - 1]
        
        print(f"\n{'='*80}")
        print(f"Analyzing Page {page_num} of {pdf_path}")
        print(f"{'='*80}\n")
        
        # Image detection
        print("IMAGE DETECTION:")
        image_info = detect_images_in_pdf_page(page)
        for key, value in image_info.items():
            print(f"  {key}: {value}")
        
        # Text density analysis
        print("\nTEXT DENSITY ANALYSIS:")
        density_info = analyze_text_density(page)
        for key, value in density_info.items():
            print(f"  {key}: {value}")
        
        # Table detection
        print("\nTABLE DETECTION:")
        text = page.get_text()
        has_table = contains_table_indicators(text)
        print(f"  Contains table: {has_table}")
        
        # Vision AI decision
        print("\nVISION AI DECISION:")
        use_vision, analysis_info = should_use_vision_ai(page, pdf_path, page_num)
        print(f"  Use Vision AI: {use_vision}")
        print(f"  Reasoning:")
        for reason in analysis_info.get('reasoning', []):
            print(f"    - {reason}")
        
        print(f"\n{'='*80}\n")
        
        doc.close()
        
    except Exception as e:
        print(f"Error testing page: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Analyze PDF files for images and tables
    """
    
    print("\n" + "="*80)
    print("PDF IMAGE AND TABLE DETECTION MODULE - DEMO")
    print("="*80)
    
    # PDF file path - Update this to your PDF location
    pdf_path = "G:/BAS/BAS_Files/02_6111628_BA_Notbedienung_211007_en.pdf"
    
    import os
    if not os.path.exists(pdf_path):
        print("\n⚠️  PDF file not found at the specified path.")
        print(f"   Path: {pdf_path}")
        print("\nPlease update the 'pdf_path' variable in the code to point to your PDF file.")
        print("\nExample usage:")
        print("1. Analyze an entire PDF file:")
        print("   analyze_pdf_file('path/to/your/file.pdf')")
        print("\n2. Test a single page:")
        print("   test_single_page('path/to/your/file.pdf', 1)")
        print("\n3. Use individual functions:")
        print("   - detect_images_in_pdf_page(page)")
        print("   - contains_table_indicators(text)")
        print("   - analyze_text_density(page)")
        print("   - should_use_vision_ai(page, filename, page_num)")
        print("\n" + "="*80 + "\n")
    else:
        print(f"\n📄 Analyzing: {os.path.basename(pdf_path)}\n")
        
        # FULL PDF ANALYSIS
        print("-"*80)
        print("FULL PDF ANALYSIS")
        print("-"*80 + "\n")
        
        summary = analyze_pdf_file(pdf_path, verbose=False)
        
        if summary:
            # DETAILED ANALYSIS OF FIRST 3 PAGES
            print("\n" + "-"*80)
            print("DETAILED ANALYSIS OF FIRST 3 PAGES")
            print("-"*80 + "\n")
            
            for page_num in range(1, min(4, summary['total_pages'] + 1)):
                print(f"\n{'='*80}")
                test_single_page(pdf_path, page_num)
            
            # INTERESTING PAGES SUMMARY
            print("\n" + "-"*80)
            print("INTERESTING PAGES SUMMARY")
            print("-"*80 + "\n")
            
            # Pages with images
            image_pages = [p for p in summary['page_details'] if p['has_images']]
            if image_pages:
                print(f"📷 Pages with embedded images ({len(image_pages)} total):")
                for page in image_pages[:10]:
                    print(f"   Page {page['page_num']}: {page['image_count']} image(s), {page['drawing_count']} drawing(s)")
            else:
                print("📷 No embedded images found")
            
            print()
            
            # Pages with tables
            table_pages = [p for p in summary['page_details'] if p['has_table']]
            if table_pages:
                print(f"📊 Pages with tables ({len(table_pages)} total):")
                for page in table_pages[:10]:
                    print(f"   Page {page['page_num']}")
            else:
                print("📊 No tables detected")
            
            print()
            
            # Pages requiring Vision AI
            vision_pages = [p for p in summary['page_details'] if p['use_vision_ai']]
            if vision_pages:
                print(f"🤖 Pages requiring Vision AI ({len(vision_pages)} total):")
                for page in vision_pages[:10]:
                    reasons = ', '.join(page['reasoning'][:2])  # First 2 reasons
                    print(f"   Page {page['page_num']}: {reasons}")
            else:
                print("🤖 No pages require Vision AI")
            
            # COST ANALYSIS
            print("\n" + "="*80)
            print("COST ANALYSIS")
            print("="*80)
            
            total_pages = summary['total_pages']
            vision_pages_count = summary['pages_requiring_vision_ai']
            text_only_pages = total_pages - vision_pages_count
            
            print(f"\n📄 Total Pages: {total_pages}")
            print(f"✅ Text-only pages (no Vision AI): {text_only_pages}")
            print(f"🔍 Vision AI pages: {vision_pages_count}")
            print(f"💰 Cost Savings: {100 * (1 - vision_pages_count / total_pages):.1f}%")
            
            # Estimate cost savings (assuming $0.01 per Vision AI call)
            cost_per_vision_call = 0.01
            original_cost = total_pages * cost_per_vision_call
            optimized_cost = vision_pages_count * cost_per_vision_call
            savings = original_cost - optimized_cost
            
            print(f"\n💵 Estimated Cost (at ${cost_per_vision_call} per page):")
            print(f"   Without optimization: ${original_cost:.2f}")
            print(f"   With optimization: ${optimized_cost:.2f}")
            print(f"   Savings: ${savings:.2f}")
            
            print("\n" + "="*80)
            print("✅ Analysis completed!")
            print("="*80 + "\n")
        else:
            print("❌ Failed to analyze PDF. Please check the file.")
            print("\n" + "="*80 + "\n")
