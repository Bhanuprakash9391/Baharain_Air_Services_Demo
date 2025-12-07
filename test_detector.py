"""
Test script for PDF Image and Table Detection Module

This script demonstrates how to use the pdf_image_table_detector module
to analyze PDF files for images and tables.
"""

from pdf_image_table_detector import (
    analyze_pdf_file,
    test_single_page,
    detect_images_in_pdf_page,
    contains_table_indicators,
    analyze_text_density,
    should_use_vision_ai
)
import fitz  # PyMuPDF
import os


def main():
    """Main test function"""
    
    print("\n" + "="*80)
    print("PDF IMAGE AND TABLE DETECTOR - TEST SCRIPT")
    print("="*80 + "\n")
    
    # Check if there are any PDF files in the current directory
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the current directory.")
        print("\nPlease provide a PDF file path to test:")
        print("Example usage:")
        print("  python test_detector.py")
        print("\nOr modify this script to point to your PDF file.\n")
        
        # Example with manual path
        print("You can also test with a specific file by uncommenting these lines:")
        print("  pdf_path = 'path/to/your/file.pdf'")
        print("  analyze_pdf_file(pdf_path)")
        print("  test_single_page(pdf_path, 1)")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) in the current directory:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf}")
    
    # Test with the first PDF found
    test_pdf = pdf_files[0]
    print(f"\nTesting with: {test_pdf}\n")
    
    # Test 1: Analyze entire PDF
    print("\n" + "-"*80)
    print("TEST 1: Analyzing entire PDF file")
    print("-"*80)
    summary = analyze_pdf_file(test_pdf, verbose=False)
    
    if summary:
        # Test 2: Detailed analysis of first page
        print("\n" + "-"*80)
        print("TEST 2: Detailed analysis of first page")
        print("-"*80)
        test_single_page(test_pdf, 1)
        
        # Test 3: Show pages that require Vision AI
        print("\n" + "-"*80)
        print("TEST 3: Pages requiring Vision AI")
        print("-"*80)
        vision_pages = [p for p in summary['page_details'] if p['use_vision_ai']]
        if vision_pages:
            print(f"\nFound {len(vision_pages)} pages requiring Vision AI:\n")
            for page in vision_pages[:5]:  # Show first 5
                print(f"Page {page['page_num']}:")
                print(f"  Images: {page['image_count']}, Drawings: {page['drawing_count']}")
                print(f"  Has Table: {page['has_table']}")
                print(f"  Reasoning: {', '.join(page['reasoning'])}")
                print()
        else:
            print("\nNo pages require Vision AI processing.")
        
        # Test 4: Show pages with tables
        print("\n" + "-"*80)
        print("TEST 4: Pages with detected tables")
        print("-"*80)
        table_pages = [p for p in summary['page_details'] if p['has_table']]
        if table_pages:
            print(f"\nFound {len(table_pages)} pages with tables:\n")
            for page in table_pages[:5]:  # Show first 5
                print(f"Page {page['page_num']}")
        else:
            print("\nNo tables detected in the PDF.")
        
        # Test 5: Show pages with images
        print("\n" + "-"*80)
        print("TEST 5: Pages with embedded images")
        print("-"*80)
        image_pages = [p for p in summary['page_details'] if p['has_images']]
        if image_pages:
            print(f"\nFound {len(image_pages)} pages with embedded images:\n")
            for page in image_pages[:5]:  # Show first 5
                print(f"Page {page['page_num']}: {page['image_count']} image(s)")
        else:
            print("\nNo embedded images found in the PDF.")
    
    print("\n" + "="*80)
    print("Testing completed!")
    print("="*80 + "\n")


def test_with_custom_pdf(pdf_path: str):
    """
    Test with a custom PDF file path
    
    Args:
        pdf_path: Path to the PDF file to test
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    print(f"\nTesting with custom PDF: {pdf_path}\n")
    
    # Analyze the entire PDF
    summary = analyze_pdf_file(pdf_path, verbose=True)
    
    if summary and summary['total_pages'] > 0:
        # Test first page in detail
        print("\nDetailed analysis of first page:")
        test_single_page(pdf_path, 1)


def test_table_detection_samples():
    """Test table detection with sample text"""
    
    print("\n" + "="*80)
    print("TABLE DETECTION - SAMPLE TEXT TESTS")
    print("="*80 + "\n")
    
    # Sample 1: Clear table with pipes
    sample1 = """
    | Name    | Age | City      |
    |---------|-----|-----------|
    | John    | 30  | New York  |
    | Jane    | 25  | London    |
    """
    
    # Sample 2: Tab-separated table
    sample2 = """
    Product\tPrice\tQuantity
    Apple\t$1.50\t100
    Orange\t$2.00\t75
    Banana\t$0.75\t150
    """
    
    # Sample 3: Space-separated columnar data
    sample3 = """
    Name          Department    Salary
    John Smith    Engineering   75000
    Jane Doe      Marketing     65000
    Bob Johnson   Sales         70000
    """
    
    # Sample 4: Not a table (regular text)
    sample4 = """
    This is just regular text without any table structure.
    It has multiple lines but they don't form columns.
    There's no consistent spacing or delimiters.
    """
    
    samples = [
        ("Pipe-delimited table", sample1),
        ("Tab-separated table", sample2),
        ("Space-separated columns", sample3),
        ("Regular text (not a table)", sample4)
    ]
    
    for name, text in samples:
        result = contains_table_indicators(text)
        print(f"{name}: {'✓ TABLE DETECTED' if result else '✗ No table'}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run main tests
    main()
    
    # Uncomment to test table detection with samples
    # test_table_detection_samples()
    
    # Uncomment to test with a specific PDF file
    # test_with_custom_pdf("path/to/your/file.pdf")
