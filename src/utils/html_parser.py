# src/html_parser.py

import os
import re
from bs4 import BeautifulSoup
import pandas as pd # Used for reading HTML tables into DataFrames


# --- Helper Function: _remove_references_from_paragraphs (Keeping consistent with xml_parser.py) ---
def _remove_references_from_paragraphs(paragraphs: list) -> list:
    """
    Helper function: Filters out paragraphs that belong to the references,
    acknowledgements, or appendix sections.
    """
    if not paragraphs:
        return []
    reference_section_patterns = [
        r'^\s*References?\s*$', r'^\s*BIBLIOGRAPHY\s*$', r'^\s*LITERATURE\s+CITED\s*$',
        r'^\s*Acknowledgement(s)?\s*$', r'^\s*Appendix(es)?\s*$', r'^\s*SUPPORTING\s+INFORMATION\s*$',
        r'^\s*SUPPLEMENTARY\s+MATERIALS?\s*$', r'^\s*Note(s)?\s+on\s+Contributor(s)?\s*$',
        r'^\s*AUTHOR\s+CONTRIBUTIONS?\s*$', r'^\s*FUNDING\s*$', r'^\s*CONFLICTS?\s+OF\s+INTEREST\s*$',
        r'^\s*DATA\s+AVAILABILITY\s+STATEMENT\s*$', r'^\s*ORCID\s*$',
    ]
    in_ancillary_section = False
    processed_paragraphs = []
    for p in paragraphs:
        is_ancillary_header = any(re.fullmatch(pattern, p, re.IGNORECASE) for pattern in reference_section_patterns)
        if is_ancillary_header:
            in_ancillary_section = True
            continue
        if in_ancillary_section:
            continue
        processed_paragraphs.append(p)
    final_cleaned_paragraphs = [
        p for p in processed_paragraphs
        if len(p) > 20 and not re.fullmatch(r'^\s*\d+\.?\s*$', p.strip())
    ]
    return final_cleaned_paragraphs

# --- Helper Function: Convert Pandas DataFrame to Markdown Table Text ---
def _convert_df_to_markdown_table(df: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame to a Markdown table string.
    This is used for tables extracted from HTML to ensure consistent output format.
    """
    if df.empty:
        return ""
    return df.to_markdown(index=False)


def extract_from_html(html_file_path: str) -> dict | None:
    """
    Extracts common literary information from an HTML file, including title, authors,
    abstract, keywords, body paragraphs, structured sections, and table data.
    It cleans the HTML content before extraction.

    Args:
        html_file_path (str): The path to the HTML file.

    Returns:
        dict | None: A dictionary containing the extracted information if successful,
                     otherwise None.
    """
    extracted_data = {
        'file_path': html_file_path,
        'title': None,
        'authors': [],
        'abstract': None,
        'keywords': [],
        'body_paragraphs': [],  # Main text paragraphs (flattened, references removed)
        'sections': [],         # New: Structured list of sections (title, content)
        'tables_data': []       # List of dictionaries, each containing table ID, caption, and markdown data
    }

    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # --- Initial Cleaning: Remove noise elements ---
        unwanted_selectors = [
            'script', 'style', 'nav', 'footer', 'header', 'aside',
            '.sidebar', '.ad', '.ads', '.header', '.footer', '.navbar',
            '.related-articles', '.comments', '#comments', '#sidebar',
            '[aria-label="breadcrumbs"]',
            'noscript',
            'a', # Remove links, their text might be distracting
            'button', 'input', 'form' # Remove interactive elements
        ]
        for selector in unwanted_selectors:
            for tag in soup.select(selector):
                tag.extract() # Remove the tag and its contents

        # --- 1. Extract Title ---
        title_tag = soup.find('title')
        if title_tag and title_tag.text:
            extracted_data['title'] = title_tag.text.strip()
        else:
            h1_tag = soup.find('h1')
            if h1_tag and h1_tag.text:
                extracted_data['title'] = h1_tag.text.strip()


        # --- 2. Extract Authors ---
        meta_author = soup.find('meta', {'name': 'author'})
        if meta_author and meta_author.get('content'):
            extracted_data['authors'].append(meta_author['content'].strip())
        else:
            author_tags = soup.find_all(class_=re.compile(r'author|contrib-name|contributor', re.IGNORECASE))
            if author_tags:
                for tag in author_tags:
                    author_name = tag.get_text(strip=True)
                    if author_name and len(author_name) < 100:
                         extracted_data['authors'].append(author_name)
                extracted_data['authors'] = list(dict.fromkeys(extracted_data['authors']))

        # --- 3. Extract Abstract ---
        abstract_tag = soup.find(class_=re.compile(r'abstract', re.IGNORECASE)) or \
                       soup.find('meta', {'name': 'description'})
        if abstract_tag:
            if abstract_tag.name == 'meta':
                abstract_text = abstract_tag.get('content', '')
            else:
                abstract_text = abstract_tag.get_text(separator=' ', strip=True)
            
            if abstract_text:
                extracted_data['abstract'] = re.sub(r'\s+', ' ', abstract_text).strip()


        # --- 4. Extract Keywords ---
        meta_keywords = soup.find('meta', {'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            keywords_str = meta_keywords['content']
            extracted_data['keywords'] = [k.strip() for k in keywords_str.split(',') if k.strip()]
        else:
            kwd_tags = soup.find_all(class_=re.compile(r'keyword|kwd', re.IGNORECASE))
            for tag in kwd_tags:
                kwd_text = tag.get_text(strip=True)
                if kwd_text and len(kwd_text) < 50:
                    extracted_data['keywords'].extend([k.strip() for k in kwd_text.split(',') if k.strip()])
            extracted_data['keywords'] = list(dict.fromkeys(extracted_data['keywords']))


        # --- 5. Extract Body Sections and Paragraphs (with Ancillary Removal) ---
        # Strategy: Identify common main content containers, then iterate through their headings (h1-h6)
        # to define sections.
        
        # Look for the most likely main content container. Customize this based on specific journal HTML.
        main_content_area = soup.find('article') or \
                            soup.find(id=re.compile(r'article-body|main-content|body', re.IGNORECASE)) or \
                            soup.find(class_=re.compile(r'article-body|main-content|body', re.IGNORECASE))

        if not main_content_area:
            main_content_area = soup.body # Fallback to entire body if no specific main area found

        # all_body_paragraphs_flat_list will store all main content paragraphs for global filtering
        all_body_paragraphs_flat_list = []
        
        # --- Sectioning Logic ---
        # Iterate through elements, looking for headings that define new sections.
        # This is a heuristic approach, as HTML isn't as strictly structured as XML for <sec> tags.
        
        # Keep track of the current section's paragraphs
        current_section_title = "Introduction" # Default for initial content before first heading
        current_section_paragraphs = []
        
        # Find all top-level headings and paragraphs that are direct children of a potential main content area,
        # or just iterate through all p and h tags in order.
        # Use a list of all relevant tags in parsing order
        content_tags = main_content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'div', 'section'])
        
        for element in content_tags:
            # Skip elements that are likely captions or table content already handled
            if element.name == 'p' and element.find_parent(['caption', 'table']):
                continue
            if element.name.startswith('h'): # Found a heading, potential new section
                # If current section has content, save it
                if current_section_paragraphs:
                    extracted_data['sections'].append({
                        'title': current_section_title.strip(),
                        'paragraphs': list(current_section_paragraphs), # Save a copy
                        'content_flat': " ".join(current_section_paragraphs).strip()
                    })
                
                # Start new section
                current_section_title = element.get_text(strip=True)
                current_section_paragraphs = []
            
            elif element.name == 'p': # Found a paragraph
                p_text = element.get_text(separator=' ', strip=True)
                if p_text:
                    cleaned_p_text = re.sub(r'\s+', ' ', p_text).strip()
                    current_section_paragraphs.append(cleaned_p_text)
                    all_body_paragraphs_flat_list.append(cleaned_p_text) # Add to flat list

            elif element.name in ['ul', 'ol']: # Handle lists within sections
                list_items_text = [li.get_text(separator=' ', strip=True) for li in element.find_all('li')]
                if list_items_text:
                    # Append list items as separate paragraphs or joined as one (LLM prefers paragraphs)
                    for li_text in list_items_text:
                        cleaned_li_text = re.sub(r'\s+', ' ', li_text).strip()
                        current_section_paragraphs.append(cleaned_li_text)
                        all_body_paragraphs_flat_list.append(cleaned_li_text)
            
            # You might need to add specific handling for <div> or <section> elements
            # that act as content containers but aren't headings or paragraphs.
            # This is highly site-specific.

        # Add the last section after the loop
        if current_section_paragraphs:
            extracted_data['sections'].append({
                'title': current_section_title.strip(),
                'paragraphs': list(current_section_paragraphs),
                'content_flat': " ".join(current_section_paragraphs).strip()
            })

        # Apply global reference removal to the flat list of paragraphs
        extracted_data['body_paragraphs'] = _remove_references_from_paragraphs(all_body_paragraphs_flat_list)


        # --- 6. Extract Table Data ---
        tables_dfs = pd.read_html(html_file_path, flavor='bs4') # Use 'bs4' flavor with BeautifulSoup

        for i, df in enumerate(tables_dfs):
            caption = ""
            # Try to find caption for the table by looking for <caption/> or <p> near the table
            # This is a heuristic and might need refinement based on actual HTML structures.
            # We need to find the *original* table tag in the soup first.
            original_table_tags = soup.find_all('table')
            if i < len(original_table_tags):
                table_tag = original_table_tags[i]
                caption_tag = table_tag.find('caption')
                if caption_tag and caption_tag.text:
                    caption = caption_tag.get_text(strip=True)
                else: # Fallback to common patterns near table, but only if it's not a general paragraph
                    # Look for preceding p tags that might be captions
                    prev_elements = table_tag.find_previous_siblings()
                    for prev_elem in prev_elements:
                        if prev_elem.name == 'p' and ('table' in prev_elem.get_text(strip=True).lower() or 'fig' in prev_elem.get_text(strip=True).lower()) and len(prev_elem.get_text(strip=True)) < 200:
                            caption = prev_elem.get_text(strip=True)
                            break
                        # Stop if we hit another section/heading before finding caption
                        if prev_elem.name.startswith('h') or prev_elem.name in ['div', 'section']:
                            break

            # Convert DataFrame to Markdown text
            markdown_table_text = _convert_df_to_markdown_table(df)

            extracted_data['tables_data'].append({
                'id': f"HTML_Table_{i+1}", # Unique ID for HTML tables
                'caption': caption,
                'data_rows': df.values.tolist(), # Store raw rows as list of lists
                'text_representation': markdown_table_text # Text for LLM
            })
        

    except Exception as e:
        print(f"  [HTML Parser Error] An unexpected error occurred processing {html_file_path}: {e}")
        return None

    return extracted_data

# --- Testing Block (Only runs if the script is executed directly) ---
if __name__ == '__main__':
    # Create a dummy HTML file for testing
    sample_html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>HTML Test Article: Nanoparticles in Drug Delivery</title>
    <meta name="author" content="Alice Smith, Bob Johnson">
    <meta name="keywords" content="nanoparticles, drug delivery, PLGA, test, html">
    <style>body { font-family: sans-serif; }</style>
</head>
<body>
    <header><nav>Link1 | Link2</nav></header>
    <div class="main-content-area">
        <h1>HTML Test Article: Nanoparticles in Drug Delivery</h1>
        <section class="abstract-section">
            <h2>Abstract</h2>
            <p>This study explores PLGA nanoparticles (MW 50 kDa, 1:1 LA:GA) for enhanced drug delivery. Particles had a size of 150 nm and zeta potential of -20 mV. Encapsulation efficiency was 90%.</p>
        </section>
        <section class="methods-section">
            <h2>Methods</h2>
            <p>PLGA was dissolved in acetone. Nanoparticles were prepared via nanoprecipitation at 25°C. Stirring was at 800 rpm for 2 hours.</p>
            <p>Particle size was measured by DLS.</p>
            <ul><li>Item 1</li><li>Item 2 with more text</li></ul>
        </section>
        <section class="results-section">
            <h2>Results</h2>
            <p>The particles exhibited uniform size distribution.</p>
            <p>Table 1 provides a summary of characterization data.</p>
            <table>
                <caption>Table 1. Key Properties of Nanoparticles</caption>
                <thead>
                    <tr><th>Property</th><th>Value</th><th>Unit</th></tr>
                </thead>
                <tbody>
                    <tr><td>Size (DLS)</td><td>150</td><td>nm</td></tr>
                    <tr><td>PDI</td><td>0.18</td><td></td></tr>
                    <tr><td>Zeta Potential</td><td>-20</td><td>mV</td></tr>
                </tbody>
            </table>
            <p>Drug loading was 12% (w/w).</p>
        </section>
        <section class="discussion-section">
            <h2>Discussion</h2>
            <p>Our findings align with previous studies on PLGA. The high encapsulation efficiency is promising for clinical applications.</p>
        </section>
        <section class="conclusion-section">
            <h2>Conclusion</h2>
            <p>PLGA nanoparticles show great potential.</p>
        </section>
    </div>
    <footer>
        <p>Copyright 2025. All rights reserved.</p>
        <p>References:</p>
        <ul>
            <li>[1] Ref A. et al., J.X, 2023.</li>
            <li>[2] Ref B. et al., J.Y, 2024.</li>
        </ul>
    </footer>
</body>
</html>
"""
    # Create directory if it doesn't exist
    html_dir = 'data/raw_htmls'
    os.makedirs(html_dir, exist_ok=True)
    test_html_path = os.path.join(html_dir, 'sample_article.html')

    with open(test_html_path, 'w', encoding='utf-8') as f:
        f.write(sample_html_content)

    print(f"Sample HTML created at: {test_html_path}")

    # Call the extraction function
    extracted = extract_from_html(test_html_path)

    if extracted:
        print("\n--- Extracted Data Summary from Sample HTML ---")
        for key, value in extracted.items():
            if key == 'body_paragraphs':
                print(f"{key}: {len(value)} paragraphs (flat list, references removed)")
                print(f"  Last {min(5, len(value))} Body Paragraphs:")
                for i, p in enumerate(value[-min(5, len(value)):]):
                    print(f"    - {p[:150]}...")
            elif key == 'sections':
                print(f"{key}: {len(value)} sections found (structured)")
                for sec in value[:3]: # Print first 3 sections
                    print(f"  - Section Title: '{sec.get('title', 'N/A')}'")
                    print(f"    Paragraphs: {len(sec.get('paragraphs', []))}")
                    print(f"    Content (flat): {sec.get('content_flat', '')[:100]}...")
            elif key == 'tables_data':
                print(f"{key}: {len(value)} tables found.")
                for table_info in value:
                    print(f"  Table ID: {table_info.get('id', 'N/A')}, Caption: {table_info.get('caption', 'N/A')[:50]}...")
                    print("  Table Text Representation (for LLM):")
                    print(table_info.get('text_representation', 'N/A'))
            elif isinstance(value, list):
                print(f"{key}: {'; '.join(value)}")
            else:
                print(f"{key}: {value}")
    else:
        print("HTML extraction failed.")