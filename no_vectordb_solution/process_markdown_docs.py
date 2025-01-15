# from docling.datamodel.base_models import InputFormat
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# pipeline_options = PdfPipelineOptions(do_table_structure=True)
# pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model

# doc_converter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
#     }
# )

# source = r"./TariffDocs/Port Tariff.pdf"
# result = doc_converter.convert(source)
# result_md = result.document.export_to_markdown()

# ### A. Light Dues
# light_dues_info = result_md.split('## 1.1.1 LIGHT DUES')[1].split('## Exemptions')[0]

# ### B. Vehicle Traffic Service (VTS) Dues
# vts_dues_info = result_md.split('## 2.1.1 VTS CHARGES')[1].split('## Exemptions')[0]

# ### C. Pilotage Dues
# pilotage_dues_info = result_md.split('## 3.3 PILOTAGE SERVICES')[1].split('## 3.6 TUGS/VESSEL ASSISTANCE AND/OR ATTENDANCE')[0]

# ### D. Towage Dues
# towage_dues_info = result_md.split('## 3.6 TUGS/VESSEL ASSISTANCE AND/OR ATTENDANCE')[1].split('## 3.8 BERTHING SERVICES')[0]

# ### E. Line Handling Dues
# line_handling_dues_info = result_md.split('## 3.9 RUNNING OF VESSEL LINES')[1].split('## 3.10 HIRE OF MARINE EQUIPMENT/MARINE SERVICES')[0]

# ### F. Port Dues
# port_dues_info = result_md.split('## 4.1 PORT FEES ON VESSELS')[1].split('## Exemptions')[0]





import os
import re
from typing import Dict, Optional
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode


class MarkdownExtractor:
    def __init__(self, markdown_content: str):
        self.content = markdown_content
        
    def extract_section(self, start_marker: str, end_marker: str) -> str:
        """
        Extracts a section from markdown while preserving formatting.
        
        Args:
            start_marker: The heading or text that marks the start of the section
            end_marker: The heading or text that marks the end of the section
            
        Returns:
            Formatted markdown string for the extracted section
        """
        try:
            # Find the section using regex to handle potential formatting characters
            pattern = f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}"
            match = re.search(pattern, self.content, re.DOTALL)
            
            if not match:
                return ""
                
            section = match.group(1).strip()
            
            # Clean up the extracted section
            # Remove extra newlines while preserving intentional line breaks
            section = re.sub(r'\n{3,}', '\n\n', section)
            
            # Preserve header formatting
            section = re.sub(r'^(#+)\s*(.+)$', r'\1 \2', section, flags=re.MULTILINE)
            
            # Preserve list formatting
            section = re.sub(r'^\s*[-*]\s+(.+)$', r'- \1', section, flags=re.MULTILINE)
            
            # Preserve bold and italic formatting
            section = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', section)
            section = re.sub(r'\*([^*]+)\*', r'*\1*', section)
            
            return section
            
        except Exception as e:
            print(f"Error extracting section: {e}")
            return ""

    def extract_all_sections(self, section_markers: Dict[str, tuple]) -> Dict[str, str]:
        """
        Extracts multiple sections at once.
        
        Args:
            section_markers: Dictionary with section names as keys and tuples of (start_marker, end_marker) as values
            
        Returns:
            Dictionary with section names as keys and extracted formatted content as values
        """
        results = {}
        for section_name, (start_marker, end_marker) in section_markers.items():
            results[section_name] = self.extract_section(start_marker, end_marker)
        return results

def extract_port_tariff_sections(markdown_content: str) -> Dict[str, str]:
    extractor = MarkdownExtractor(markdown_content)
    
    sections = {
        "light_dues": ("## 1.1.1 LIGHT DUES", "## Exemptions"),
        "vts_dues": ("## 2.1.1 VTS CHARGES", "## Exemptions"),
        "pilotage_dues": ("## 3.3 PILOTAGE SERVICES", "## 3.6 TUGS/VESSEL ASSISTANCE AND/OR ATTENDANCE"),
        "towage_dues": ("## 3.6 TUGS/VESSEL ASSISTANCE AND/OR ATTENDANCE", "## 3.8 BERTHING SERVICES"),
        "line_handling_dues": ("## 3.7 MISCELLANEOUS TUG/VESSEL SERVICES", "## 3.10 HIRE OF MARINE EQUIPMENT/MARINE SERVICES"),
        "port_dues": ("## 4.1 PORT FEES ON VESSELS", "## Exemptions")
    }
    
    return extractor.extract_all_sections(sections)

def save_markdown_sections(markdown_content: str, output_dir: str = "markdown_docs_processed"):
    """
    Extracts and saves markdown sections to separate files in the specified directory.
    
    Args:
        markdown_content: The full markdown content to process
        output_dir: Directory where the markdown files will be saved
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all sections
    sections = extract_port_tariff_sections(markdown_content)
    
    # Define pretty names for the files
    file_names = {
        "light_dues": "light_dues.md",
        "vts_dues": "vts_dues.md",
        "pilotage_dues": "pilotage_dues.md",
        "towage_dues": "towage_dues.md",
        "line_handling_dues": "line_handling_dues.md",
        "port_dues": "port_dues.md"
    }
    
    # Write each section to a separate file in the output directory
    for section_name, content in sections.items():
        file_path = os.path.join(output_dir, file_names[section_name])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {file_path}")

# Example usage:
if __name__ == "__main__":
    # Your existing PDF conversion code here
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    source = r"./TariffDocs/Port Tariff.pdf"
    result = doc_converter.convert(source)
    result_md = result.document.export_to_markdown()
    
    # Process and save the markdown sections
    save_markdown_sections(result_md)