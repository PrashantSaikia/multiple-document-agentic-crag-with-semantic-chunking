from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
import os

pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

if not os.path.exists('MarkdownDocs/'):
    os.makedirs('MarkdownDocs/')

for source in os.listdir('./TariffDocs'):
    if source.endswith('.pdf'):
        result = doc_converter.convert(r"./TariffDocs/"+source)
        md = result.document.export_to_markdown()
        filename = source.split('.pdf')[0]+'.md'
        with open('MarkdownDocs/' + filename, 'w') as file:
            file.write(md)