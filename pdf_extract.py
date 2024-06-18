import pymupdf


class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
