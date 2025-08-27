# services/document_processor.py
import io
import logging
import pdfplumber
import docx
from bs4 import BeautifulSoup
from utils.text_processing import clean_extracted_text

logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Enhanced document processor with PDF improvements and progress tracking"""
    
    def __init__(self):
        self.progress_cache = {}
        self.cache_service = None  # Will be injected if available
    
    async def process_file(self, file_content: bytes, filename: str, document_id: str) -> str:
        """Process file and extract text based on file type"""
        try:
            filename_lower = filename.lower()
            
            if filename_lower.endswith('.pdf'):
                return await self.extract_text_from_pdf(file_content, filename, document_id)
            elif filename_lower.endswith('.docx'):
                return await self.extract_text_from_docx(file_content, filename, document_id)
            elif filename_lower.endswith(('.txt', '.md')):
                await self.update_progress(document_id, 30, "Reading text file...")
                text_content = file_content.decode('utf-8', errors='ignore')
                await self.update_progress(document_id, 60, "Text file processed")
                return text_content
            elif filename_lower.endswith(('.html', '.htm')):
                return await self.extract_text_from_html(file_content, filename, document_id)
            elif filename_lower.endswith('.json'):
                await self.update_progress(document_id, 30, "Parsing JSON...")
                text_content = file_content.decode('utf-8', errors='ignore')
                await self.update_progress(document_id, 60, "JSON file processed")
                return text_content
            else:
                await self.update_progress(document_id, 30, "Reading file as text...")
                text_content = file_content.decode('utf-8', errors='ignore')
                await self.update_progress(document_id, 60, "File processed as text")
                return text_content
                
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
            raise Exception(f"Failed to extract text from {filename}: {str(e)}")
    
    async def extract_text_from_pdf(self, file_content: bytes, filename: str, document_id: str) -> str:
        """Enhanced PDF text extraction using pdfplumber with progress tracking"""
        try:
            text_content = []
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"📄 Processing PDF: {filename} ({total_pages} pages)")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    progress = 30 + (page_num / total_pages) * 30
                    await self.update_progress(document_id, progress, f"Extracting text from page {page_num}/{total_pages}")
                    
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = clean_extracted_text(page_text)
                        if cleaned_text.strip():
                            text_content.append(cleaned_text)
            
            if not text_content:
                return f"Warning: PDF file '{filename}' appears to be empty or contains only images."
            
            final_text = "\n\n".join(text_content)
            logger.info(f"✅ PDF extraction successful: {len(final_text)} characters from {len(text_content)} pages")
            return final_text
            
        except Exception as e:
            logger.warning(f"⚠️ PDF extraction failed for {filename}: {e}")
            return f"Error: Unable to extract text from PDF file '{filename}'. The file may be corrupted, encrypted, or contain only images."
    
    async def extract_text_from_docx(self, file_content: bytes, filename: str, document_id: str) -> str:
        """Extract text from DOCX files with progress tracking"""
        try:
            await self.update_progress(document_id, 30, "Extracting text from DOCX...")
            doc = docx.Document(io.BytesIO(file_content))
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())
            
            final_text = "\n\n".join(text_content)
            logger.info(f"✅ DOCX extraction successful: {len(final_text)} characters")
            return final_text
            
        except Exception as e:
            logger.error(f"❌ DOCX extraction failed for {filename}: {e}")
            raise ValueError(f"Failed to extract text from DOCX file: {str(e)}")
    
    async def extract_text_from_html(self, file_content: bytes, filename: str, document_id: str) -> str:
        """Extract text from HTML files with progress tracking"""
        try:
            await self.update_progress(document_id, 30, "Parsing HTML...")
            html_content = file_content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            text_content = soup.get_text()
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            final_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"✅ HTML extraction successful: {len(final_text)} characters")
            return final_text
            
        except Exception as e:
            logger.error(f"❌ HTML extraction failed for {filename}: {e}")
            raise ValueError(f"Failed to extract text from HTML file: {str(e)}")
    
    async def update_progress(self, document_id: str, progress: float, message: str):
        """Update processing progress - uses cache service if available"""
        try:
            if self.cache_service:
                self.cache_service.update_progress(document_id, "processing", progress, message)
            else:
                # Just log if no cache service
                logger.info(f"📊 Progress [{document_id}]: {progress}% - {message}")
        except Exception as e:
            logger.warning(f"Could not update progress: {e}")
