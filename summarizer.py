from PyPDF2 import PdfReader
import json
import requests
from bs4 import BeautifulSoup
import json
from transformers import pipeline
import re
import math
import fitz 
import yaml

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data loaded from the file.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load configuration at the start
config = load_config()


class PDFExtractor:
      #Initialize PDFExtractor with the path to the PDF file. Args:pdf_path (str): Path to the PDF file.
    
    def __init__(self, pdf_path):
        """
        Initialize PDFExtractor with the path to the PDF file.

        Args:
            pdf_path (str): Path to the PDF file.
        """
        self.pdf_path = pdf_path

    def extract_metadata(self):
        """
        Extract metadata from a PDF file.

        Returns:
            dict or str: Extracted metadata as a dictionary, or an error message.
        """
        try:
            reader = PdfReader(self.pdf_path)
            metadata = reader.metadata
            if metadata:
                return {key[1:]: value for key, value in metadata.items()}  # Clean up keys
            else:
                return "No metadata found in this PDF"
        except Exception as e:
            return f"An error occurred: {e}"

    def extract_metadata_to_json(self, output_json_path):
        """
        Extract metadata from the PDF and save it to a JSON file.

        Args:
            output_json_path (str): Path to save the metadata JSON file.

        Returns:
            str: Success or error message.
        """
        try:
            reader = PdfReader(self.pdf_path)
            metadata = reader.metadata

            if metadata:
                structured_metadata = {
                    "Title": metadata.get('/Title', 'Title not found'),
                    "Author": metadata.get('/Author', 'Author not found'),
                    "Creator": metadata.get('/Creator', 'Creator not found'),
                    "Producer": metadata.get('/Producer', 'Producer not found'),
                    "CreationDate": metadata.get('/CreationDate', 'Creation date not found'),
                    "ModificationDate": metadata.get('/ModDate', 'Modification date not found'),
                    "Keywords": metadata.get('/Keywords', 'Keywords not found'),
                    "Subject": metadata.get('/Subject', 'Subject not found'),
                    "Trapped": metadata.get('/Trapped', 'Trapped not found')
                }

                with open(output_json_path, 'w') as json_file:
                    json.dump(structured_metadata, json_file, indent=4)

                return f"Metadata successfully saved to {output_json_path}"
            else:
                return "No metadata found in this PDF"
        except Exception as e:
            return f"An error occurred: {e}"
        
        


class WebScraper:
    #A class to extract metadata from a webpage.

    def __init__(self, url):

        #Initialize WebScraper with the webpage URL.
        self.url = url

    def fetch_webpage(self):
        """
        Fetch the HTML content of a webpage.

        Returns:
            str: HTML content of the webpage.

        Raises:
            Exception: If the webpage cannot be fetched.
        """
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch webpage. Status code: {response.status_code}")

    def extract_metadata(self):
        """
        Extract metadata from a webpage.

        Returns:
            dict: Extracted metadata including title, author, keywords, etc.
        """
        html_content = self.fetch_webpage()
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract Title
        title = soup.title.string if soup.title else "Title not found"

        # Extract Author
        author = None
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag:
            author = author_tag.get("content", "Author not found")

        # Fallback: Try extracting from <h1> or similar elements
        if not author:
            author_tag = soup.find("h1")
            if author_tag:
                author = author_tag.string.strip() if author_tag.string else "Author not found"

        # Extract additional metadata
        creation_date = soup.find("meta", attrs={"name": "creation_date"})
        modification_date = soup.find("meta", attrs={"name": "modification_date"})
        keywords = soup.find("meta", attrs={"name": "keywords"})
        subject = soup.find("meta", attrs={"name": "subject"})

        # Structure metadata
        structured_metadata = {
            "Title": title,
            "Author": author if author else "Author not found",
            "CreationDate": creation_date.get("content", "Creation date not found") if creation_date else "Creation date not found",
            "ModificationDate": modification_date.get("content", "Modification date not found") if modification_date else "Modification date not found",
            "Keywords": keywords.get("content", "Keywords not found") if keywords else "Keywords not found",
            "Subject": subject.get("content", "Subject not found") if subject else "Subject not found"
        }

        return structured_metadata

    def save_metadata_to_json(self, output_json_path):
        """
        Extract and save webpage metadata to a JSON file.

        Args:
            output_json_path (str): Path to save the metadata JSON file.

        Returns:
            str: Success message.
        """
        metadata = self.extract_metadata()
        with open(output_json_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        return f"Metadata successfully saved to {output_json_path}"



class TextSummarizer:
    """
    A class to extract and summarize text from a PDF file.
    """
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Initialize TextSummarizer with a pretrained summarization model.

        Args:
            model_name (str): Name of the transformer model to use for summarization.
        """
        self.model_name = model_name
        self.summarizer = pipeline("summarization", model=model_name, framework="pt")  # Force PyTorch

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text.
        """
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()

    def clean_text(self, text):
        """
        Clean extracted text by removing references and unnecessary characters.

        Args:
            text (str): Extracted text.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r'\[\d+\]|\(\d+\)', '', text)  # Remove references
        text = re.sub(r'[^a-zA-Z0-9.,!? \n]+', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        text = text.replace('.', '.\n')  # Standardize line breaks
        return text

    def summarize_text_in_chunks(self, text, max_chunk_size=256):
        """
        Summarize text in smaller chunks to avoid exceeding model token limits.

        Args:
            text (str): The text to summarize.
            max_chunk_size (int): Maximum size of each chunk.

        Returns:
            str: Combined summary.
        """
        num_chunks = math.ceil(len(text) / max_chunk_size)
        summaries = []

        for i in range(num_chunks):
            chunk = text[i * max_chunk_size : (i + 1) * max_chunk_size]
            summary = self.summarizer(chunk, max_length=60, min_length=30, do_sample=False, clean_up_tokenization_spaces=True)
            summaries.append(summary[0]['summary_text'])

        return " ".join(summaries)

    def summarize_pdf(self, pdf_path):
        """
        Extract, clean, and summarize text from a PDF.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Summary of the extracted text.
        """
        text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = self.clean_text(text)
        summary = self.summarize_text_in_chunks(cleaned_text)
        return summary


#function calls
if __name__ == "__main__":
    # PDF Extraction
    pdf_extractor = PDFExtractor(config["pdf"]["path"])
    pdf_metadata = pdf_extractor.extract_metadata()
    print("PDF Metadata:", pdf_metadata)

    # Web Scraping
    web_scraper = WebScraper(config["web"]["url"])
    web_metadata = web_scraper.extract_metadata()
    print("Web Metadata:", web_metadata)

    # Text Summarization
    text_summarizer = TextSummarizer(model_name=config["summarization"]["model_name"])
    pdf_path = config["pdf"]["path"] 
    summary = text_summarizer.summarize_pdf(pdf_path)
    print("Summary:", summary)