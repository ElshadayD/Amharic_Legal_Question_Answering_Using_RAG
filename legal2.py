import os
from PyPDF2 import PdfReader
import nltk

# Make sure you have the NLTK tokenizer resources
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Folder containing PDFs
pdf_folder = 'all_pdfs'

all_tokens = []

# Loop through all PDF files in the folder
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        reader = PdfReader(pdf_path)
        text = ''
        
        # Extract text from all pages
        for page in reader.pages:
            text += page.extract_text() + ' '
        
        # Tokenize the text
        tokens = word_tokenize(text)
        all_tokens.extend(tokens)

# Print all tokens
print(all_tokens)

