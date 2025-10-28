import os
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

RAW_DOCS_DIR = 'data'
OUTPUT_PATH = 'processed/chunks.txt'

def chunk_documents():
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as chunk_file:
        for fname in os.listdir(RAW_DOCS_DIR):
            with open(os.path.join(RAW_DOCS_DIR, fname), encoding='utf-8') as f:
                text = f.read()
                sentences = sent_tokenize(text)
                for sent in sentences:
                    chunk_file.write(sent.strip() + '\n')

if __name__ == "__main__":
    chunk_documents()
    print("Chunking complete! All sentences saved to:", OUTPUT_PATH)
