import re
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

class DocumentStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print("DocumentStore::__init__")
        self.model = SentenceTransformer(model_name)
        self.documents = list()
        self.vector_db = None
        self.all_chunks = []

    def __extract_text_from_pdf_content(self, content):
        print("DocumentStore::__extract_text_from_pdf_content")
        text = ""
        pdf_stream = BytesIO(content)
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    def __clean_text(self, text: str) -> str:
        print("DocumentStore::__clean_text")
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space.
        text = re.sub(r'[\x00-\x1F\x7F]+', '', text)  # Remove non-printable characters.
        text = text.strip()
        return text

    def __split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        print("DocumentStore::__split_into_chunks")
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def __embed_text_chunks(self, chunks: List[str]) -> np.ndarray:
        print("DocumentStore::__embed_text_chunks")
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        return embeddings

    def __create_vector_database(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        print("DocumentStore::__create_vector_database")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def __preProcessDocument(self, name, content):
        print("DocumentStore::__preProcessDocument")
        cleaned_text = ""
        raw_text = self.__extract_text_from_pdf_content(content)
        cleaned_text = self.__clean_text(raw_text)
        return cleaned_text

    def addDocument(self, name, content):
        print("DocumentStore::addDocument")
        documentName = str(name)
        if documentName in self.documents:
            return

        text = self.__preProcessDocument(name, content)
        if (len(text) > 0) :
            chunks = self.__split_into_chunks(text)
            self.all_chunks.extend(chunks)
            self.documents.append(documentName)
            print(f"{documentName} added.")
            return True
        return False

    def buildEmbeddings(self):
        print("DocumentStore::buildEmbeddings")
        embeddings = self.__embed_text_chunks(self.all_chunks)
        self.vector_db = self.__create_vector_database(embeddings)
        print("Embeddings generated.")

