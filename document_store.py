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
        self.model = SentenceTransformer(model_name)
        self.documents = list()
        self.vector_db = None
        self.all_chunks = []

    def __extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def __clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space.
        text = re.sub(r'[\x00-\x1F\x7F]+', '', text)  # Remove non-printable characters.
        text = text.strip()
        return text

    def __save_cleaned_text(self, text: str, output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def __preprocess_pdfs(self, input_dir: str, output_dir: str) -> List[Dict[str, str]]:
        results = []

        for pdf_file in tqdm(Path(input_dir).glob("*.pdf"), desc="Processing PDFs"):
            file = str(pdf_file)
            if file in self.documents:
                continue
            try:
                raw_text = self.__extract_text_from_pdf(str(pdf_file))
                cleaned_text = self.__clean_text(raw_text)
                output_file = Path(output_dir) / (pdf_file.stem + "_cleaned.txt")
                self.__save_cleaned_text(cleaned_text, str(output_file))
                results.append({"original_file": str(pdf_file), "cleaned_file": str(output_file)})
                self.documents.append(str(pdf_file))
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        return results
 
    def __split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def __embed_text_chunks(self, chunks: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
        #model = SentenceTransformer(model_name)
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        return embeddings

    def __create_vector_database(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def __extract_text_from_pdf_content(self, content):
        text = ""
        pdf_stream = BytesIO(content)
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    def __preProcessDocument(self, name, content):
        cleaned_text = ""
        raw_text = self.__extract_text_from_pdf_content(content)
        cleaned_text = self.__clean_text(raw_text)
        return cleaned_text

    def addDocument(self, name, content):
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
        embeddings = self.__embed_text_chunks(self.all_chunks)
        self.vector_db = self.__create_vector_database(embeddings)
        print("Embeddings generated.")

    def build(self, in_dirs, out_dirs):
        # Step 1: Preprocess PDFs
        results = self.__preprocess_pdfs(in_dirs, out_dirs)
        if (len(results) > 1) :
            # Step 2: Convert to Text Chunks and Embed
            self.all_chunks = []
            for result in results:
                with open(result["cleaned_file"], "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = self.__split_into_chunks(text)
                    self.all_chunks.extend(chunks)

            embeddings = self.__embed_text_chunks(self.all_chunks)

            # Step 3: Create Vector Database
            self.vector_db = self.__create_vector_database(embeddings)

