from typing import List
from datetime import datetime, timezone

import xxhash
from keybert import KeyBERT
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
from transformers import AutoTokenizer

from src.config import Config

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

keybert = KeyBERT(model="sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_md")

def lemmatize_text(text: str) -> str:
    doc = nlp(text.lower())
    text = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return text

def get_tokenizer():
    return AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL_NAME)

def get_text_splitter(tokenizer, chunk_size: int = 1024):
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,  # The maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=int(chunk_size/10),  # The number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )

def get_embedding_function():
    return HuggingFaceEmbeddings(
     model_name=Config.EMBEDDING_MODEL_NAME,
     encode_kwargs={'normalize_embeddings': True},
)

def get_text_hash(doc: Document) -> int:
    return xxhash.xxh64(doc.page_content).hexdigest()

def get_tokens_in_text(chunk: str) -> int:
    tokenizer = get_tokenizer()
    encoding = tokenizer(chunk, return_tensors="pt")
    return encoding.input_ids.shape[1]

def add_metadata_to_chunk(chunk: Document, doc_id: str, document_hash:str, source_type: str, index: int, token_count: bool = True) -> Document:
    chunk.metadata["document_id"] = doc_id
    chunk.metadata["document_hash"] = document_hash
    chunk.metadata["chunk_hash"] = get_text_hash(chunk)
    chunk.metadata["chunk_index"] = index
    if token_count:
        chunk.metadata["token_count"] = get_tokens_in_text(chunk.page_content)
    chunk.metadata["source_type"] = source_type
    chunk.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    chunk.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
    return chunk

def get_chunk_tags(chunk: Document, top_n: int = 3) -> List[str]:
    text = lemmatize_text(chunk.page_content)
    keywords = keybert.extract_keywords(
        docs=text, 
        keyphrase_ngram_range=(1, 1), 
        stop_words='english',
        top_n=top_n
    )
    return [keyword[0] for keyword in keywords]