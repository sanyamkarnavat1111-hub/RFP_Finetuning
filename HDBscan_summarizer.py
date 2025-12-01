# hera_summarizer_final.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, Dict
import os
import numpy as np
import hdbscan
from tqdm import tqdm


# === Structured Output Models ===
class EventTitle(BaseModel):
    title: str = Field(..., description="Concise English title")

class EventSummary(BaseModel):
    summary: str = Field(..., description="Clear English summary")


class HERASummarizer:
    def __init__(self, ollama_model: str = "gemma2:9b"):
        self.llm = ChatOllama(model=ollama_model, temperature=0.0)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.llm_title = self.llm.with_structured_output(EventTitle)
        self.llm_summary = self.llm.with_structured_output(EventSummary)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1400,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # === STRICT ENGLISH PROMPTS ===
        self.title_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating concise titles. "
                       "Even if the input is in Arabic or any other language, "
                       "you MUST respond ONLY in English. Return only the title. No quotes. No extra text."),
            ("human", "Text: {text}\n\nTitle:")
        ])

        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. "
                       "Even if the input text is in Arabic or any other language, "
                       "your summary MUST be in clear, professional English. "
                       "Do not write in Arabic. Do not mix languages. No meta-commentary."),
            ("human", "Text:\n{text}\n\nSummary:")
        ])

    def split_into_segments(self, text: str) -> List[Document]:
        segments = self.text_splitter.split_text(text)
        return [Document(page_content=s) for s in segments]

    def group_with_hdbscan(self, segments: List[Document]) -> Dict[int, List[Document]]:
        print("Step 2: Grouping segments using HDBSCAN + nomic-embed-text...")
        texts = [seg.page_content for seg in segments]
        print(f"   → Embedding {len(texts)} segments...")
        vectors = self.embeddings.embed_documents(texts)
        vectors = np.array(vectors)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        print("   → Running HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(vectors)

        groups = {}
        for label, seg in zip(labels, segments):
            if label == -1:
                label = f"outlier_{len(groups)}"
            groups.setdefault(label, []).append(seg)

        # Sort by first appearance
        sorted_groups = {}
        for label in sorted(groups, key=lambda x: segments.index(groups[x][0])):
            sorted_groups[label] = groups[label]

        print(f"   → Found {len([l for l in labels if l != -1])} main events")
        return sorted_groups

    def generate_title(self, segments: List[Document]) -> str:
        sample = "\n\n".join([s.page_content[:1000] for s in segments[:3]])
        chain = self.title_prompt | self.llm_title
        try:
            result = chain.invoke({"text": sample})
            return result.title.strip()
        except:
            return "Untitled Event"

    def summarize_event(self, segments: List[Document]) -> str:
        text = "\n\n".join([s.page_content for s in segments])
        if len(text) > 15000:
            text = text[:15000] + "\n\n[Content truncated]"

        chain = self.summary_prompt | self.llm_summary
        try:
            result = chain.invoke({"text": text})
            return result.summary.strip()
        except:
            return "[Summary generation failed]"

    def summarize_document(self, text: str) -> str:
        print("Step 1: Splitting document...")
        segments = self.split_into_segments(text)
        print(f"   → {len(segments)} segments")

        event_groups = self.group_with_hdbscan(segments)

        print("Step 3: Generating English titles and summaries...")
        event_summaries = []

        for group_id, segs in tqdm(event_groups.items(), desc="Events"):
            title = self.generate_title(segs)
            summary = self.summarize_event(segs)
            event_summaries.append(f"{title}\n{summary}\n")

        output = ""
        output += "\n".join(event_summaries)
        return output


# === PDF Loader ===
def load_and_summarize_pdf(pdf_path: str, ollama_model: str = "gemma2:9b") -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loading: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n\n".join([d.page_content for d in docs])
    print(f"   → {len(docs)} pages extracted")

    summarizer = HERASummarizer(ollama_model=ollama_model)
    result = summarizer.summarize_document(text)

    # TEXT FILE ONLY
    with open("hera_summary.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print("\nDone! Summary saved to hera_summary.txt")
    return result


if __name__ == "__main__":
    pdf_path = "document.pdf"
    load_and_summarize_pdf(pdf_path, ollama_model="gemma2:9b")