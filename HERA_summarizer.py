from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, Dict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# === Structured Output Models ===
class EventLabel(BaseModel):
    label: str = Field(..., description="A concise, descriptive label for the main event or topic")

class EventSummary(BaseModel):
    summary: str = Field(..., description="Clear and concise summary of the event")


class HERASummarizer:
    def __init__(self, ollama_model: str = "gemma2:9b", max_workers: int = 8):
        self.llm = ChatOllama(model=ollama_model, temperature=0.0)
        self.max_workers = max_workers

        # Structured output LLMs
        self.llm_label = self.llm.with_structured_output(EventLabel)
        self.llm_event_summary = self.llm.with_structured_output(EventSummary)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Prompts
        self.event_grouping_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying the core topic of a text segment. "
                       "Check if the given text input is English, if not your answers should always be in English. "
                       "Return ONLY a short, precise label. No explanations, no extra text."),
            ("human", "Text: {text}\n\nEvent/Topic Label:")
        ])

        self.event_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Write a clear, concise, and coherent summary "
                       "of the following related text segments. Do not add meta-commentary."),
            ("human", "Related segments:\n{event_segments}\n\nSummary:")
        ])

    def split_into_segments(self, text: str) -> List[Document]:
        segments = self.text_splitter.split_text(text)
        return [Document(page_content=s) for s in segments]

    def label_segment(self, segment: Document) -> tuple[str, Document]:
        chain = self.event_grouping_prompt | self.llm_label
        result = chain.invoke({"text": segment.page_content})
        clean_label = result.label.strip().strip('"').strip("'")
        return clean_label, segment

    def group_segments_into_events(self, segments: List[Document]) -> Dict[str, List[Document]]:
        print("Step 2: Labeling segments and grouping into events (parallel)...")
        event_groups: Dict[str, List[Document]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.label_segment, seg): seg for seg in segments}
            for future in tqdm(as_completed(futures), total=len(segments), desc="Labeling segments"):
                label, segment = future.result()
                event_groups.setdefault(label, []).append(segment)

        print(f"Identified {len(event_groups)} distinct events")
        return event_groups

    def summarize_single_event(self, label: str, segments: List[Document]) -> str:
        combined = "\n\n".join([s.page_content for s in segments])
        chain = self.event_summary_prompt | self.llm_event_summary
        result = chain.invoke({"event_segments": combined})
        return f"{label}\n{result.summary.strip()}"

    def summarize_document(self, text: str) -> str:
        print("Step 1: Splitting document into segments...")
        segments = self.split_into_segments(text)
        print(f"Created {len(segments)} segments")

        # Step 2: Parallel labeling (kept — this actually helps)
        event_groups = self.group_segments_into_events(segments)

        # Step 3: Sequential event summarization (no threading — as requested)
        print("Step 3: Summarizing each event (sequential)...")
        event_summaries: List[str] = []

        for label, segments in tqdm(event_groups.items(), desc="Summarizing events"):
            summary = self.summarize_single_event(label, segments)
            event_summaries.append(summary)

        structured_output = f"""Document details:\n{"\n\n".join(event_summaries)}"""
        return structured_output


def load_and_summarize_pdf(pdf_path: str, ollama_model: str = "gemma2:9b", max_workers: int = 10) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loading PDF: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n\n".join([doc.page_content for doc in docs])
    print(f"Extracted {len(docs)} pages → {len(full_text.split())} words")

    summarizer = HERASummarizer(ollama_model=ollama_model, max_workers=max_workers)
    summary = summarizer.summarize_document(full_text)
    return summary


# ========================== USAGE ==========================
if __name__ == "__main__":
    pdf_path = "document.pdf"

    try:
        result = load_and_summarize_pdf(
            pdf_path=pdf_path,
            ollama_model="gemma2:9b",
            max_workers=12
        )

        print("\n" + "="*80)
        print("FINAL HERA SUMMARY")
        print("="*80)
        print(result)

        with open("hera_summary.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("\nSummary saved to hera_summary.txt")

    except Exception as e:
        print(f"Error: {e}")