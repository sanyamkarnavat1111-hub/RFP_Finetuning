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
    """Structured output for event/topic labeling"""
    label: str = Field(..., description="A concise, descriptive label for the main event or topic")

class EventSummary(BaseModel):
    """Structured output for individual event summary"""
    summary: str = Field(..., description="Clear and concise summary of the event")

class FinalSummary(BaseModel):
    """Structured output for the final document summary"""
    comprehensive_summary: str = Field(..., description="Overall summary of the entire document")

class HERASummarizer:
    def __init__(self, ollama_model: str = "gemma2:9b", max_workers: int = 8):
        self.llm = ChatOllama(model=ollama_model, temperature=0.0)
        self.max_workers = max_workers

        # LLM with structured output
        self.llm_label = self.llm.with_structured_output(EventLabel)
        self.llm_event_summary = self.llm.with_structured_output(EventSummary)
        self.llm_final_summary = self.llm.with_structured_output(FinalSummary)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Prompts (clean, strict instructions)
        self.event_grouping_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying the core topic of a text segment. "
                       "Return ONLY a short, precise label. No explanations, no extra text."),
            ("human", "Text: {text}\n\nEvent/Topic Label:")
        ])

        self.event_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Write a clear, concise, and coherent summary "
                       "of the following related text segments. Do not add meta-commentary."),
            ("human", "Related segments:\n{event_segments}\n\nSummary:")
        ])

        self.final_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at synthesizing high-level document summaries. "
                       "Produce a comprehensive, well-structured overview."),
            ("human", "Event summaries:\n{event_summaries}\n\nFinal Summary:")
        ])

    def split_into_segments(self, text: str) -> List[Document]:
        segments = self.text_splitter.split_text(text)
        return [Document(page_content=s) for s in segments]

    def label_segment(self, segment: Document) -> tuple[str, Document]:
        """Thread-safe labeling of one segment"""
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

    def summarize_single_event(self, args: tuple[str, List[Document]]) -> str:
        """Summarize one event (used in parallel)"""
        label, segments = args
        combined = "\n\n".join([s.page_content for s in segments])
        chain = self.event_summary_prompt | self.llm_event_summary
        result = chain.invoke({"event_segments": combined})
        return f"Event: {label}\n{result.summary.strip()}"

    def summarize_document(self, text: str) -> str:
        print("Step 1: Splitting document into segments...")
        segments = self.split_into_segments(text)
        print(f"Created {len(segments)} segments")

        # Step 2: Parallel event labeling + grouping
        event_groups = self.group_segments_into_events(segments)

        # Step 3: Parallel event summarization
        print("Step 3: Summarizing each event...")
        event_items = list(event_groups.items())
        event_summaries: List[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.summarize_single_event, item) for item in event_items]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing events"):
                event_summaries.append(future.result())

        # Step 4: Final comprehensive summary
        print("Step 4: Generating final comprehensive summary...")
        combined_events = "\n\n".join(event_summaries)
        final_chain = self.final_summary_prompt | self.llm_final_summary
        final_result = final_chain.invoke({"event_summaries": combined_events})

        structured_output = f"""HERA Document Summary
================================================================================
Event-Based Summaries:
{"\n\n".join(event_summaries)}

================================================================================
Comprehensive Document Summary:
{final_result.comprehensive_summary.strip()}
"""
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
    pdf_path = "research.pdf"  # Change to your PDF

    try:
        result = load_and_summarize_pdf(
            pdf_path=pdf_path,
            ollama_model="gemma2:9b",   # or "llama3.1:8b", "mistral", etc.
            max_workers=12              # Adjust based on your CPU/RAM
        )

        print("\n" + "="*80)
        print("FINAL HERA SUMMARY")
        print("="*80)
        print(result)

        # Optional: save to file
        with open("hera_summary.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("\nSummary saved to hera_summary.txt")

    except Exception as e:
        print(f"Error: {e}")