from typing import Literal
from parse_files import FileParser


Language = Literal["arabic", "english", "unknown"]

class LanguageDetection:
    def __init__(self):
        # Full Arabic Unicode ranges (covers 99.9% of real-world Arabic text in PDFs/DOCs)
        self.arabic_ranges = [
            range(0x0600, 0x06FF + 1),   # Arabic
            range(0x0750, 0x077F + 1),   # Arabic Supplement
            range(0x08A0, 0x08FF + 1),   # Arabic Extended-A
            range(0xFB50, 0xFDFF + 1),   # Arabic Presentation Forms-A
            range(0xFE70, 0xFEFF + 1),   # Arabic Presentation Forms-B
        ]
        # Basic Latin (English + common punctuation/symbols)
        self.latin_range = range(0x00, 0x07F + 1)

    def _is_arabic(self, char: str) -> bool:
        if not char:
            return False
        code = ord(char)
        return any(code in r for r in self.arabic_ranges)

    def _is_latin(self, char: str) -> bool:
        if not char:
            return False
        return ord(char) in self.latin_range

    def detect(self, text: str, threshold: float = 0.30) -> Language:
        """
        Detect language from extracted text.
        
        Args:
            text: Extracted text (from FileParser or anywhere)
            threshold: Min % of Arabic chars to classify as Arabic (default 30%)

        Returns:
            "arabic" | "english" | "unknown"
        """
        if not text or not text.strip():
            return "unknown"

        total_relevant = 0
        arabic_count = 0

        for char in text:
            if self._is_arabic(char):
                arabic_count += 1
                total_relevant += 1
            elif self._is_latin(char):
                total_relevant += 1
            # Ignore whitespace, numbers, symbols, CJK, etc.

        if total_relevant == 0:
            return "unknown"

        arabic_ratio = arabic_count / total_relevant

        # print(f"Language Detection → Arabic: {arabic_count}, Latin: {total_relevant - arabic_count}, "
        #       f"Ratio: {arabic_ratio:.2%}")

        if arabic_ratio >= threshold:
            return "arabic"
        else:
            return "english"

    def detect_with_confidence(self, text: str) -> dict:
        """Return detailed detection result with confidence."""
        if not text or not text.strip():
            return {"language": "unknown", "arabic_ratio": 0.0, "confidence": 0.0}

        total_relevant = 0
        arabic_count = 0

        for char in text:
            if self._is_arabic(char):
                arabic_count += 1
                total_relevant += 1
            elif self._is_latin(char):
                total_relevant += 1

        if total_relevant == 0:
            return {"language": "unknown", "arabic_ratio": 0.0, "confidence": 0.0}

        arabic_ratio = arabic_count / total_relevant
        language = "arabic" if arabic_ratio >= 0.30 else "english"
        confidence = abs(arabic_ratio - 0.5) * 2  # 0.0 to 1.0 (higher = more confident)

        return {
            "language": language,
            "arabic_ratio": round(arabic_ratio, 3),
            "arabic_chars": arabic_count,
            "latin_chars": total_relevant - arabic_count,
            "confidence": round(confidence, 3)
        }
    
if __name__ == "__main__":

    file_path = "Evaluation_Files/ea-standard-arabic.pdf"

    parser = FileParser(file_path=file_path)
    text = parser.get_text()

    if text:
        print(f"Successfully extracted {len(text):,} characters")
        print("First 500 characters:")
        print(text[:500].replace("\n", " ").strip())
        print("\n" + "="*50)

        # Optional: get chunks
        chunks = parser.load_and_split()
        print(f"Split into {len(chunks)} chunks")
        print("First chunk:", chunks[0][:200].replace("\n", " "))
    else:
        print("Failed to extract text.")


    detector = LanguageDetection()
    language = detector.detect(text)
    print(f"Detected Language: {language.upper()}")

    # Or get detailed result
    result = detector.detect_with_confidence(text)
    print("Detailed Result:", result)