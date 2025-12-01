from download_and_parse import FileParser
from language_detection import LanguageDetection
import pandas as pd
import os

# Initialize an empty dataframe to collect all text chunks
df = pd.DataFrame(columns=['text', 'type'])

class Agent:
    def __init__(self, file_path: str):
        self.parser = FileParser(file_path_or_url=file_path)
        self.detector = LanguageDetection()
    
    def Start_Process(self, document_type: str):
        try:
            text = self.parser.parse_text()
            language = self.detector.detect(text=text)

            if text:

                if language == "arabic":
                    print(f"Arabic language detected , trying to convert to engish ...")
                    text = self.detector.convert_arabic_to_english(arabic_text=text)

            
                # Check if the text length is greater than 1000 characters
                if len(text) > self.parser.max_chunk_size:
                    print(f"Text length ({len(text)} characters) exceeds {self.parser.max_chunk_size} characters. Splitting into chunks.")
                    # Split the text using the parser's split_text function
                    splitted_text = self.parser.split_text(text=text)
                    print(f"Text successfully split into {len(splitted_text)} chunks.")
                else:
                    # If text is 1000 characters or less, use it as a single chunk
                    print(f"Text length ({len(text)} characters) is 1000 characters or less. No splitting required.")
                    splitted_text = [text]
                
                # Add each chunk to the dataframe
                for chunk_text in splitted_text:
                    # Only add non-empty chunks
                    if chunk_text.strip():
                        temp_df = pd.DataFrame({
                            "text": [chunk_text.strip()],
                            "type": [document_type]
                        })
                        # Append to the global dataframe
                        global df
                        df = pd.concat([df, temp_df], ignore_index=True)
                        print(f"Added chunk of {len(chunk_text)} characters for document type: {document_type}")
            else:
                print("No text was extracted from the document.")
        except Exception as e:
            print(f"Error processing document: {e}")

def process_documents():
    """Helper function to process all documents and save the combined dataframe"""
    global df
    
    # Process EA standard files
    ea_dir = os.path.join("Dataset", "EA_standards")
    if os.path.exists(ea_dir):
        ea_standard_files = [os.path.join(ea_dir, file) for file in os.listdir(ea_dir) 
                           if os.path.isfile(os.path.join(ea_dir, file))]
        print(f"Processing {len(ea_standard_files)} EA standard files...")
        
        for file in ea_standard_files:
            try:
                agent = Agent(file_path=file)
                agent.Start_Process(document_type="EA")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    # Process Proposal files
    proposal_dir = os.path.join("Dataset", "Proposal")
    if os.path.exists(proposal_dir):
        proposal_files = [os.path.join(proposal_dir, file) for file in os.listdir(proposal_dir) 
                        if os.path.isfile(os.path.join(proposal_dir, file))]
        print(f"Processing {len(proposal_files)} proposal files...")
        
        for file in proposal_files:
            try:
                agent = Agent(file_path=file)
                agent.Start_Process(document_type="proposal")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    # Process RFP files
    rfp_dir = os.path.join("Dataset", "RFP")
    if os.path.exists(rfp_dir):
        rfp_files = [os.path.join(rfp_dir, file) for file in os.listdir(rfp_dir) 
                   if os.path.isfile(os.path.join(rfp_dir, file))]
        print(f"Processing {len(rfp_files)} RFP files...")
        
        for file in rfp_files:
            try:
                agent = Agent(file_path=file)
                agent.Start_Process(document_type="proposal")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    return df

if __name__ == "__main__":
    # Process all documents
    final_df = process_documents()
    
    # Save the combined dataframe
    output_path = "Dataset/combined_data.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\nProcessing complete. Total chunks saved: {len(final_df)}")
    print(f"Data saved to: {output_path}")
    print(f"Breakdown by document type:")
    print(final_df['type'].value_counts())
    
    # Optional: Print some statistics
    total_characters = final_df['text'].str.len().sum()
    avg_chunk_size = final_df['text'].str.len().mean()
    print(f"Total characters across all chunks: {total_characters:,}")
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")