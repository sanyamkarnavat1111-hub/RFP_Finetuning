from download_and_parse import FileParser
# from language_detection_and_translation import LanguageDetection
import pandas as pd
import os
from hera_hdbscan_summarizer import HERASummarizer
import sys


# Initialize an empty dataframe to collect all text chunks
df = pd.DataFrame(columns=['text', 'type'])

class FineTuneAgent:
    def __init__(self, input_file_path : str , check_against_file_path : str ):

        try:

            if not input_file_path:
                raise ValueError(f"Input file path not provided to agent ...")

            if not check_against_file_path :
                raise ValueError(f"Check against file path not provided to agent ...")

            self.input_file_path = input_file_path
            self.check_against_file_path = check_against_file_path

            self.parser = FileParser()
            self.hera_summarizer = HERASummarizer()

            #################################################################################################################
                        ### No need for language detection and translation , Hera summarize is already doing that ###
            
            # self.detector = LanguageDetection()
            # language = self.detector.detect(text=text)
            
            #################################################################################################################

        except Exception as e :
            print(f"Error Initializing th agent :- \n " , e)
            sys.exit(0)

        
    
    def Start_Process(self):
        try:
            input_file_text = self.parser.parse_text(file_path_or_url=self.input_file_path)
            check_against_file_text = self.parser.parse_text(file_path_or_url=self.check_against_file_path)


            if not input_file_text:
                raise ValueError(f"Not text was extracted from file :- {self.input_file_path}")
            
            if not check_against_file_text:
                raise ValueError(f"Not text was extracted from file :- {self.check_against_file_path}")
            

            input_file_summarized_text = self.hera_summarizer.summarize_document(text=input_file_text)
            check_against_summarized_text = self.hera_summarizer.summarize_document(text=check_against_file_text)
            generated_html_report = """"""
            total_assigned_score = int(0)

            # Now train the ML  model to iterate over html report to train on "met" , "partially-met" and "not-met"
            # So now in the html report besides "ea requirement" , "gap analysis" and "status" we will have "ml_prediction" as well which tries to predict "status"
                # File :- train_bert_status.py
                # Input required(For now ) :- EA requirement , gap analysis , RFP coverage and status
                # Exptected output will have :- EA requirement , gap analysis , RFP coverage , status and ML prediction of "status" 


            
            # Train the ML model for score prediction
                # File :-  train_bert_scoring.py 
                # Input required(For now) :- EA requirement , RFP coverage and Score
                # Exptected output will have :- EA requirement , RFP coverage , Score and predicted score


             
            # Take the Input file , check against file , updated generated html report with predictions for status and score and provide to our LLM
            #  File :- train_llm_unsloth.py 


        except Exception as e:
            print(f"Error processing document: {e}")


if __name__ == "__main__":

    INPUT_FILES_DIR = "Data_Files"
    EA_DIR = "EA_Standards"
    RFP_DIR = "RFP"

    ea_file_path = os.path.join( INPUT_FILES_DIR, EA_DIR ,"ea_requirement.docx")
    rfp_file_path = os.path.join( INPUT_FILES_DIR , RFP_DIR ,"rfp_proposal.pdf")


    agent = FineTuneAgent(
        input_file_path=ea_file_path,
        check_against_file_path=rfp_file_path
    )

    agent.Start_Process()


