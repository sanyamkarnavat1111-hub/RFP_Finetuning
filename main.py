from parse_files import FileParser
from language_detection import LanguageDetection





class Agent:

    def __init__(self , file_path : str):
        
        self.parser = FileParser(file_path=file_path)
        self.detector = LanguageDetection()

    
    def Start_Process(self):

        try:
            text = self.parser.get_text()
            print(text)

            if text:
                language = self.detector.detect_with_confidence(text)
                print(f"Detected Language: {language}")
            else:
                print("Failed to extract text.")
        except Exception as e :
            print(f"Error in the Pipeline :- {e}")



if __name__ == "__main__":
    file_path = "Evaluation_Files/ea-standard-arabic.pdf"

    # agent_1 = Agent(file_path="downloaded_file.pdf")
    # agent_2 = Agent(file_path_or_url="Evaluation_Files/ea-standard-english.pdf")
    # agent_3 = Agent(file_path_or_url="https://compliancebotai.blob.core.windows.net/compliancebotdev/rfp/document/67b5be9c749273GORz1739964060.pdf")
    agent_4 = Agent(file_path="Evaluation_Files/rfp-proposal-arabic-2.pdf")


    # agent_1.Start_Process()
    # print("-"*100)
    # agent_2.Start_Process()
    # print("-"*100)
    # agent_3.Start_Process()
    print("-"*100)
    agent_4.Start_Process()


