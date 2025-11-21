import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd



# Path to your saved model
model_path = "./saved_model"


print("Loading the model ...")
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # set model to evaluation mode

print("Model loaded successfully ...")

def predict_status(ea_requirement, rfp_coverage, gap_analysis):
    # Combine text in the same way as training
    text = f"EA_Requirement: {ea_requirement} [SEP] " \
           f"RFP_Coverage: {rfp_coverage}" \
           f"Gap_analysis: {gap_analysis} [SEP]"

    # Tokenize
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class

df = pd.read_csv("extracted_file.csv")


for _ , row in df.iterrows():
    ea_req = row["EA_Requirement"]
    actual_status = row["Status"]
    rfp_cov = row["RFP_Coverage"]
    gap = row["Gap_analysis"]
    
    predicted_status = predict_status(ea_req, rfp_cov, gap)
    if predicted_status == 1:
        print(f"Actual status {"Partially Met" if actual_status == 1 else "Not Met"}")
        print("Prediction :- Partially Met")
        print("-"*100)
    else:
        print(f"Actual status {"Partially Met" if actual_status == 1 else "Not Met"}")
        print("Predicted :- Not Met")
        print("-"*100)
