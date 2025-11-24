import time
libraries_load_start_time = time.perf_counter()
from unsloth import FastLanguageModel
libraries_load_end_time = time.perf_counter()

print("Time required to load libraries :-" , libraries_load_end_time - libraries_load_start_time)

start_time = time.perf_counter()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="16_bit_trained",
    dtype=None,                   
    load_in_4bit=True, # Passing the load in 4 bit model is very essential when models loaded for inferencing            
    device_map="auto",
    max_memory={0: "80GB"}, ### Note passing an arbitarily large number than the capacity of GPU forces to use whatever memeory is available
)

print("Model loaded from pretrained...")

model = FastLanguageModel.for_inference(model=model)


end_time = time.perf_counter()


print("Time required to load the model :-" , end_time-start_time)




prompt = """### Enterprise Architecture (EA) Requirement:
(This is a specific requirement from the organization's enterprise architecture, describing technical, compliance, or process needs.)
{}

### RFP Coverage:
(This section describes how the Request for Proposal addresses the EA requirement, detailing proposed solutions, compliance, and coverage.)
{}

### Gap Analysis:
(This summarizes what analysis was done to compare the RFP with the EA requirement and identifies any gaps found.)
{}

## You need to understand the status and if not present analyze the EA requirements , RFP coverage and Gap analysis and the provide status.
Provide the status 
### Status
{} 

"""



EA_requirement = """Content
 Management
 System Setup """


RFP_coverage = """ The RFP does not
 mention content
 management system
 (CMS) setup as a
 deliverable."""


Gap_analysis = """The EA standard
 requires CMS
 setup as a key
 deliverable, but this
 is not addressed in
 the RFP content."""



inputs = tokenizer(
    [
        prompt.format(EA_requirement , RFP_coverage , Gap_analysis , "") # Keeping the status as blank
    ],
    
    return_tensors="pt"
    ).to("cuda")


outputs = model.generate(**inputs , max_new_tokens = 100 , use_cache=True)

result = tokenizer.batch_decode(outputs)

print(result)