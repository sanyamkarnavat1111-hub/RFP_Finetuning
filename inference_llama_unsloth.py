import time
libraries_load_start_time = time.perf_counter()
from unsloth import FastLanguageModel
libraries_load_end_time = time.perf_counter()

print("Time required to load libraries :-" , libraries_load_end_time - libraries_load_start_time)



start_time = time.perf_counter()


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="16_bit_trained",
    dtype=None,                   
    load_in_4bit=False, # Passing the load in 4 bit model is very essential when models loaded for inferencing            
)


model = FastLanguageModel.for_inference(model=model)


end_time = time.perf_counter()


print("Time required to load the model :-" , end_time-start_time)