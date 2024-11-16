from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Generate text
prompt = "Write a poem about a lonely robot."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2)

print(tokenizer.decode(output[0], skip_special_tokens=True))