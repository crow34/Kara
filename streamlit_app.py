import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Define the available models and their names
models = {
    "GPT-3.5": "openai/gpt-3.5-turbo",
    "GPT-2": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
}

# Set up the selected model
def setup_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Generate response using the selected model
def generate_response(model_name, prompt, max_length):
    tokenizer, model = setup_model(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = generator(prompt, max_length=max_length)[0]['generated_text']
    return response

# Streamlit app
def main():
    st.title("ChatGPT Clone with Multiple Language Models")

    model_name = st.selectbox("Select a language model", list(models.keys()))

    prompt = st.text_area("Enter your message")

    if st.button("Send"):
        max_length = 100  # Maximum number of tokens in the generated response
        response = generate_response(models[model_name], prompt, max_length)
        st.text_area("Response", response)

if __name__ == "__main__":
    main()
