import streamlit as st
from huggingface_hub import InferenceClient

st.subheader("🤖 StudyPal Chatbot – Ask Anything From Your Text")

API_KEY = st.secrets.get("HF_API_TOKEN") or "YOUR_HF_API_KEY"
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=API_KEY)

context = st.session_state.get("context", "")

if context:
    st.info("Context loaded from uploaded document.")
else:
    st.warning("No context loaded. Please upload a file in the main app first.")

user_input = st.chat_input("Ask a question about the uploaded content...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking..."):
        prompt = f"Answer this question based on the following context:\n\n{context}\n\nQ: {user_input}\nA:"
        response = client.text_generation(prompt, max_new_tokens=200, temperature=0.7)

    with st.chat_message("assistant"):
        st.success(response.strip())
