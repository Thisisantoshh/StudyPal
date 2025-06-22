import streamlit as st
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import heapq
import string
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import base64

nltk.download("punkt")
nltk.download("stopwords")

st.set_page_config(page_title="🧠 StudyPal – AI-Powered Learning Assistant", layout="wide")
st.markdown("""
<style>
body {
    background-color: #eef2f5;
}
[data-testid="stSidebar"] {
    background-image: linear-gradient(to top, #667eea, #764ba2);
    color: white;
}
[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0);
}
.reportview-container .markdown-text-container {
    font-family: 'Segoe UI', sans-serif;
    font-size: 16px;
    color: #333333;
}
.big-font {
    font-size: 22px !important;
    font-weight: 600;
    color: #444;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=100)
st.sidebar.title("StudyPal AI")
st.sidebar.markdown("Your Personal AI-Powered Learning Assistant")
st.sidebar.info("✨ Built with Streamlit + NLP + Visualizations")

st.title("📚 StudyPal – Smart Document Analyzer & Learning Companion")
st.markdown("""
Welcome to **StudyPal** — your all-in-one educational toolkit! 🎓

🔍 Features:
- Upload and read PDF/TXT documents
- Generate summaries, word clouds & keyword analytics
- Detect emails/URLs
- View sentence structure patterns
- Export results to text
""")

uploaded_file = st.file_uploader("📤 Upload your PDF or Text file", type=["pdf", "txt"])

text = ""
if uploaded_file:
    file_details = {"filename": uploaded_file.name, "type": uploaded_file.type}
    st.success(f"Uploaded File: {file_details['filename']}")

    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    else:
        text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Document Preview")
    st.text_area("Extracted Text:", text, height=300)
    st.session_state["context"] = text

    tab1, tab2, tab3, tab4 = st.tabs(["📑 Summary", "📊 Analytics", "☁️ Word Cloud", "📤 Export"])

    with tab1:
        st.subheader("📝 Text Summarization")
        if st.button("Generate Summary"):
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words("english"))

            word_frequencies = {}
            for word in words:
                if word not in stop_words and word.isalpha():
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1

            max_freq = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] /= max_freq

            sentence_scores = {}
            for sent in sentences:
                for word in word_tokenize(sent.lower()):
                    if word in word_frequencies:
                        if len(sent.split(" ")) < 30:
                            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

            summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
            summary = " ".join(summary_sentences)
            st.success(summary)

    with tab2:
        st.subheader("📊 Keyword Frequency Analysis")
        words = word_tokenize(text.lower())
        clean_words = [word for word in words if word.isalpha() and word not in stopwords.words("english")]
        freq_dist = Counter(clean_words)
        top_words = freq_dist.most_common(10)

        labels, counts = zip(*top_words)
        fig1, ax1 = plt.subplots()
        ax1.bar(labels, counts, color='teal')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.subheader("📈 Sentence Length Histogram")
        sentences = sent_tokenize(text)
        sent_lengths = [len(sent.split()) for sent in sentences]
        fig2, ax2 = plt.subplots()
        ax2.hist(sent_lengths, bins=10, color='coral', edgecolor='black')
        ax2.set_xlabel("Words per Sentence")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        st.subheader("🔍 Emails & URLs Found")
        emails = re.findall(r"[\w\.-]+@[\w\.-]+", text)
        urls = re.findall(r"https?://\S+", text)

        if emails:
            st.markdown("**📧 Emails**")
            for email in emails:
                st.code(email)
        else:
            st.info("No emails found.")

        if urls:
            st.markdown("**🔗 URLs**")
            for url in urls:
                st.code(url)
        else:
            st.info("No URLs found.")

    with tab3:
        st.subheader("☁️ Word Cloud Generator")
        clean_text = ' '.join(clean_words)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_text)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis("off")
        st.pyplot(fig3)

    with tab4:
        st.subheader("📥 Export Summary")
        if "summary" in locals():
            b64 = base64.b64encode(summary.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="summary.txt">📄 Download Summary</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Please generate a summary first.")
