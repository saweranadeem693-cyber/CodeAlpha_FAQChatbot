# CodeAlpha_FAQChatbot
CodeAlpha AI Internship – **Task 2: Chatbot for FAQs**

## How it works
1. FAQs (question + answer) are stored in `faqs.json` (e-commerce store example).
2. **Preprocessing (NLTK):** lowercase, remove punctuation, tokenize, remove stop-words, lemmatize.
3. **Matching:** TF-IDF vectors (unigrams + bigrams) and **cosine similarity** between the user's question and every FAQ.
4. The best-matching answer is returned (with a fallback message below a similarity threshold).
5. Optional chat UI built with Streamlit.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py      # chat UI
python chatbot.py         # command-line version
```
Swap `faqs.json` with your own FAQs to adapt it to any product/topic.
