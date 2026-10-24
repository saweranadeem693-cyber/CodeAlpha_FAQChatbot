"""Chat UI for the FAQ chatbot.  Run: streamlit run app.py"""
import streamlit as st
from chatbot import FAQChatbot

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 FAQ Chatbot")
st.caption("CodeAlpha AI Internship – Task 2")


@st.cache_resource
def load_bot():
    return FAQChatbot()


bot = load_bot()

with st.sidebar:
    st.header("Sample questions")
    for f in bot.faqs[:6]:
        st.write("•", f["question"])

if "history" not in st.session_state:
    st.session_state.history = [("assistant", "Hi! Ask me anything about orders, delivery, returns or payments.")]

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Type your question..."):
    answer, matched, score = bot.ask(q)
    st.session_state.history += [("user", q), ("assistant", answer)]
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(answer)
    if matched:
        st.caption(f"Matched FAQ: “{matched}” (similarity {score:.2f})")
