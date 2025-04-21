import os
import fitz  # PyMuPDF
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
import threading

flask_app = Flask(__name__)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Trích xuất nội dung từ PDF thành đoạn nhỏ
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

def search_best_chunk(question):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)
    best_idx = scores[0].argmax()
    return chunks[best_idx]

def generate_answer(question):
    context = search_best_chunk(question)
    prompt = f"""Câu hỏi: {question}

Dữ liệu hướng dẫn: {context}

Trả lời:"""
    output = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
    return output.strip()

# Giao diện Web
history = []
@flask_app.route("/", methods=["GET", "POST"])
def index():
    global history
    if request.method == "POST":
        question = request.form["question"]
        answer = generate_answer(question)
        history.append((question, answer))
    return render_template("index.html", history=history)

# Giao diện Telegram
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    answer = generate_answer(question)
    await update.message.reply_text(answer)

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False, use_reloader=False)

def main():
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
