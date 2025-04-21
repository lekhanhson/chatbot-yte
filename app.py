import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

app = Flask(__name__)

# Tải mô hình HuggingFace miễn phí
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Trích xuất nội dung từ PDF và chia nhỏ
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
    context_chunk = search_best_chunk(question)
    prompt = f"Câu hỏi: {question}\n\nDữ liệu hướng dẫn: {context_chunk}\n\nTrả lời:"
    result = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
    return result.strip()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    try:
        answer = generate_answer(question)
    except Exception as e:
        answer = f"Lỗi bot: {str(e)}"
    return render_template("index.html", answer=answer)

# TELEGRAM
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    try:
        answer = generate_answer(user_question)
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"Lỗi bot: {str(e)}")

def run_telegram():
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        print("⚠️ Thiếu TELEGRAM_TOKEN")
        return
    app_tg = ApplicationBuilder().token(token).build()
    app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_tg.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_telegram).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
