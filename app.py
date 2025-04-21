import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

# Init Flask
flask_app = Flask(__name__)

# Init OpenAI
client = OpenAI()  # API key lấy từ biến môi trường OPENAI_API_KEY

# Tải và chia nhỏ dữ liệu PDF
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# Tìm đoạn văn phù hợp nhất với câu hỏi
def search_best_chunk(question):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)
    best_idx = scores[0].argmax()
    return chunks[best_idx]

# Tạo câu trả lời bằng OpenAI ChatCompletion
def generate_answer(question):
    context = search_best_chunk(question)
    prompt = f"""Bạn là một chuyên gia y tế trả lời câu hỏi dựa trên tài liệu. Hãy trả lời ngắn gọn, chính xác và dễ hiểu.

Câu hỏi: {question}

Dữ liệu tham khảo: {context}

Trả lời:"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Hoặc gpt-4 nếu có quyền
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

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

# Telegram Bot
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    answer = generate_answer(question)
    await update.message.reply_text(answer)

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

def main():
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

# Chạy song song web + telegram
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
