import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import openai
import threading

# Khởi tạo Flask app
flask_app = Flask(__name__)

# Cấu hình API Key cho OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Đọc và chia nhỏ file PDF
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")

# Tìm đoạn văn chứa nội dung liên quan
def search_best_chunk(question):
    return max(chunks, key=lambda chunk: question.lower() in chunk.lower())

# Gọi OpenAI để sinh câu trả lời
def generate_answer(question):
    context = search_best_chunk(question)
    messages = [
        {"role": "system", "content": "Bạn là trợ lý y tế dựa trên dữ liệu PDF."},
        {"role": "user", "content": f"Dữ liệu tham khảo: {context}\n\nCâu hỏi: {question}" },
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
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

# Bot Telegram
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

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
