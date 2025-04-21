import os
import fitz  # PyMuPDF
import openai
from flask import Flask, request, render_template
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

# --- Cấu hình Flask ---
flask_app = Flask(__name__)

# --- Cấu hình API OpenAI ---
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-turbo"

# --- Xử lý PDF ---
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# --- Tìm 3 đoạn liên quan nhất ---
def search_top_chunks(question, top_n=3):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

# --- Sinh câu trả lời từ GPT ---
def generate_answer(question):
    contexts = search_top_chunks(question)
    prompt = f"""Bạn là trợ lý y tế. Hãy trả lời chính xác và rõ ràng cho câu hỏi sau dựa vào tài liệu dưới đây. Nếu không thấy thì trả lời 'Trong tài liệu không có thông tin'.

Câu hỏi: {question}

Tài liệu tham khảo:
1. {contexts[0]}

2. {contexts[1]}

3. {contexts[2]}

Trả lời:"""

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# --- Web UI ---
history = []

@flask_app.route("/", methods=["GET", "POST"])
def index():
    global history
    if request.method == "POST":
        question = request.form["question"]
        answer = generate_answer(question)
        history.append((question, answer))
    return render_template("index.html", history=history)

# --- Telegram Bot ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    answer = generate_answer(question)
    await update.message.reply_text(answer)

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

def main():
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("Missing TELEGRAM_TOKEN in environment!")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
