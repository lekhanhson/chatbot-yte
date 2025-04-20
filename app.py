import os
import fitz  # PyMuPDF
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# (Tuỳ chọn) Mở port để Render không báo lỗi (khi deploy dạng Web Service)
import threading
from flask import Flask
flask_app = Flask(__name__)
@flask_app.route("/")
def home():
    return "Bot đang chạy!"

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# Tải mô hình HuggingFace (miễn phí)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Hàm trích xuất nội dung PDF
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

# Đọc nội dung PDF
chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")

# Vector hóa để tìm kiếm
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# Hàm tìm đoạn văn phù hợp nhất
def search_best_chunk(question):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)
    best_idx = scores[0].argmax()
    return chunks[best_idx]

# Xử lý khi người dùng nhắn tin
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    context_chunk = search_best_chunk(user_question)
    prompt = f"Câu hỏi: {user_question}\n\nDữ liệu hướng dẫn: {context_chunk}\n\nTrả lời:"
    try:
        result = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
        await update.message.reply_text(result.strip())
    except Exception as e:
        await update.message.reply_text(f"Lỗi bot: {str(e)}")

# Hàm chính để chạy bot
def main():
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

# Chạy Flask (giả lập cổng) và Telegram song song
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
