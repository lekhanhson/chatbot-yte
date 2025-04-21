import os
import fitz  # PyMuPDF
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Flask UI đơn giản để báo bot đang chạy ---
import threading
from flask import Flask

flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "Bot đang chạy!"

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# --- Mô hình HuggingFace miễn phí ---
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# --- Trích xuất nội dung từ PDF ---
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

# --- Đọc file PDF khi khởi chạy ---
PDF_PATH = "huong_dan_chan_doan.pdf"
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"Không tìm thấy file PDF: {PDF_PATH}")

chunks = extract_pdf_chunks(PDF_PATH)
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# --- Tìm đoạn văn phù hợp nhất ---
def search_best_chunk(question):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)
    best_idx = scores[0].argmax()
    return chunks[best_idx]

# --- Xử lý tin nhắn từ người dùng ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    context_chunk = search_best_chunk(user_question)
    prompt = f"Câu hỏi: {user_question}\n\nDữ liệu hướng dẫn: {context_chunk}\n\nTrả lời:"
    try:
        result = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
        await update.message.reply_text(result.strip())
    except Exception as e:
        await update.message.reply_text(f"Lỗi bot: {str(e)}")

# --- Hàm chạy Telegram Bot ---
def main():
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("❌ Chưa khai báo TELEGRAM_TOKEN trong biến môi trường.")
        return

    print("✅ Bot Telegram đang khởi động...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🚀 Polling Telegram...")
    app.run_polling()

# --- Chạy Flask + Telegram song song ---
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
