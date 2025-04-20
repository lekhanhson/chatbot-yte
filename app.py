from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import os

# 1. Tải model từ HuggingFace
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# 2. Tải nội dung từ file PDF
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")  # <-- đổi tên file tùy bạn

# 3. Vector hóa nội dung
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# 4. Hàm tìm đoạn liên quan nhất
def search_best_chunk(question):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)
    best_idx = scores[0].argmax()
    return chunks[best_idx]

# 5. Xử lý tin nhắn Telegram
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    context_text = search_best_chunk(query)
    prompt = f"Câu hỏi: {query}\n\nThông tin liên quan: {context_text}\n\nTrả lời:"
    try:
        answer = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
        await update.message.reply_text(answer.strip())
    except Exception as e:
        await update.message.reply_text(f"Bot gặp lỗi: {str(e)}")

# 6. Chạy bot
def main():
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
