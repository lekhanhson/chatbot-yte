from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import os

# Dùng HuggingFace model miễn phí
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Tách nội dung PDF thành đoạn nhỏ
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

# Tải nội dung file PDF bạn đã gửi
chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")

# Vector hóa nội dung
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# Tìm đoạn liên quan nhất đến câu hỏi
def search_best_chunk(question):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)
    best_idx = scores[0].argmax()
    return chunks[best_idx]

# Trả lời người dùng
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    context_chunk = search_best_chunk(user_question)
    prompt = f"Câu hỏi: {user_question}\n\nDữ liệu hướng dẫn: {context_chunk}\n\nTrả lời:"
    try:
        result = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
        await update.message.reply_text(result.strip())
    except Exception as e:
        await update.message.reply_text(f"Lỗi bot: {str(e)}")

def main():
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
