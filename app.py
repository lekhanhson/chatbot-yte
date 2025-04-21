import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import threading

# Init Flask
flask_app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load PDF content
def extract_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

chunks = extract_pdf_chunks("huong_dan_chan_doan.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

def search_top_chunks(question, top_k=3):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_answer(question):
    top_chunks = search_top_chunks(question)
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(top_chunks))
    prompt = f"""Trả lời câu hỏi dựa duy nhất vào tài liệu dưới đây. Nếu tài liệu không chứa thông tin phù hợp, hãy trả lời: 'Tài liệu không có thông tin liên quan.'
    
Tài liệu:
{context}

Câu hỏi: {question}

Trả lời:"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()




# Flask web UI
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

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
