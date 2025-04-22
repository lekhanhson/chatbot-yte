import os
import random
import fitz  # PyMuPDF
import openai
from flask import Flask, request, render_template
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

# --- Flask App ---
flask_app = Flask(__name__)

# --- OpenAI config ---
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-turbo"

# --- Load PDF và chia đoạn ---
def extract_pdf_chunks(path, chunk_size=500):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

chunks = extract_pdf_chunks("tinh_huong_khan_cap.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# --- Gợi ý 1 tình huống ngẫu nhiên từ tài liệu ---
def pick_random_scenario():
    scenarios = [chunk for chunk in chunks if "Mô tả triệu chứng ban đầu" in chunk]
    return random.choice(scenarios)

# --- Tìm đoạn liên quan để đánh giá trả lời ---
def search_top_chunks(answer_text, top_n=3):
    vec = vectorizer.transform([answer_text])
    sims = cosine_similarity(vec, chunk_vectors).flatten()
    top_ids = sims.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_ids]

# --- Trạng thái hội thoại (phiên bản đơn giản) ---
user_states = {}

# --- Xử lý trả lời của điều dưỡng ---
def evaluate_response(user_input, expected_context):
    context_chunks = search_top_chunks(expected_context)
    prompt = f"""
Bạn là một trợ lý đào tạo điều dưỡng, có nhiệm vụ đánh giá phản hồi của học viên về cách xử lý tình huống khẩn cấp. Dưới đây là:
- Tình huống khẩn cấp đã đưa ra
- Câu trả lời của điều dưỡng viên
- Tài liệu hướng dẫn nội bộ liên quan

Hãy phân tích:
1. Câu trả lời có phù hợp không?
2. Nếu đúng thì vì sao đúng? Nếu chưa đúng thì thiếu gì?
3. Gợi ý thêm hoặc lưu ý đặc biệt

---  
📌 **Tình huống:**  
{expected_context}

✏️ **Câu trả lời của học viên:**  
{user_input}

📚 **Tài liệu nội bộ:**  
1. {context_chunks[0]}
2. {context_chunks[1]}
3. {context_chunks[2]}
---
Trả lời:
"""
    res = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return res.choices[0].message.content.strip()

# --- Telegram Bot ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    msg = update.message.text

    if user_id not in user_states or user_states[user_id]["mode"] == "idle":
        scenario = pick_random_scenario()
        user_states[user_id] = {"mode": "awaiting_answer", "scenario": scenario}
        await update.message.reply_text(f"🧪 Tình huống khẩn cấp:\n\n{scenario}\n\nHãy mô tả cách bạn sẽ xử lý.")
    else:
        scenario = user_states[user_id]["scenario"]
        evaluation = evaluate_response(msg, scenario)
        await update.message.reply_text(f"📋 Đánh giá:\n\n{evaluation}")
        user_states[user_id]["mode"] = "idle"

# --- Web UI đơn giản (tùy chọn) ---
@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --- Start Flask ---
def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# --- Start Telegram ---
def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("Missing TELEGRAM_TOKEN in environment!")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
