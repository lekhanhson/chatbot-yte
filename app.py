import os
import random
import fitz  # PyMuPDF
import openai
from flask import Flask, render_template
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

# --- Load và chia nhỏ PDF ---
def extract_pdf_chunks(path, chunk_size=500):
    doc = fitz.open(path)
    full_text = " ".join([page.get_text() for page in doc])
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

chunks = extract_pdf_chunks("50_tinh_huong_cap_cuu.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# --- Chọn 1 tình huống khẩn cấp ngẫu nhiên ---
def pick_random_scenario():
    candidates = [chunk for chunk in chunks if "Mô tả triệu chứng ban đầu" in chunk]
    return random.choice(candidates)

# --- Tìm các đoạn liên quan đến câu trả lời ---
def search_relevant_chunks(text, top_n=3):
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, chunk_vectors).flatten()
    top_ids = sims.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_ids]

# --- Phân tích câu trả lời từ người dùng ---
def analyze_response(user_answer, scenario_text):
    context_chunks = search_relevant_chunks(scenario_text)
    prompt = f"""
Bạn là trợ lý đào tạo điều dưỡng. Hãy đánh giá phản hồi của học viên dựa trên tình huống khẩn cấp và tài liệu hướng dẫn. Hãy phân tích:
1. Câu trả lời có phù hợp không?
2. Nếu chưa đúng thì sai ở đâu?
3. Gợi ý và lưu ý thêm cho học viên.

---  
📌 Tình huống:
{scenario_text}

✏️ Phản hồi của học viên:
{user_answer}

📚 Tài liệu nội bộ:
1. {context_chunks[0]}
2. {context_chunks[1]}
3. {context_chunks[2]}

Trả lời:
"""
    res = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return res.choices[0].message.content.strip()

# --- Lưu trạng thái hội thoại của từng người dùng ---
user_states = {}

# --- Bot Telegram chính ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_text = update.message.text.strip()
    lowered_text = message_text.lower()

    greetings = ["hi", "hello", "xin chào", "chào", "alo", "yo"]

    # Nếu chưa có trạng thái hoặc đã kết thúc vòng trước
    if user_id not in user_states or user_states[user_id]["status"] == "idle":
        scenario = pick_random_scenario()
        user_states[user_id] = {"status": "awaiting_response", "scenario": scenario}

        if lowered_text in greetings:
            await update.message.reply_text(
                "👋 Xin chào! Tôi là **Trợ lý Hội Nhập Điều Dưỡng**, nhiệm vụ của tôi là hỗ trợ bạn luyện phản xạ trong các tình huống khẩn cấp thực tế.\n\n"
                "Bây giờ, hãy bắt đầu với một tình huống đầu tiên nhé:"
            )
        else:
            await update.message.reply_text("🔔 Bắt đầu kiểm tra tình huống khẩn cấp đầu tiên:")

        await update.message.reply_text(f"🧪 Tình huống:\n\n{scenario}\n\n👉 Bạn sẽ xử lý thế nào trong 3 phút đầu tiên?")
        return

    # Nếu đang chờ người dùng phản hồi
    if user_states[user_id]["status"] == "awaiting_response":
        scenario = user_states[user_id]["scenario"]
        feedback = analyze_response(message_text, scenario)

        await update.message.reply_text(f"📋 Đánh giá từ trợ lý:\n\n{feedback}")
        user_states[user_id]["status"] = "idle"
        return

# --- Giao diện web đơn giản ---
@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --- Chạy Flask ---
def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# --- Khởi động Telegram bot ---
def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("⚠️ TELEGRAM_TOKEN chưa được thiết lập trong môi trường!")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

# --- Chạy song song Flask và Telegram ---
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
