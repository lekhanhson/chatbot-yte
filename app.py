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
import re

# --- Flask App ---
flask_app = Flask(__name__)

# --- OpenAI API ---
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-turbo"

# --- Tách từng tình huống theo số thứ tự ---
def extract_cases_by_structure(path):
    doc = fitz.open(path)
    full_text = "\n".join([page.get_text() for page in doc])

    pattern = r'\n\d{1,2}\.\s'  # Tìm đầu các tình huống: 1. 2. 3.
    parts = re.split(pattern, full_text)
    headers = re.findall(r'\n\d{1,2}\.\s', full_text)

    cases = []
    for i, part in enumerate(parts[1:], start=1):
        header = headers[i-1].strip()
        case_text = f"{header} {part}".strip()
        cases.append(case_text)
    return cases

# --- Load dữ liệu ---
chunks = extract_cases_by_structure("tinh_huong_khan_cap.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# --- Chọn 1 tình huống bất kỳ ---
def pick_random_scenario():
    return random.choice(chunks)

# --- Tìm các đoạn liên quan nhất đến câu trả lời ---
def search_relevant_chunks(text, top_n=3):
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, chunk_vectors).flatten()
    top_ids = sims.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_ids]

# --- Phân tích câu trả lời của học viên ---
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

# --- Chỉ hiện phần mô tả, không hiện cách xử lý ---
def extract_visible_part(scenario_text):
    cutoff = "Xử lý tại chỗ"
    parts = scenario_text.split(cutoff)
    return parts[0].strip() + "\n\n👉 Bạn sẽ xử lý thế nào trong 3 phút đầu tiên?"

# --- Giao tiếp Telegram ---
user_states = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_text = update.message.text.strip()
    lowered_text = message_text.lower()

    greetings = ["hi", "hello", "xin chào", "chào", "alo", "yo"]

    if user_id not in user_states or user_states[user_id]["status"] == "idle":
        scenario = pick_random_scenario()
        user_states[user_id] = {"status": "awaiting_response", "scenario": scenario}
        scenario_number = chunks.index(scenario) + 1
        visible = extract_visible_part(scenario)

        if lowered_text in greetings:
            await update.message.reply_text(
                "👋 Xin chào! Tôi là **Trợ lý Hội Nhập Điều Dưỡng**, nhiệm vụ của tôi là hỗ trợ bạn luyện phản xạ trong các tình huống khẩn cấp thực tế.\n\n"
                "Bây giờ, hãy bắt đầu với một tình huống đầu tiên nhé:"
            )
        else:
            await update.message.reply_text("🔔 Bắt đầu kiểm tra tình huống khẩn cấp đầu tiên:")

        await update.message.reply_text(f"🧪 Tình huống {scenario_number:02d}:\n\n{visible}")
        return

    if user_states[user_id]["status"] == "awaiting_response":
        scenario = user_states[user_id]["scenario"]
        feedback = analyze_response(message_text, scenario)
        await update.message.reply_text(f"📋 Đánh giá từ trợ lý:\n\n{feedback}")
    
        # ➕ Gợi ý tiếp tục
        await update.message.reply_text("🔄 Nào, thêm một tình huống tiếp theo nhé:")
    
        # Gửi tiếp tình huống mới
        next_scenario = pick_random_scenario()
        scenario_number = chunks.index(next_scenario) + 1
        visible = extract_visible_part(next_scenario)
    
        await update.message.reply_text(f"🧪 Tình huống {scenario_number:02d}:\n\n{visible}")
        user_states[user_id] = {"status": "awaiting_response", "scenario": next_scenario}
        return

# --- Web UI ---
@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# --- Telegram Bot ---
def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("⚠️ TELEGRAM_TOKEN chưa được thiết lập!")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
