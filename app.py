import os
import random
import fitz  # PyMuPDF
import openai
from flask import Flask, render_template
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import threading
import re
import asyncio

# --- Flask App ---
flask_app = Flask(__name__)

# --- OpenAI API ---
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-turbo"

# --- Load và xử lý PDF ---
def extract_scenarios_from_pdf(path, is_emergency=True):
    """Trích xuất từng tình huống từ file PDF theo định dạng 'Tình huống khẩn cấp' hoặc 'Tình huống giao tiếp'"""
    doc = fitz.open(path)
    full_text = "\n".join([page.get_text() for page in doc])
    pattern = r"Tình huống khẩn cấp\s+\d{1,2}:" if is_emergency else r"Tình huống giao tiếp\s+\d+"
    parts = re.split(pattern, full_text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

# --- Load dữ liệu từ hai file ---
emergency_scenarios = extract_scenarios_from_pdf("tinh_huong_khan_cap.pdf", is_emergency=True)
communication_scenarios = extract_scenarios_from_pdf("tinh_huong_giao_tiep.pdf", is_emergency=False)

# --- Hiển thị tình huống khẩn cấp ---
def extract_visible_emergency(scenario):
    lines = scenario.split("\n")
    title = lines[0] if lines else "Tình huống khẩn cấp"
    desc = ""
    for line in lines[1:]:
        if line.lower().startswith("cần thực hiện"):
            break
        desc += line.strip() + " "
    return f"{title} \n{desc.strip()}\n\n 💗 Bạn sẽ xử lý thế nào?"

# --- Hiển thị tình huống giao tiếp ---
def extract_visible_communication(scenario):
    parts = scenario.split("Đáp án:")
    question = parts[0].strip()
    return f"{question}\n\n 💗 Bạn sẽ xử lý thế nào?"

# --- Phân tích phản hồi từ người dùng bằng GPT ---
def analyze_response(user_answer, scenario_text, mode):
    prompt = f"""
Bạn là trợ lý đào tạo điều dưỡng. Hãy đánh giá phản hồi của học viên dựa trên tình huống và đưa ra nhận xét ngắn gọn theo 3 mục:
1. Câu trả lời (có phù hợp không)
2. Gợi ý và lưu ý (ngắn gọn)
3. Đánh giá mức độ: X sao (dùng ký hiệu ⭐ từ 1 đến 5)

---
📌 Tình huống:
{scenario_text}

✍️ Phản hồi của học viên:
{user_answer}
"""
    res = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return res.choices[0].message.content.strip()

# --- Tách số sao từ phản hồi GPT ---
def extract_star_rating(feedback_text):
    star_match = re.search(r"(\d)\s*sao", feedback_text.lower())
    if star_match:
        num = int(star_match.group(1))
        return "⭐" * min(max(num, 1), 5)
    return "⭐"

# --- Quản lý trạng thái người dùng ---
user_states = {}

# --- Xử lý tin nhắn Telegram ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_text = update.message.text.strip()
    lowered_text = message_text.lower()

    greetings = ["hi", "hello", "xin chào", "chào", "alo", "yo", "chao", "2", "/start"]

    if user_id not in user_states:
        user_states[user_id] = {"mode": "emergency", "status": "idle"}

    state = user_states[user_id]

    if state["status"] == "idle":
        mode = state["mode"]
        scenario = random.choice(emergency_scenarios) if mode == "emergency" else random.choice(communication_scenarios)
        text = extract_visible_emergency(scenario) if mode == "emergency" else extract_visible_communication(scenario)

        if lowered_text in greetings:
            await update.message.reply_text("👋 Xin chào! Tôi là TRỢ LÝ AI [BV Lâm Hoa].\n\nChúng ta sẽ cùng luyện phản xạ tình huống điều dưỡng.\nBắt đầu với tình huống đầu tiên nhé!")
            await asyncio.sleep(1)

        await update.message.reply_text(f"{'🔥 Đây là tình huống KHẨN CẤP' if mode == 'emergency' else '💬 Đây là tình huống GIAO TIẾP'}\n\n{text}")
        user_states[user_id] = {"mode": mode, "status": "awaiting_response", "scenario": scenario}
        return

    if state["status"] == "awaiting_response":
        scenario = state["scenario"]
        mode = state["mode"]

        feedback = analyze_response(message_text, scenario, mode)
        stars = extract_star_rating(feedback)

        await update.message.reply_text(f"📋 NHẬN XÉT TỪ TRỢ LÝ AI:\n\n{feedback}")

        next_mode = "communication" if mode == "emergency" else "emergency"
        next_scenario = random.choice(emergency_scenarios) if next_mode == "emergency" else random.choice(communication_scenarios)
        next_text = extract_visible_emergency(next_scenario) if next_mode == "emergency" else extract_visible_communication(next_scenario)

        await update.message.reply_text(f"{'🔥 Tiếp tục với tình huống KHẨN CẤP' if next_mode == 'emergency' else '💬 Tiếp tục với tình huống GIAO TIẾP'} \n\n{next_text}")
        user_states[user_id] = {"mode": next_mode, "status": "awaiting_response", "scenario": next_scenario}
        return

# --- Web UI đơn giản ---
@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --- Khởi động Flask song song với bot ---
def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# --- Khởi động bot Telegram ---
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
