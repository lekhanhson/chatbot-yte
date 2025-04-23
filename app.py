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
    """Trích xuất từng tình huống từ file PDF theo kiểu định dạng"""
    doc = fitz.open(path)
    full_text = "\n".join([page.get_text() for page in doc])
    pattern = r"\n\d{1,2}\.\s" if not is_emergency else r"🧪 Tình huống khẩn cấp \d{1,2}:"
    parts = re.split(pattern, full_text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

# --- Load dữ liệu từ hai file ---
emergency_scenarios = extract_scenarios_from_pdf("tinh_huong_khan_cap.pdf", is_emergency=True)
communication_scenarios = extract_scenarios_from_pdf("tinh_huong_giao_tiep.pdf", is_emergency=False)

# --- Xử lý nội dung để hiển thị cho người dùng ---
def extract_visible_emergency(scenario):
    lines = scenario.split("\n")
    title = lines[0] if lines else "Tình huống khẩn cấp"
    desc = ""
    actions = []
    mode = "desc"
    for line in lines[1:]:
        if line.lower().startswith("cần thực hiện"):
            mode = "actions"
            continue
        if mode == "desc":
            desc += line.strip() + " "
        elif mode == "actions" and line.strip():
            actions.append("- " + line.strip())
    return f"🧪 {title}\n\n📍 Mô tả: {desc.strip()}\n\n✅ Cần thực hiện:\n" + "\n".join(actions)

def extract_visible_communication(scenario):
    parts = scenario.split("Đáp án:")
    question = parts[0].strip()
    answer = parts[1].strip() if len(parts) > 1 else ""
    actions = re.findall(r"•\s+(.*)", answer)
    formatted = "\n".join(f"- {a}" for a in actions)
    return f"💬 {question}\n\n✅ Cách xử lý đề xuất:\n{formatted}"

# --- Gửi prompt GPT để đánh giá phản hồi của người dùng ---
def analyze_response(user_answer, scenario_text, mode):
    prompt = f"""
Bạn là trợ lý đào tạo điều dưỡng. Hãy đánh giá phản hồi của học viên dựa trên tình huống và đưa ra nhận xét theo 4 mục:
1. Câu trả lời có phù hợp không?
2. Nếu chưa đúng thì sai ở đâu?
3. Gợi ý và lưu ý thêm cho học viên
4. Đánh giá mức độ: X sao (dùng ký hiệu ⭐ từ 1 đến 5)

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

# --- Trích xuất số sao từ phản hồi của GPT ---
def extract_star_rating(feedback_text):
    star_match = re.search(r"(\d)\s*sao", feedback_text.lower())
    if star_match:
        num = int(star_match.group(1))
        return "⭐" * min(max(num, 1), 5)
    return "⭐"

# --- Trạng thái người dùng ---
user_states = {}

# --- Hàm chính xử lý tin nhắn ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_text = update.message.text.strip()
    lowered_text = message_text.lower()

    greetings = ["hi", "hello", "xin chào", "chào", "alo", "yo", "chao", "2", "/start"]

    # Tạo trạng thái mới nếu lần đầu tương tác
    if user_id not in user_states:
        user_states[user_id] = {"mode": "emergency", "status": "idle"}

    state = user_states[user_id]

    # Gửi tình huống đầu tiên nếu ở trạng thái idle
    if state["status"] == "idle":
        mode = state["mode"]
        scenario = random.choice(emergency_scenarios) if mode == "emergency" else random.choice(communication_scenarios)
        text = extract_visible_emergency(scenario) if mode == "emergency" else extract_visible_communication(scenario)

        if lowered_text in greetings:
            await update.message.reply_text("👋 Xin chào! Tôi là TRỢ LÝ AI [BV Lâm Hoa]. \n\nChúng ta sẽ cùng luyện phản xạ tình huống điều dưỡng. Hãy bắt đầu với tình huống đầu tiên nhé!")
            await asyncio.sleep(1)

        await update.message.reply_text(f"📌 Đây là tình huống {'KHẨN CẤP' if mode == 'emergency' else 'GIAO TIẾP'} – hãy đưa ra xử lý phù hợp.\n\n{text}")
        user_states[user_id] = {"mode": mode, "status": "awaiting_response", "scenario": scenario}
        return

    # Phân tích phản hồi nếu đang chờ trả lời
    if state["status"] == "awaiting_response":
        scenario = state["scenario"]
        mode = state["mode"]

        # Gửi phản hồi đến GPT và trích sao
        feedback = analyze_response(message_text, scenario, mode)
        stars = extract_star_rating(feedback)

        # Phản hồi lại học viên
        await update.message.reply_text(f"📋 Đánh giá từ trợ lý: {stars}\n\n{feedback}")

        # Tạo tình huống tiếp theo (luân phiên)
        next_mode = "communication" if mode == "emergency" else "emergency"
        next_scenario = random.choice(emergency_scenarios) if next_mode == "emergency" else random.choice(communication_scenarios)
        next_text = extract_visible_emergency(next_scenario) if next_mode == "emergency" else extract_visible_communication(next_scenario)

        await update.message.reply_text(f"🔄 Nào, tiếp tục với tình huống {'KHẨN CẤP' if next_mode == 'emergency' else 'GIAO TIẾP'} tiếp theo nhé:\n\n{next_text}")
        user_states[user_id] = {"mode": next_mode, "status": "awaiting_response", "scenario": next_scenario}
        return

# --- Web UI nếu có ---
@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --- Chạy Flask song song với Telegram bot ---
def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# --- Chạy bot Telegram ---
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
