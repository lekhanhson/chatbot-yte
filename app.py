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
    doc = fitz.open(path)
    full_text = "\n".join([page.get_text() for page in doc])
    pattern = r"\n\d{1,2}\.\s" if not is_emergency else r"🧪 Tình huống khẩn cấp \d{1,2}:"
    parts = re.split(pattern, full_text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

# --- Load dữ liệu từ hai file ---
emergency_scenarios = extract_scenarios_from_pdf("tinh_huong_khan_cap.pdf", is_emergency=True)
communication_scenarios = extract_scenarios_from_pdf("tinh_huong_giao_tiep.pdf", is_emergency=False)

# --- Trích phần hiển thị phù hợp ---
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

# --- Giao tiếp Telegram ---
user_states = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_text = update.message.text.strip()
    lowered_text = message_text.lower()

    greetings = ["hi", "hello", "xin chào", "chào", "alo", "yo"]
    if user_id not in user_states:
        user_states[user_id] = {"mode": "emergency", "status": "idle"}

    if user_states[user_id]["status"] == "idle":
        mode = user_states[user_id]["mode"]
        scenario = random.choice(emergency_scenarios) if mode == "emergency" else random.choice(communication_scenarios)
        if mode == "emergency":
            text = extract_visible_emergency(scenario)
        else:
            text = extract_visible_communication(scenario)

        if lowered_text in greetings:
            await update.message.reply_text("👋 Xin chào! Tôi là TRỢ LÝ AI BV Lâm Hoa – sẽ cùng bạn luyện phản xạ tình huống điều dưỡng. Hãy bắt đầu với câu hỏi đầu tiên nhé!")
        else:
            await update.message.reply_text("🔄 Tiếp tục luyện tập nhé!")

        await update.message.reply_text(f"📌 Đây là tình huống {'KHẨN CẤP' if mode == 'emergency' else 'GIAO TIẾP'} – hãy đưa ra xử lý phù hợp.

{text}")

        # Ghi lại để luân phiên
        next_mode = "communication" if mode == "emergency" else "emergency"
        user_states[user_id] = {"mode": next_mode, "status": "idle"}
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
