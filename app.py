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
OPENAI_MODEL = "gpt-3.5-turbo"

# --- Load và xử lý PDF ---
def extract_scenarios_from_pdf(path, is_emergency=True):
    doc = fitz.open(path)
    full_text = "\n".join([page.get_text() for page in doc])
    pattern = r"Tình huống khẩn cấp\s+\d{1,2}:" if is_emergency else r"Tình huống giao tiếp\s+\d+"
    parts = re.split(pattern, full_text)
    return [p.strip() for p in parts if p.strip()]

emergency_scenarios = extract_scenarios_from_pdf("tinh_huong_khan_cap.pdf", is_emergency=True)
communication_scenarios = extract_scenarios_from_pdf("tinh_huong_giao_tiep.pdf", is_emergency=False)

# --- Hiển thị nội dung tình huống ---
def extract_visible_emergency(scenario):
    lines = scenario.split("\n")
    title = lines[0] if lines else "Tình huống khẩn cấp"
    desc = ""
    for line in lines[1:]:
        if line.lower().startswith("cần thực hiện"):
            break
        desc += line.strip() + " "
    return f"{title}\n{desc.strip()}\n\n 💗 Bạn sẽ xử lý thế nào?"

def extract_visible_communication(scenario):
    parts = scenario.split("Đáp án:")
    question = parts[0].strip()
    return f"{question}\n\n 💗 Bạn sẽ xử lý thế nào?"

# --- Đánh giá bằng GPT ---
def analyze_response(user_answer, scenario_text, mode):
    prompt = f"""
Bạn là trợ lý đào tạo điều dưỡng. Hãy đánh giá phản hồi của học viên dựa trên tình huống và đưa ra nhận xét ngắn gọn theo 2 mục:
1. Mức độ phù hợp của câu trả lời: X (thay X bằng ký hiệu ⭐, từ 1 đến 5 ⭐ theo mức độ phù hợp của câu trả lời từ học viện, sau đó giải thích ngắn gọn mang tính khuyến khích)
2. Đáp án từ tài liệu: (sau đó trích nguyên văn đáp án/cách xử lý trong tài liệu, không giải thích thêm bớt từ gì)

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

# --- Tách sao từ GPT ---
def extract_star_rating(feedback_text):
    star_match = re.search(r"(\d)\s*sao", feedback_text.lower())
    if star_match:
        num = int(star_match.group(1))
        return min(max(num, 1), 5)
    return 1

# --- Tóm tắt sau 4 lượt ---
def summarize_feedback(star_list):
    avg = sum(star_list) / len(star_list)
    stars = "⭐" * round(avg)
    if avg >= 4.5:
        msg = "Bạn thể hiện xuất sắc! Tiếp tục giữ phong độ nhé."
    elif avg >= 3.5:
        msg = "Bạn có nền tảng tốt, hãy luyện tập thêm để nâng cao hơn nữa."
    else:
        msg = "Bạn cần luyện thêm để nắm vững kỹ năng phản xạ."
    return f"🎯 Bạn vừa hoàn thành 4 tình huống.\nĐiểm trung bình: {stars}\n\n💬 Nhận xét: {msg}"

# --- Quản lý trạng thái người dùng ---
user_states = {}

# --- Tương tác Telegram ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()
    lowered_text = text.lower()

    greetings = ["hi", "hello", "xin chào", "chào", "alo", "yo", "chao", "/start"]
    affirm = ["ok", "oki", "có", "yes"]
    deny = ["không", "ko", "no"]

    if user_id not in user_states:
        user_states[user_id] = {
            "mode": "emergency",
            "status": "idle",
            "scenario": None,
            "history": [],
            "awaiting_continue": False
        }

    state = user_states[user_id]

    if state["awaiting_continue"]:
        if lowered_text in affirm:
            state.update({"awaiting_continue": False, "history": []})
            state["status"] = "idle"
            await update.message.reply_text("👍 Tuyệt vời! Chúng ta tiếp tục luyện nhé!")
            return
        elif lowered_text in deny:
            await update.message.reply_text("⏳ Mình sẽ chờ 30 giây rồi hỏi lại nhé...")
            await asyncio.sleep(30)
            await update.message.reply_text("🔁 Bạn có muốn tiếp tục luyện tập không? (ok / không)")
            return
        else:
            await update.message.reply_text("❓ Mình chưa rõ. Bạn có muốn tiếp tục luyện tập không? (ok / không)")
            return

    if state["status"] == "idle":
        mode = state["mode"]
        scenario = random.choice(emergency_scenarios) if mode == "emergency" else random.choice(communication_scenarios)
        visible_text = extract_visible_emergency(scenario) if mode == "emergency" else extract_visible_communication(scenario)

        if lowered_text in greetings:
            await update.message.reply_text("👋 Xin chào! Tôi là TRỢ LÝ AI \n[Bệnh Viện Đa khoa Lâm Hoa].\n\nChúng ta sẽ cùng luyện phản xạ tình huống điều dưỡng.\nBắt đầu với tình huống đầu tiên nhé!")
            await asyncio.sleep(1)

        await update.message.reply_text(f"{'🔥 Tình huống KHẨN CẤP' if mode == 'emergency' else '💬 Tình huống GIAO TIẾP'}\n\n{visible_text}")
        state.update({"scenario": scenario, "status": "awaiting_response"})
        return

    if state["status"] == "awaiting_response":
        scenario = state["scenario"]
        mode = state["mode"]

        # Gửi phản hồi GPT đánh giá câu trả lời
        feedback = analyze_response(text, scenario, mode)
        stars = extract_star_rating(feedback)
        state["history"].append(stars)  # lưu lại điểm số

        await update.message.reply_text(f"📋 NHẬN XÉT TỪ TRỢ LÝ AI:n\n\n{feedback}")

        # Nếu đã trả lời đủ 4 tình huống thì tổng kết và hỏi tiếp tục
        if len(state["history"]) >= 4:
            await update.message.reply_text("🙏 Cảm ơn bạn, chúng ta đã luyện tập 4 tình huống, cùng nhìn lại nhé!")
            summary = summarize_feedback(state["history"])
            await update.message.reply_text(summary)
            await update.message.reply_text("🔁 Bạn có muốn tiếp tục luyện tập không? (ok / không)")
            state["awaiting_continue"] = True
            return  # 🔁 Dừng tại đây, không gửi tiếp tình huống mới

        # Nếu chưa đủ 4 tình huống thì gửi tiếp
        next_mode = "communication" if mode == "emergency" else "emergency"
        next_scenario = random.choice(emergency_scenarios) if next_mode == "emergency" else random.choice(communication_scenarios)
        next_text = extract_visible_emergency(next_scenario) if next_mode == "emergency" else extract_visible_communication(next_scenario)

        await update.message.reply_text(f"{'🔥 Tình huống KHẨN CẤP' if next_mode == 'emergency' else '💬 Tình huống GIAO TIẾP'}\n\n{next_text}")

        # Cập nhật trạng thái cho lần tiếp theo
        state.update({"mode": next_mode, "scenario": next_scenario, "status": "awaiting_response"})
        return

@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

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
