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
import asyncio

# --- Flask App ---
flask_app = Flask(__name__)

# --- OpenAI API ---
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-turbo"

# --- T&#225;ch t&#7915;ng t&#236;nh hu&#7889;ng theo s&#7889; th&#7913; t&#7921; ---
def extract_cases_by_structure(path):
    doc = fitz.open(path)
    full_text = "\n".join([page.get_text() for page in doc])
    pattern = r'\n\d{1,2}\.\s'
    parts = re.split(pattern, full_text)
    headers = re.findall(r'\n\d{1,2}\.\s', full_text)
    cases = []
    for i, part in enumerate(parts[1:], start=1):
        header = headers[i-1].strip()
        case_text = f"{header} {part}".strip()
        cases.append(case_text)
    return cases

# --- Load d&#7919; li&#7879;u ---
chunks = extract_cases_by_structure("tinh_huong_khan_cap.pdf")
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# --- Ch&#7885;n 1 t&#236;nh hu&#7889;ng b&#7845;t k&#7923; ---
def pick_random_scenario():
    return random.choice(chunks)

# --- T&#236;m c&#225;c &#273;o&#7841;n li&#234;n quan &#273;&#7875; &#273;&#225;nh gi&#225; ph&#7843;n h&#7891;i ---
def search_relevant_chunks(text, top_n=3):
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, chunk_vectors).flatten()
    top_ids = sims.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_ids]

# --- Ph&#226;n t&#237;ch ph&#7843;n h&#7891;i c&#7911;a h&#7885;c vi&#234;n ---
def analyze_response(user_answer, scenario_text):
    context_chunks = search_relevant_chunks(scenario_text)
    prompt = f"""
B&#7841;n l&#224; tr&#7907; l&#253; &#273;&#224;o t&#7841;o &#273;i&#7873;u d&#432;&#7905;ng. H&#227;y &#273;&#225;nh gi&#225; ph&#7843;n h&#7891;i c&#7911;a h&#7885;c vi&#234;n d&#7921;a tr&#234;n t&#236;nh hu&#7889;ng kh&#7849;n c&#7845;p v&#224; t&#224;i li&#7879;u h&#432;&#7899;ng d&#7851;n. H&#227;y ph&#7843;n h&#7891;i theo c&#7845;u tr&#250;c sau:

1. **C&#226;u tr&#7843; l&#7901;i c&#243; ph&#249; h&#7907;p kh&#244;ng?**
2. **N&#7871;u ch&#432;a &#273;&#250;ng th&#236; sai &#7903; &#273;&#226;u?**
3. **G&#7907;i &#253; v&#224; l&#432;u &#253; th&#234;m cho h&#7885;c vi&#234;n**
4. **&#272;&#225;nh gi&#225; m&#7913;c &#273;&#7897;: X sao (t&#7915; 1 &#273;&#7871;n 5 sao, d&#249;ng k&#253; hi&#7879;u &#11088; t&#432;&#417;ng &#7913;ng)**

---
&#55357;&#56524; T&#236;nh hu&#7889;ng:
{scenario_text}

&#9997;&#65039; Ph&#7843;n h&#7891;i c&#7911;a h&#7885;c vi&#234;n:
{user_answer}

&#55357;&#56538; T&#224;i li&#7879;u n&#7897;i b&#7897;:
1. {context_chunks[0]}
2. {context_chunks[1]}
3. {context_chunks[2]}
"""
    res = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return res.choices[0].message.content.strip()

# --- D&#7921; &#273;o&#225;n m&#7913;c sao t&#7915; ph&#7843;n h&#7891;i GPT ---
def guess_star_rating(feedback_text):
    star_match = re.search(r"(\d)\s*sao", feedback_text.lower())
    if star_match:
        num = int(star_match.group(1))
        return "&#11088;" * min(max(num, 1), 5)
    return "&#11088;"  # fallback n&#7871;u kh&#244;ng t&#236;m th&#7845;y

# --- T&#226;m tr&#7841;ng ph&#7843;n h&#7891;i t&#432;&#417;ng &#7913;ng v&#7899;i s&#7889; sao ---
def get_emotional_feedback(stars):
    mapping = {
        "&#11088;&#11088;&#11088;&#11088;&#11088;": "&#55356;&#57119; Tuy&#7879;t v&#7901;i! B&#7841;n &#273;&#227; x&#7917; l&#253; r&#7845;t t&#7889;t, ti&#7871;p t&#7909;c ph&#225;t huy nh&#233;!",
        "&#11088;&#11088;&#11088;&#11088;": "&#55357;&#56397; Kh&#225; t&#7889;t! Nh&#432;ng v&#7851;n c&#243; th&#7875; chi ti&#7871;t h&#417;n.",
        "&#11088;&#11088;&#11088;": "&#55357;&#56848; B&#7841;n &#273;&#227; &#273;i &#273;&#250;ng h&#432;&#7899;ng, c&#7889; g&#7855;ng ho&#224;n thi&#7879;n h&#417;n.",
        "&#11088;&#11088;": "&#9888;&#65039; B&#7841;n c&#242;n b&#7887; s&#243;t nhi&#7873;u b&#432;&#7899;c quan tr&#7885;ng.",
        "&#11088;": "&#10060; C&#7847;n luy&#7879;n t&#7853;p k&#7929; h&#417;n, &#273;&#7915;ng lo &#8211; c&#7913; ti&#7871;p t&#7909;c nh&#233;!"
    }
    return mapping.get(stars, "&#55357;&#56898; Ti&#7871;p t&#7909;c nh&#233;!")

# --- C&#7855;t ph&#7847;n hi&#7875;n th&#7883; &#273;&#7871;n m&#244; t&#7843; tri&#7879;u ch&#7913;ng th&#244;i ---
def extract_visible_part(scenario_text):
    cutoff = "X&#7917; l&#253; t&#7841;i ch&#7895;"
    parts = scenario_text.split(cutoff)
    return parts[0].strip() + "\n\n&#55357;&#56589; B&#7841;n s&#7869; x&#7917; l&#253; th&#7871; n&#224;o trong 3 ph&#250;t &#273;&#7847;u ti&#234;n?"

# --- Giao ti&#7871;p Telegram ---
user_states = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message_text = update.message.text.strip()
    lowered_text = message_text.lower()
    greetings = ["hi", "hello", "xin ch&#224;o", "ch&#224;o", "alo", "yo"]

    if user_id not in user_states or user_states[user_id]["status"] == "idle":
        scenario = pick_random_scenario()
        user_states[user_id] = {"status": "awaiting_response", "scenario": scenario}
        scenario_number = chunks.index(scenario) + 1
        visible = extract_visible_part(scenario)

        if lowered_text in greetings:
            await update.message.reply_text(
                "&#55357;&#56395; Xin ch&#224;o! T&#244;i l&#224; **Tr&#7907; l&#253; H&#7897;i Nh&#7853;p &#272;i&#7873;u D&#432;&#7905;ng**, nhi&#7879;m v&#7909; c&#7911;a t&#244;i l&#224; h&#7895; tr&#7907; b&#7841;n luy&#7879;n ph&#7843;n x&#7841; trong c&#225;c t&#236;nh hu&#7889;ng kh&#7849;n c&#7845;p th&#7921;c t&#7871;.\n\n"
                "B&#226;y gi&#7901;, h&#227;y b&#7855;t &#273;&#7847;u v&#7899;i m&#7897;t t&#236;nh hu&#7889;ng &#273;&#7847;u ti&#234;n nh&#233;:"
            )
        else:
            await update.message.reply_text("&#55357;&#56596; B&#7855;t &#273;&#7847;u ki&#7875;m tra t&#236;nh hu&#7889;ng kh&#7849;n c&#7845;p &#273;&#7847;u ti&#234;n:")

        await update.message.reply_text(f"&#55358;&#56810; T&#236;nh hu&#7889;ng {scenario_number:02d}:\n\n{visible}")
        return

    if user_states[user_id]["status"] == "awaiting_response":
        scenario = user_states[user_id]["scenario"]
        feedback = analyze_response(message_text, scenario)
        stars = guess_star_rating(feedback)
        emotion = get_emotional_feedback(stars)

        await update.message.reply_text(f"&#55357;&#56523; &#272;&#225;nh gi&#225; t&#7915; tr&#7907; l&#253;: {stars}\n\n{feedback}")
        await update.message.reply_text(emotion)
        await update.message.reply_text("&#55357;&#56580; N&#224;o, th&#234;m m&#7897;t t&#236;nh hu&#7889;ng ti&#7871;p theo nh&#233;:")

        next_scenario = pick_random_scenario()
        scenario_number = chunks.index(next_scenario) + 1
        visible = extract_visible_part(next_scenario)

        await update.message.reply_text(f"&#55358;&#56810; T&#236;nh hu&#7889;ng {scenario_number:02d}:\n\n{visible}")
        user_states[user_id] = {"status": "awaiting_response", "scenario": next_scenario}
        return

# --- Web UI (n&#7871;u d&#249;ng) ---
@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def run_flask():
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# --- Telegram bot kh&#7903;i &#273;&#7897;ng ---
def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        print("&#9888;&#65039; TELEGRAM_TOKEN ch&#432;a &#273;&#432;&#7907;c thi&#7871;t l&#7853;p!")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    try:
        app.run_polling()
    except Exception as e:
        print(f"🚨 Bot bị lỗi khi chạy polling: {e}")

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    main()
    
    
