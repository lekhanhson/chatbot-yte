import fitz  # PyMuPDF
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# ====== THIẾT LẬP TOKEN ======
TELEGRAM_TOKEN = "7849525479:AAHV_73BjqOh3yMe3rOxy5w1YSlFPg3Z7jE"
OPENAI_API_KEY = "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"

# === XỬ LÝ PDF ===
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

pdf_text = extract_text_from_pdf("huong_dan_chan_doan.pdf")

# === CHIA NHỎ & EMBED ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(pdf_text)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embedding)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever()
)

# === TRẢ LỜI CÂU HỎI ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    result = qa.run(user_input)
    await update.message.reply_text(result)

# === KHỞI CHẠY ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
