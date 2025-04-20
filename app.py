import fitz  # PyMuPDF
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ====== THIẾT LẬP TOKEN ======
TELEGRAM_TOKEN = "7849525479:AAHV_73BjqOh3yMe3rOxy5w1YSlFPg3Z7jE"
OPENAI_API_KEY = "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"

# === TRÍCH XUẤT VĂN BẢN TỪ PDF ===
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

pdf_text = extract_text_from_pdf("huong_dan_tang_huyet_ap.pdf")

# === TẠO VECTORSTORE ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(pdf_text)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embedding)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
    retriever=vectorstore.as_retriever()
)

# === XỬ LÝ CÂU HỎI NGƯỜI DÙNG ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    result = qa.run(user_input)
    await update.message.reply_text(result)

# === KHỞI CHẠY BOT ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
