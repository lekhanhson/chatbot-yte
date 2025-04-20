import os
import fitz  # PyMuPDF
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ====== THIẾT LẬP TOKEN ======
TELEGRAM_TOKEN = "7849525479:AAHV_73BjqOh3yMe3rOxy5w1YSlFPg3Z7jE"
OPENAI_API_KEY = "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"

# ====== TRÍCH XUẤT DỮ LIỆU TỪ PDF ======
def load_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# ====== TẠO VECTORSTORE ======
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ====== XỬ LÝ CÂU HỎI NGƯỜI DÙNG ======
pdf_text = load_pdf_text("huong_dan_chan_doan.pdf")
vs = create_vectorstore(pdf_text)
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
    retriever=vs.as_retriever()
)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    response = qa.run(user_input)
    await update.message.reply_text(response)

# ====== CHẠY TELEGRAM BOT ======
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
