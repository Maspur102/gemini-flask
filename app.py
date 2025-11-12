import os
import google.generativeai as genai
# --- INI YANG BENAR: Menggunakan render_template ---
from flask import Flask, request, jsonify, render_template

# --- Import RAG (LangChain) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

# --- Variabel Global RAG ---
rag_chain = None

# --- Konfigurasi Model & API Key (TIDAK BERUBAH) ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY') 
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Cek di Koyeb.")
    
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

except Exception as e:
    print(f"Error initializing Google GenAI models: {e}")
    llm = None
    embeddings = None

# --- Fungsi Setup RAG (TIDAK BERUBAH) ---
def setup_rag_pipeline():
    global rag_chain 
    
    # Ganti 'dokumen_saya.pdf' dengan nama file Anda
    file_path = "dokumen_saya.pdf" 
    
    if not os.path.exists(file_path):
        print(f"Error: File dokumen '{file_path}' tidak ditemukan.")
        return False

    try:
        print(f"Memulai RAG pipeline setup untuk: {file_path}")
        loader = PyPDFLoader(file_path) 
        docs = loader.load()
        if not docs:
            print("Error: Dokumen PDF tidak bisa dimuat atau kosong.")
            return False
        print(f"Dokumen dimuat, {len(docs)} halaman.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Dokumen dipecah menjadi {len(splits)} potongan (chunks).")

        print("Membuat vector store FAISS...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("Vector store berhasil dibuat.")

        retriever = vectorstore.as_retriever()

        prompt_template = ChatPromptTemplate.from_template("""
        Anda adalah asisten AI yang membantu menjawab pertanyaan HANYA berdasarkan konteks yang diberikan.
        Jawab pertanyaan pengguna dengan ringkas dan jelas menggunakan bahasa Indonesia.
        Jika informasi tidak ada di konteks, katakan "Maaf, saya tidak menemukan informasi tersebut di dalam dokumen."

        Konteks:
        {context}

        Pertanyaan:
        {input}

        Jawaban:
        """)

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        print("RAG pipeline setup selesai dan siap digunakan.")
        return True

    except Exception as e:
        print(f"Error besar saat setup RAG pipeline: {e}")
        return False

# --- Route Flask ---

@app.route('/')
def home():
    # --- INI BAGIAN YANG PENTING ---
    # Memanggil file 'index.html' dari folder 'templates'
    return render_template('index.html')

# --- Route API (TIDAK BERUBAH) ---
@app.route('/generate', methods=['POST'])
def generate_api():
    if not rag_chain:
        print("Error: /generate dipanggil sebelum RAG pipeline siap.")
        return jsonify({'error_text': "Sistem RAG (AI) belum siap. Cek log server di Koyeb."}), 503

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error_text': 'Input "prompt" tidak ditemukan.'}), 400

    try:
        prompt = data['prompt']
        print(f"Menerima prompt: {prompt}")

        print("Memanggil RAG chain...")
        response = rag_chain.invoke({"input": prompt})
        answer = response.get('answer', 'Tidak ada jawaban dihasilkan.')
        print("Jawaban RAG diterima.")

        return jsonify({'response_text': answer})

    except Exception as e:
        print(f"Error saat /generate: {e}")
        return jsonify({'error_text': str(e)}), 500

# --- Bagian Startup (TIDAK BERUBAH) ---
if __name__ == '__main__':
    if not llm or not embeddings:
        print("Gagal memulai: Model AI atau Embeddings tidak terinisialisasi. Cek API Key.")
    else:
        setup_rag_pipeline()
    
    print("Menjalankan Flask server...")
    app.run(debug=True, host='0.0.0.0', port=8080)