import os
import google.generativeai as genai
# Menggunakan render_template untuk memanggil file .html
from flask import Flask, request, jsonify, render_template

# --- Import RAG (LangChain) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader # Pastikan pypdf ada di requirements.txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# --- Variabel Global untuk RAG ---
# Ini akan diisi oleh setup_rag_pipeline() saat aplikasi menyala
rag_chain = None

# --- Konfigurasi Model & API Key ---
try:
    # Ambil API Key dari Koyeb Environment Variables
    api_key = os.environ.get('GOOGLE_API_KEY') 
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Cek di Koyeb.")
    
    genai.configure(api_key=api_key)
    
    # Model untuk Chat (Menjawab Pertanyaan)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    # Model untuk Embeddings (Mengubah teks jadi vektor/angka)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

except Exception as e:
    print(f"Error initializing Google GenAI models: {e}")
    llm = None
    embeddings = None

# --- Fungsi Setup RAG ---
def setup_rag_pipeline():
    """
    Dijalankan sekali saat aplikasi menyala.
    Memuat PDF, memecahnya, membuat embedding, dan menyiapkan RAG chain.
    """
    global rag_chain # Mengisi variabel global
    
    # !!! PENTING: Ganti 'dokumen_saya.pdf' dengan nama file Anda !!!
    file_path = "dokumen_saya.pdf" 
    
    if not os.path.exists(file_path):
        print(f"Error: File dokumen '{file_path}' tidak ditemukan.")
        print("Pastikan file ada di direktori yang sama dengan app.py")
        return False

    try:
        print(f"Memulai RAG pipeline setup untuk: {file_path}...")
        
        # 1. Muat Dokumen (Load)
        loader = PyPDFLoader(file_path) 
        docs = loader.load()
        
        if not docs:
            print("Error: Dokumen PDF tidak bisa dimuat atau kosong.")
            return False
        print(f"Dokumen dimuat, {len(docs)} halaman.")

        # 2. Pecah Teks (Chunk/Split)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Dokumen dipecah menjadi {len(splits)} potongan (chunks).")

        # 3. Buat & Simpan Embeddings (Store)
        print("Membuat vector store FAISS (database di memori)...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("Vector store berhasil dibuat.")

        # 4. Buat Retriever (Alat Pencari Dokumen)
        retriever = vectorstore.as_retriever()

        # 5. Buat Prompt Template
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

        # 6. Buat RAG Chain (Rantai Logika)
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
    """Menyajikan halaman utama (index.html)."""
    # Flask otomatis mencari 'index.html' di dalam folder 'templates'
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_api():
    """Endpoint API untuk menerima prompt dan mengembalikan jawaban RAG."""
    
    # 1. Cek apakah RAG chain sudah siap
    if not rag_chain:
        print("Error: /generate dipanggil sebelum RAG pipeline siap.")
        # 503 Service Unavailable
        return jsonify({'error_text': "Sistem RAG (AI) belum siap. Cek log server di Koyeb."}), 503

    # 2. Ambil data JSON
    data = request.get_json()
    if not data or 'prompt' not in data:
        # 400 Bad Request
        return jsonify({'error_text': 'Input "prompt" tidak ditemukan.'}), 400

    try:
        prompt = data['prompt']
        print(f"Menerima prompt: {prompt}")

        # 3. Panggil RAG chain
        print("Memanggil RAG chain...")
        response = rag_chain.invoke({"input": prompt})
        answer = response.get('answer', 'Tidak ada jawaban dihasilkan.')
        print("Jawaban RAG diterima.")

        # 4. Kirim balik jawaban
        return jsonify({'response_text': answer})

    except Exception as e:
        print(f"Error saat /generate: {e}")
        # 500 Internal Server Error
        return jsonify({'error_text': str(e)}), 500

# --- BAGIAN STARTUP APLIKASI ---

# Logika ini sekarang ada di Global Scope, 
# Gunicorn akan menjalankannya saat 'import' file app.py.
if not llm or not embeddings:
    print("FATAL ERROR: Model AI atau Embeddings tidak terinisialisasi. Cek API Key.")
else:
    # Panggil setup_rag_pipeline() DI LUAR __name__ == '__main__'
    print("Memulai setup RAG pipeline untuk Gunicorn...")
    setup_rag_pipeline()


# Blok ini HANYA untuk tes lokal (saat Anda menjalankan 'python app.py')
# Gunicorn akan mengabaikan blok ini.
if __name__ == '__main__':
    print("Menjalankan Flask server untuk tes lokal (setup RAG sudah selesai)...")
    # debug=True akan otomatis me-reload jika ada perubahan kode
    app.run(debug=True, host='0.0.0.0', port=8080)