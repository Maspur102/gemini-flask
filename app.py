import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template

# --- Import RAG (Perhatikan, importnya berkurang) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# --- Variabel Global Baru ---
# Kita tidak lagi pakai 'rag_chain'. Kita hanya butuh 'retriever' (pencari)
# dan 'genai_model' (model AI-nya)
retriever = None
genai_model = None

# --- Konfigurasi Model & API Key ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY') 
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Cek di Koyeb.")
    
    genai.configure(api_key=api_key)
    
    # --- INI PENTING ---
    # Kita inisialisasi model 'gemini-pro' LANGSUNG dari pustaka Google
    # BUKAN dari LangChain. Ini 100% bypass error 'v1beta'.
    genai_model = genai.GenerativeModel('gemini-1.0-pro')
    
    # Model Embedding (TIDAK BERUBAH, INI SUDAH BENAR)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

except Exception as e:
    print(f"Error initializing Google GenAI models: {e}")
    genai_model = None
    embeddings = None

# --- Fungsi Setup RAG (DIROMBAK) ---
def setup_rag_pipeline():
    # Kita hanya mengisi 'retriever'
    global retriever
    
    file_path = "dokumen_saya.txt" 
    
    if not os.path.exists(file_path):
        print(f"Error: File dokumen '{file_path}' tidak ditemukan.")
        return False

    try:
        print(f"Memulai RAG pipeline setup untuk: {file_path}...")
        
        loader = TextLoader(file_path, encoding='utf-8') 
        docs = loader.load()
        
        if not docs:
            print("Error: Dokumen teks tidak bisa dimuat atau kosong.")
            return False
        print(f"Dokumen dimuat, {len(docs)} bagian.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Dokumen dipecah menjadi {len(splits)} potongan (chunks).")

        print("Membuat vector store FAISS...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("Vector store berhasil dibuat.")

        # --- INI ADALAH HASIL AKHIR KITA ---
        # Kita simpan retriever-nya secara global
        retriever = vectorstore.as_retriever()
        
        # (Kita hapus semua kode 'chain', 'prompt_template', dll. dari sini)
        
        print("Setup Retriever selesai dan siap digunakan.")
        return True

    except Exception as e:
        print(f"Error besar saat setup RAG pipeline: {e}")
        return False

# --- Route Flask ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_api():
    # --- LOGIKA DI SINI DIROMBAK TOTAL ---
    
    # 1. Cek apakah retriever sudah siap (bukan rag_chain)
    if not retriever:
        print("Error: /generate dipanggil sebelum Retriever siap.")
        return jsonify({'error_text': "Sistem RAG (Retriever) belum siap. Cek log server."}), 503

    # 2. Cek apakah model GenAI sudah siap
    if not genai_model:
        print("Error: /generate dipanggil sebelum Model GenAI siap.")
        return jsonify({'error_text': "Sistem AI (GenAI Model) belum siap. Cek log server."}), 503

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error_text': 'Input "prompt" tidak ditemukan.'}), 400

    try:
        prompt_user = data['prompt']
        print(f"Menerima prompt: {prompt_user}")

        # --- INI INTI DARI RAG MANUAL ---
        
        # A. Ambil dokumen relevan dari retriever
        print("Mencari konteks di retriever...")
        context_docs = retriever.invoke(prompt_user)
        # Gabungkan semua dokumen relevan menjadi 1 teks
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        print("Konteks ditemukan.")

        # B. Buat prompt final secara manual
        final_prompt = f"""
        Anda adalah asisten AI yang membantu menjawab pertanyaan HANYA berdasarkan konteks yang diberikan.
        Jawab pertanyaan pengguna dengan ringkas dan jelas menggunakan bahasa Indonesia.
        Jika informasi tidak ada di konteks, katakan "Maaf, saya tidak menemukan informasi tersebut di dalam dokumen."

        Konteks:
        {context_text}

        Pertanyaan:
        {prompt_user}

        Jawaban:
        """
        
        # C. Kirim ke Google GenAI (Bypass LangChain)
        print("Mengirim prompt final ke Google GenAI (bukan LangChain)...")
        response = genai_model.generate_content(final_prompt)
        
        answer = response.text
        print(f"Jawaban RAG diterima: {answer}")

        return jsonify({'response_text': answer})

    except Exception as e:
        print(f"Error saat /generate: {e}")
        return jsonify({'error_text': str(e)}), 500

# --- Bagian Startup (Sedikit diubah) ---
if not genai_model or not embeddings:
    print("FATAL ERROR: Model GenAI atau Embeddings tidak terinisialisasi. Cek API Key.")
else:
    print("Memulai setup RAG pipeline (Retriever) untuk Gunicorn...")
    setup_rag_pipeline()

if __name__ == '__main__':
    print("Menjalankan Flask server untuk tes lokal (setup RAG sudah selesai)...")
    app.run(debug=True, host='0.0.0.0', port=8080)