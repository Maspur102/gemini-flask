import os
import google.generativeai as genai
from flask import Flask, request, render_template_string, jsonify

# --- Import Baru untuk RAG (LangChain) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader # Ganti jika pakai .txt atau .docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

# --- Variabel Global untuk RAG ---
# Kita akan isi ini saat aplikasi startup
rag_chain = None

# --- Konfigurasi Model & API Key ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY') # Ambil dari Koyeb Secret
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)
    
    # Model untuk Chat (Menjawab)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    # Model untuk Embeddings (Mengubah teks jadi vektor)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

except Exception as e:
    print(f"Error initializing Google GenAI models: {e}")
    llm = None
    embeddings = None

# --- FUNGSI BARU: Untuk Setup RAG ---
def setup_rag_pipeline():
    """
    Fungsi ini dijalankan SEKALI saat aplikasi menyala.
    Ini memuat PDF, memecahnya, membuat embedding, dan menyiapkan RAG chain.
    """
    global rag_chain # Mengisi variabel global
    
    # Ganti 'dokumen_saya.pdf' dengan nama file Anda
    file_path = "dokumen_saya.pdf" 
    
    if not os.path.exists(file_path):
        print(f"Error: File dokumen '{file_path}' tidak ditemukan.")
        print("Pastikan file ada di direktori yang sama dengan app.py")
        return False

    try:
        print(f"Memulai RAG pipeline setup untuk: {file_path}")
        
        # 1. Muat Dokumen (Load)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        if not docs:
            print("Error: Dokumen PDF tidak bisa dimuat atau kosong.")
            return False

        print(f"Dokumen dimuat, {len(docs)} halaman.")

        # 2. Pecah Teks (Chunk)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Dokumen dipecah menjadi {len(splits)} potongan (chunks).")

        # 3. Buat & Simpan Embeddings (Store)
        # Ini akan membuat database vektor FAISS dari potongan teks
        print("Membuat vector store FAISS...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("Vector store berhasil dibuat di memori.")

        # 4. Buat Retriever (Alat Pencari)
        retriever = vectorstore.as_retriever()

        # 5. Buat Prompt Template
        # Ini memberi tahu AI cara menggunakan konteks
        prompt_template = ChatPromptTemplate.from_template("""
        Anda adalah asisten AI yang membantu menjawab pertanyaan HANYA berdasarkan konteks yang diberikan.
        Jawab pertanyaan pengguna dengan ringkas dan jelas.
        Jika informasi tidak ada di konteks, katakan "Maaf, saya tidak menemukan informasi tersebut di dalam dokumen."

        Konteks:
        {context}

        Pertanyaan:
        {input}

        Jawaban:
        """)

        # 6. Buat RAG Chain
        # Ini adalah inti logika RAG
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        print("RAG pipeline setup selesai dan siap.")
        return True

    except Exception as e:
        print(f"Error besar saat setup RAG pipeline: {e}")
        return False


# --- HTML Template (Tidak Berubah) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
...
(Isi HTML Anda yang panjang tidak saya ubah)
...
</html>
"""

# --- Route Flask ---

@app.route('/')
def home():
    # Menyajikan template HTML + JS
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate_api():
    
    # 1. Cek apakah RAG chain sudah siap
    if not rag_chain:
        return jsonify({'error_text': "RAG pipeline belum siap. Cek log server di Koyeb."}), 500

    # 2. Ambil data JSON (Tidak berubah)
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error_text': 'Input "prompt" tidak ditemukan.'}), 400

    try:
        prompt = data['prompt']

        # 3. --- PERUBAHAN BESAR DI SINI ---
        # Kita tidak lagi memakai model.generate_content()
        # Kita memakai RAG chain yang sudah kita buat
        
        # .invoke() akan menjalankan seluruh alur RAG
        # (Embed question -> Retrieve docs -> Generate answer)
        response = rag_chain.invoke({"input": prompt})
        
        # 'answer' adalah kunci output dari create_retrieval_chain
        answer = response.get('answer', 'Tidak ada jawaban dihasilkan.')

        # 4. Kirim balik data JSON (Mirip, tapi ambil dari 'answer')
        return jsonify({'response_text': answer})

    except Exception as e:
        # 5. Kirim balik error sebagai JSON (Tidak berubah)
        print(f"Error saat /generate: {e}")
        return jsonify({'error_text': str(e)}), 500

# Perintah Gunicorn
if __name__ == '__main__':
    # Setup RAG pipeline saat aplikasi pertama kali dijalankan
    if not llm or not embeddings:
        print("Gagal memulai: Model AI atau Embeddings tidak terinisialisasi.")
    else:
        setup_rag_pipeline()
    
    # Jalankan server
    app.run(debug=True, host='0.0.0.0', port=8080)