from flask import Flask, jsonify, render_template
from bytez import Bytez

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Konfigurasi Bytez ---
KEY = "4a691e713db62c8e26dd394f5955f1fe"
INPUT_PROMPT = "Once upon a time, there was a robot"
MODEL_NAME = "abhinema/gpt"
# -------------------------

@app.route('/')
def run_bytez_model():
    """
    Mengakses API Bytez, menjalankan model AI, dan merender hasilnya ke HTML.
    """
    output_text = "Sedang menjalankan model AI..."
    error_message = None
    
    try:
        # 1. Inisialisasi SDK Bytez
        sdk = Bytez(KEY)
        
        # 2. Memilih model
        model = sdk.model(MODEL_NAME)
        
        # 3. Mengirim input ke model
        output, error = model.run(INPUT_PROMPT)
        
        if output:
            output_text = output
        elif error:
            error_message = f"Error dari Bytez: {error}"
            output_text = "Gagal mendapatkan hasil dari model."

    except Exception as e:
        error_message = f"Terjadi kesalahan saat memproses: {e}"
        output_text = "Gagal memproses permintaan."

    # Mengirim data ke template index.html
    return render_template(
        'index.html', 
        model_name=MODEL_NAME,
        input_prompt=INPUT_PROMPT,
        ai_output=output_text,
        error=error_message
    )

if __name__ == '__main__':
    # Pastikan 'debug=True' dihapus saat deploy ke production
    app.run(debug=True)