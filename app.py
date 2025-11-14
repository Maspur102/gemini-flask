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
    Menangani potensi 'too many values to unpack'.
    """
    output_text = "Gagal memproses permintaan."
    error_message = None
    
    try:
        sdk = Bytez(KEY)
        model = sdk.model(MODEL_NAME)
        
        # --- PERBAIKAN UTAMA DI SINI ---
        # Menggunakan *response untuk menangkap semua nilai yang dikembalikan
        response = model.run(INPUT_PROMPT)
        
        if len(response) == 2:
            # Format lama/standar: (output, error)
            output, error = response
            
            if output:
                output_text = output
            elif error:
                error_message = f"Error dari Bytez: {error}"
                output_text = "Gagal mendapatkan hasil dari model."
        elif len(response) > 2:
            # Menangani jika Bytez mengembalikan lebih dari 2 nilai
            error_message = f"Bytez mengembalikan {len(response)} nilai, bukan 2. Mungkin API telah berubah."
        else:
            # Menangani jika Bytez mengembalikan 0 atau 1 nilai
            error_message = f"Bytez mengembalikan {len(response)} nilai, bukan 2."

    except Exception as e:
        # Menangani exception umum, termasuk network error atau Bytez error lainnya
        error_message = f"Terjadi kesalahan saat memproses: {e}"

    # Mengirim data ke template index.html
    return render_template(
        'index.html', 
        model_name=MODEL_NAME,
        input_prompt=INPUT_PROMPT,
        ai_output=output_text,
        error=error_message
    )

if __name__ == '__main__':
    app.run(debug=True)