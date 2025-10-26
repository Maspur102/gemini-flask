import os
import google.generativeai as genai
# Tambahkan jsonify untuk mengirim respons JSON
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Konfigurasi Model (Tidak berubah)
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

except Exception as e:
    print(f"Error initializing Google GenAI: {e}")
    model = None

# --- UI BARU DENGAN TAILWIND + JAVASCRIPT ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini Flask App (AJAX)</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">

    <div class="bg-white p-6 md:p-8 rounded-xl shadow-xl w-full max-w-lg">

        <h2 class="text-3xl font-bold text-center text-gray-800 mb-4">
            Tanya Gemini (AJAX)
        </h2>
        <p class="text-center text-gray-500 mb-6 text-sm">Model: gemini-2.5-flash</p>

        <form id="prompt-form">

            <label for="prompt-input" class="block text-sm font-medium text-gray-700 mb-2">Prompt Anda:</label>
            <textarea id="prompt-input" name="prompt" rows="5" 
                      class="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500" 
                      placeholder="Tulis prompt..."></textarea>

            <button type="submit" id="submit-button"
                    class="w-full bg-blue-600 text-white p-3 mt-4 rounded-md font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition duration-200
                           disabled:bg-gray-400 disabled:cursor-not-allowed">
                Kirim
            </button>
        </form>

        <div id="loader" class="hidden flex justify-center items-center mt-6">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <span class="ml-3 text-gray-600">Sedang berpikir...</span>
        </div>

        <div id="result-container" class="mt-6">
            </div>

    </div>

    <script>
        // Menangkap elemen-elemen penting
        const form = document.getElementById('prompt-form');
        const input = document.getElementById('prompt-input');
        const submitButton = document.getElementById('submit-button');
        const loader = document.getElementById('loader');
        const resultContainer = document.getElementById('result-container');

        // Tambahkan event listener ke form
        form.addEventListener('submit', async function(event) {
            // Hentikan aksi default form (RELOAD PAGE)
            event.preventDefault(); 

            const promptText = input.value.trim();
            if (!promptText) {
                // Jangan lakukan apa-apa jika input kosong
                return;
            }

            // --- Memulai Proses Loading ---
            // 1. Nonaktifkan tombol
            submitButton.disabled = true;
            // 2. Tampilkan loader
            loader.classList.remove('hidden');
            // 3. Bersihkan hasil sebelumnya
            resultContainer.innerHTML = ''; 

            try {
                // --- Mengirim Data ke Backend ---
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt: promptText })
                });

                // Cek jika server mengembalikan error (spt 500)
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error_text || 'Terjadi kesalahan di server.');
                }

                // --- Sukses Menerima Data ---
                const data = await response.json();

                // Buat elemen HTML untuk jawaban
                const responseDiv = document.createElement('div');
                responseDiv.className = 'bg-gray-50 p-4 rounded-md border border-gray-200 text-gray-700 whitespace-pre-wrap overflow-x-auto';
                responseDiv.textContent = data.response_text;

                // Tampilkan jawaban
                resultContainer.innerHTML = '<h3 class="text-lg font-semibold text-gray-800 mb-2">Jawaban:</h3>';
                resultContainer.appendChild(responseDiv);

            } catch (error) {
                // --- Gagal Menerima Data ---
                console.error('Error:', error);

                // Buat elemen HTML untuk error
                const errorDiv = document.createElement('div');
                errorDiv.className = 'bg-red-50 p-4 rounded-md border border-red-200 text-red-800 whitespace-pre-wrap';
                errorDiv.textContent = error.message;

                // Tampilkan error
                resultContainer.innerHTML = '<h3 class="text-lg font-semibold text-red-700 mb-2">Error:</h3>';
                resultContainer.appendChild(errorDiv);

            } finally {
                // --- Selesai Loading (Baik sukses atau gagal) ---
                // 1. Sembunyikan loader
                loader.classList.add('hidden');
                // 2. Aktifkan lagi tombolnya
                submitButton.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# --- Route Flask ---

@app.route('/')
def home():
    # Hanya menyajikan template HTML + JS
    return render_template_string(HTML_TEMPLATE)

# --- PERUBAHAN BESAR DI SINI ---
@app.route('/generate', methods=['POST'])
def generate_api():

    # 1. Kita ambil data JSON, bukan form
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error_text': 'Input "prompt" tidak ditemukan.'}), 400 # Bad Request

    if not model:
        return jsonify({'error_text': "Model AI tidak terinisialisasi. Cek API Key di Koyeb."}), 500 # Internal Server Error

    try:
        prompt = data['prompt']

        # Kirim prompt ke Gemini
        response = model.generate_content(prompt)

        # 2. Kita kirim balik data JSON, bukan render template
        return jsonify({'response_text': response.text})

    except Exception as e:
        # 3. Kirim balik error sebagai JSON
        return jsonify({'error_text': str(e)}), 500 # Internal Server Error

# Perintah Gunicorn (Tidak berubah)
if __name__ == '__main__':
    # Ini tidak dipakai Koyeb, tapi bagus untuk tes lokal
    app.run(debug=True, host='0.0.0.0', port=8080)