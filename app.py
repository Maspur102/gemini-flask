import os
import google.generativeai as genai
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- Bagian Logika ini TETAP SAMA ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)

    # Inisialisasi model
    model = genai.GenerativeModel('gemini-2.5-flash')

except Exception as e:
    print(f"Error initializing Google GenAI: {e}")
    model = None

# --- INI ADALAH UI BARU DENGAN TAILWIND ---
# Kita ganti seluruh isi variabel HTML_FORM
HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini Flask App</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">

    <div class="bg-white p-6 md:p-8 rounded-xl shadow-xl w-full max-w-lg">

        <h2 class="text-3xl font-bold text-center text-gray-800 mb-4">
            Tanya Gemini
        </h2>
        <p class="text-center text-gray-500 mb-6 text-sm">Model: gemini-2.5-flash</p>

        <form action="/generate" method="POST">

            <label for="prompt" class="block text-sm font-medium text-gray-700 mb-2">Prompt Anda:</label>
            <textarea name="prompt" rows="5" 
                      class="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500" 
                      placeholder="Contoh: Jelaskan cara kerja AI dalam 3 kalimat..."></textarea>

            <button type="submit" 
                    class="w-full bg-blue-600 text-white p-3 mt-4 rounded-md font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition duration-200">
                Kirim
            </button>
        </form>

        {% if error_text %}
        <div class="mt-6">
            <h3 class="text-lg font-semibold text-red-700 mb-2">Error:</h3>
            <div class="bg-red-50 p-4 rounded-md border border-red-200 text-red-800 whitespace-pre-wrap">
                {{ error_text }}
            </div>
        </div>
        {% endif %}

        {% if response_text %}
        <div class="mt-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-2">Jawaban:</h3>
            <div class="bg-gray-50 p-4 rounded-md border border-gray-200 text-gray-700 whitespace-pre-wrap overflow-x-auto">
                {{ response_text }}
            </div>
        </div>
        {% endif %}

    </div>
</body>
</html>
"""

# --- Bagian Route Flask ini TETAP SAMA ---
@app.route('/')
def home():
    # Tampilkan form
    return render_template_string(HTML_FORM)

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        # Kita buat pesan error lebih jelas
        return render_template_string(HTML_FORM, error_text="Model AI tidak terinisialisasi. Pastikan Environment Variable 'GOOGLE_API_KEY' sudah di-set dengan benar di Koyeb.")

    try:
        prompt = request.form['prompt']

        # Kirim prompt ke Gemini
        response = model.generate_content(prompt)

        # Tampilkan kembali form dengan hasil respons
        return render_template_string(HTML_FORM, response_text=response.text)

    except Exception as e:
        # Tampilkan error jika gagal
        return render_template_string(HTML_FORM, error_text=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)