import os
import google.generativeai as genai
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Konfigurasi API Key dari Environment Variable
# Ini aman dan cara standar di hosting
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

# Halaman depan untuk menampilkan form input
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Gemini Flask App</title>
    <style>
        body { font-family: sans-serif; margin: 2em; background: #f4f4f4; }
        h2 { color: #333; }
        form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        textarea { width: 95%; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; }
        input[type="submit"] { background: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
        .response { background: #e9ecef; padding: 15px; border-radius: 4px; margin-top: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h2>Tanya Gemini (Model: gemini-2.5-flash)</h2>
    <form action="/generate" method="POST">
        <textarea name="prompt" rows="4" placeholder="Tulis prompt Anda di sini..."></textarea>
        <br>
        <input type="submit" value="Kirim">
    </form>

    {% if response_text %}
    <h3>Jawaban:</h3>
    <div class="response">
        {{ response_text }}
    </div>
    {% endif %}

    {% if error_text %}
    <h3>Error:</h3>
    <div class="response" style="background: #f8d7da; color: #721c24;">
        {{ error_text }}
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    # Tampilkan form
    return render_template_string(HTML_FORM)

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return render_template_string(HTML_FORM, error_text="Model AI tidak terinisialisasi. Cek log server.")

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
    # Koyeb tidak akan menjalankan ini, tapi ini berguna untuk tes lokal
    app.run(debug=True, host='0.0.0.0', port=8080)