
import gradio as gr
import torch
from diffusers import AudioLDM2Pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
##########################################
from flask import send_file
import io
import soundfile as sf 
from flask import make_response


app = Flask(__name__)  # Create an instance of the Flask class


# Cargar el modelo de difusión de audio
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id).to("cuda" if torch.cuda.is_available() else "cpu")


# Función para generar audio a partir del texto de entrada
def text2audio(prompt, duration_in_s, guidance_scale=3.5, random_seed=45, n_candidates=3):
    if prompt is None or duration_in_s is None:
        raise ValueError("Please provide both prompt and duration.")

    # Generar el audio
    waveforms = pipe(
        prompt,
        audio_length_in_s=duration_in_s,
        guidance_scale=guidance_scale,
        num_inference_steps=200,
        negative_prompt="Low quality.",
        num_waveforms_per_prompt=n_candidates,
        generator=torch.Generator().manual_seed(int(random_seed))
    )["audios"]

    # Convertir el numpy.ndarray a un archivo WAV
    sf.write('audio.wav', waveforms[0], 16000)

    return 'audio.wav'


@app.route('/generateaudio', methods=['POST'])
def generate_audio():
    try:
        title = request.form.get('title')
        prompt = request.form.get('prompt')
        duration = request.form.get('duration')
        duration_in_s = float(duration)

        # Generar el audio a partir del prompt y la duración
        audio_file_path = text2audio(prompt, duration_in_s)

        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()

        # Crear la respuesta con el audio generado
        response = make_response(audio_data)
        response.headers.set('Content-Type', 'audio/wav')
        response.headers.set(
            'Content-Disposition', 'attachment', filename='audio.wav')

        return response

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/recibir_datos', methods=['POST'])
def recibir_datos():
    try:
        # Obtener datos de la solicitud POST (asumimos que se envía un número)
        data = request.json
        # Realizar algún procesamiento con los datos recibidos (multiplicar por 2)
        processed_data = data['number'] * 2
        # Devolver el resultado procesado como respuesta
        return jsonify(processed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(port=7860, debug=True)