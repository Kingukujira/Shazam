import os
import numpy as np
from pydub import AudioSegment
from scipy.spatial import distance
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import sounddevice as sd
import matplotlib.pyplot as plt
import threading
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor
import logging
import pygame
import csv
import pandas as pd  # Importa pandas para manejar datos
import requests


# Ajustar el umbral de distancia según sea necesario
THRESHOLD = 17000  # Valor ajustado para mejorar la precisión
SEGMENT_DURATION = 20  # Duración del segmento de audio en segundos
SAMPLE_RATE = 44100  # Tasa de muestreo para la captura de audio del micrófono
FFT_LENGTH = 450  # Longitud de la transformada de Fourier
WINDOW_SIZE = 10  # Tamaño de la ventana para el filtro de promedio móvil

recording = False  # Variable de control para detener la grabación
recording_thread = None


def load_spotify_data(file_path):
    """
    Carga y muestra los primeros registros de un archivo CSV de Spotify.
    """
    try:
        df = pd.read_csv(file_path)
        print(df.head())  # Muestra las primeras filas del DataFrame
        return df
    except Exception as e:
        logging.error(f"Error al cargar los datos de Spotify: {e}")
        return None

import requests

# Tus credenciales de la aplicación de Spotify
client_id = 'bd0b06421abf4966b0f06cb1689854f9'
client_secret = 'dfb4d5f642ae4e87b5a734b3fc802d47'

# Obtener el token de acceso
token_url = 'https://accounts.spotify.com/api/token'
auth_response = requests.post(token_url, {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret,
})
auth_response_json = auth_response.json()
access_token = auth_response_json['access_token']

# Configurar los encabezados para incluir el token de acceso
headers = {
    'Authorization': f'Bearer {access_token}',
}

# Parámetros de búsqueda
params = (
    ('q', 'Shape of You'),  # Reemplazado por el nombre de la canción que buscas
    ('type', 'track'),  # Asegúrate de especificar el tipo de objeto que estás buscando
)

# Realizar la búsqueda en la API de Spotify
response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=params)


# Suponiendo que 'response' es la variable que contiene la respuesta JSON de la API de Spotify
tracks_info = response.json()['tracks']['items'][0]  # Accede al primer elemento de la lista de items

# Ahora puedes acceder a diferentes campos de la canción
cancion_nombre = tracks_info['name']  # Nombre de la canción
artista_nombre = tracks_info['artists'][0]['name']  # Nombre del artista principal
album_nombre = tracks_info['album']['name']  # Nombre del álbum
imagen_url = tracks_info['album']['images'][0]['url']  # URL de la imagen del álbum

# Imprime la información obtenida
print(f"Canción: {cancion_nombre}")
print(f"Artista: {artista_nombre}")
print(f"Álbum: {album_nombre}")
print(f"URL de la imagen del álbum: {imagen_url}")



def normalize_fingerprint(fp):
    mean_fp = np.mean(fp)
    std_fp = np.std(fp)
    if std_fp == 0:  # Evitar división por cero
        return np.zeros_like(fp)
    else:
        return (fp - mean_fp) / std_fp


def generate_fingerprints(audio):
    audio = audio.set_channels(1)
    segment_duration_ms = SEGMENT_DURATION * 300
    total_duration = len(audio)
    fingerprints = []
    start = 0
    while start + segment_duration_ms <= total_duration:
        segment = audio[start:start + segment_duration_ms]
        # Aplicar filtro de reducción de ruido (filtro de promedio móvil)
        samples = np.array(segment.get_array_of_samples())
        smoothed_samples = np.convolve(samples, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='same')  # Filtro de promedio móvil
        fft_values = np.abs(np.fft.fft(smoothed_samples, n=FFT_LENGTH)[:FFT_LENGTH // 2])  # Transformada de Fourier
        fingerprint = normalize_fingerprint(fft_values)
        fingerprints.append(fingerprint)
        start += segment_duration_ms
    return fingerprints


def build_database(songs_directory):
    database = {}
    song_files = [f for f in os.listdir(songs_directory) if f.endswith('.wav')]
    for filename in song_files:
        song_path = os.path.join(songs_directory, filename)
        song_name = os.path.splitext(filename)[0]
        audio = AudioSegment.from_file(song_path)
        fingerprints = generate_fingerprints(audio)
        database[song_name] = fingerprints
        print(f"Huellas digitales generadas para '{song_name}': {len(fingerprints)}")
    return database


def identify_song(database, target_audio):
    target_fingerprints = generate_fingerprints(target_audio)
    
    best_match = None
    best_match_score = float('inf')
    
    for song_name, fingerprints in database.items():
        total_score = 0
        count = 0
        for fp1 in target_fingerprints:
            min_dist = float('inf')
            for fp2 in fingerprints:
                try:
                    dist = distance.euclidean(fp1, fp2)
                except ValueError:  # Manejar excepción por valores inválidos
                    continue
                if dist < min_dist:
                    min_dist = dist
            total_score += min_dist
            count += 1
        
        if count > 0:
            average_score = total_score / count
        else:
            average_score = float('inf')
        
        if average_score < best_match_score:
            best_match = song_name
            best_match_score = average_score
    
    print(f"Coincidencia: {best_match}, Puntuación: {best_match_score}")
    
    return best_match, best_match_score

def load_and_identify(mode='file', target_song=None):
    if mode == 'file':
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
        if not file_path:
            messagebox.showwarning("Error", "Por favor selecciona un archivo de audio.")
            return
        target_audio = AudioSegment.from_file(file_path)
        identified_song, match_score = identify_song(database, target_audio)
        show_result(identified_song, match_score)
    elif mode =='mic':
        start_recording()
    elif mode =='spotify':
        if target_song is None:
            messagebox.showwarning("Error", "Debe especificar una canción del álbum.")
            return
    # Buscar la canción en Spotify
    search_params = {'q': target_song}
    search_response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=search_params)
    search_results = search_response.json()
    
    # Verificar si 'tracks' y 'items' existen y tienen elementos
    if 'tracks' in search_results and 'items' in search_results['tracks'] and len(search_results['tracks']['items']) > 0:
        track_uri = search_results['tracks']['items'][0]['uri']
        # Aquí iría el código para descargar la canción y procesarla
    else:
        messagebox.showwarning("Error", "No se encontraron resultados para la canción en Spotify.")


    
current_song = ""
    

def update_song_entry(event=None):
    global current_song
    current_song = song_entry.get()





def start_recording():
    global recording, recording_thread
    recording = True
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()


def record_audio():
    global recording
    frames = int(SAMPLE_RATE * SEGMENT_DURATION)
    recorded_audio = np.zeros((frames, 1), dtype='int16')

    def callback(indata, frames, time, status):
        if recording:
            recorded_audio[:len(indata)] = indata
        else:
            raise sd.CallbackStop()

    print(f"Capturando audio del micrófono ({SEGMENT_DURATION} segundos)...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback):
        sd.sleep(int(SEGMENT_DURATION * 1000))
    
    if recording:
        smoothed_audio = np.convolve(recorded_audio.flatten(), np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='same').astype('int16')
        target_audio = AudioSegment(
            smoothed_audio.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=smoothed_audio.dtype.itemsize,
            channels=1
        )
        
        plot_frequency_graph(recorded_audio.flatten(), SAMPLE_RATE)
        identified_song, match_score = identify_song(database, target_audio)
        show_result(identified_song, match_score)


def stop_recording():
    global recording
    recording = False
    if recording_thread is not None:
        recording_thread.join()
    print("Grabación detenida")


def show_result(identified_song, match_score):
    if match_score > THRESHOLD:
        messagebox.showinfo("Resultado", "La canción no pudo ser identificada.")
    else:
        messagebox.showinfo("Resultado", f"La canción identificada es: {identified_song}")


def plot_frequency_graph(audio_data, sample_rate):
    root.after(0, lambda: _plot_frequency_graph(audio_data, sample_rate))

def _plot_frequency_graph(audio_data, sample_rate):
    n = len(audio_data)
    freq = np.fft.rfftfreq(n, d=1/sample_rate)
    spectrum = np.abs(np.fft.rfft(audio_data, n=n))
    
    plt.figure(figsize=(8, 4))
    plt.plot(freq, spectrum)
    plt.title('Espectro de Frecuencia del Audio Capturado')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()


def on_closing():
    if messagebox.askokcancel("Salir", "¿Seguro que quieres salir?"):
        root.destroy()




if __name__ == "__main__":
    songs_directory = 'C:/Users/KINGU/Documents/Shazam/Canciones'
    
    if not os.path.exists(songs_directory):
        print(f"Error: El directorio {songs_directory} no existe. Por favor, crea el directorio y añade archivos.wav.")
    else:
        database = build_database(songs_directory)
        
        root = tk.Tk()
        root.title("Identificación de Canciones en Tiempo Real")
        
        # Añadir un estilo visual
        style = ttk.Style(root)
        style.theme_use('clam')

        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Identificación de Canciones", font=("Helvetica", 16))
        title_label.pack(pady=10)

        file_button = ttk.Button(main_frame, text="Cargar Archivo de Audio", command=lambda: load_and_identify('file'))
        file_button.pack(pady=10)
        
        mic_button = ttk.Button(main_frame, text="Capturar desde Micrófono", command=lambda: load_and_identify('mic'))
        mic_button.pack(pady=10)

        # Dentro de tu función principal o donde configures tus widgets
        song_entry = ttk.Entry(main_frame, width=30)
        song_entry.pack(pady=10)
        
        # Vincular el evento <KeyRelease> de song_entry para actualizar su valor
        song_entry.bind("<KeyRelease>", lambda event: update_song_entry())

        spotify_button = ttk.Button(main_frame, text="Cargar Datos de Spotify", command=lambda: load_and_identify('spotify', current_song))
        spotify_button.pack(pady=10)

        stop_button = ttk.Button(main_frame, text="Detener Grabación", command=stop_recording)
        stop_button.pack(pady=10)

        progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        progress.pack(pady=10)


        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()