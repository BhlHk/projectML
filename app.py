from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
import cv2
from ultralytics import YOLO
import os

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger le modèle YOLOv8s
model = YOLO('best.pt')

# Variable de contrôle pour arrêter la détection
stop_detection = False

# Fonction de traitement vidéo pour la détection en temps réel
def generate_frames():
    global stop_detection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    while True:
        if stop_detection:
            break

        success, frame = cap.read()
        if not success:
            break

        # Effectuer la détection avec YOLOv8
        results = model(frame)

        # Annoter les résultats sur l'image
        annotated_frame = results[0].plot()

        # Encoder le cadre annoté au format JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)

        # Générer les trames pour le flux vidéo
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# Route pour servir les fichiers téléchargés
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route pour la page principale
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la page de détection en temps réel
@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

# Route pour le flux vidéo
@app.route('/video_feed')
def video_feed():
    global stop_detection
    stop_detection = False
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route pour arrêter la détection
@app.route('/stop_detection', methods=['POST'])
def stop():
    global stop_detection
    stop_detection = True
    return redirect(url_for('realtime'))

# Route pour l'upload de fichier (photo ou vidéo)
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Vérifier si le fichier est une image ou une vidéo
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Traiter comme une image
        frame = cv2.imread(file_path)
        results = model(frame)

        # Annoter les résultats sur l'image
        annotated_frame = results[0].plot()
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'annotated_{file.filename}')
        cv2.imwrite(output_path, annotated_frame)

        return redirect(url_for('result', result_image=f'annotated_{file.filename}'))

    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        # Traiter comme une vidéo
        cap = cv2.VideoCapture(file_path)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'annotated_{file.filename}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            if out is None:
                height, width, _ = annotated_frame.shape
                out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

            out.write(annotated_frame)

        cap.release()
        out.release()

        return redirect(url_for('result_video', result_video=f'annotated_{file.filename}'))

    else:
        return "Format de fichier non pris en charge", 400

# Route pour afficher le résultat de la détection sur une image
@app.route('/result')
def result():
    result_image = request.args.get('result_image')
    return render_template('result.html', result_image=result_image)

# Route pour afficher le résultat de la détection sur une vidéo
@app.route('/result_video')
def result_video():
    result_video = request.args.get('result_video')
    return render_template('result_video.html', result_video=result_video)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
