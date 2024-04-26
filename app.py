import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정
from flask import Flask, request, render_template, jsonify
from alibi_detect.utils.saving import load_detector
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# Load the pre-trained model
od = load_detector('outlier_detection_model')

# 결과를 저장할 폴더 경로
results_folder = 'results'

# 결과 폴더가 없는 경우 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 파일로 결과를 저장하는 함수
def save_result_image(fig, filename_prefix):
    if fig is None:
        return None
    
    img_path = os.path.join(results_folder, f"{filename_prefix}_{int(time.time())}.png")
    print("Saving image to:", img_path)  # 추가
    fig.savefig(img_path)
    plt.close(fig)  # 그림 객체 닫기
    return img_path

# Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file submitted. Please submit a file.")

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")

        if file:
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((64, 64))
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.
            img = img.reshape(1, 64, 64, 3)

            # Reconstruction
            x_recon = od.vae(img).numpy()

            # Prediction
            od_preds = od.predict(img, outlier_type='instance', return_feature_score=True, return_instance_score=True)

            # 시각화를 하지 않고 결과를 파일로 저장
            instance_score_plot = plot_instance_score(od_preds, [0], ['normal', 'outlier'], od.threshold)
            instance_score_plot_path = save_result_image(instance_score_plot, 'instance_score_plot')

            feature_outlier_image = plot_feature_outlier_image(od_preds, img, X_recon=x_recon, max_instances=5, outliers_only=False)
            feature_outlier_image_path = save_result_image(feature_outlier_image, 'feature_outlier_image')

            return render_template('index.html', prediction="Anomaly Score: {:.4f}".format(od_preds['data']['instance_score'][0]), instance_score_plot=instance_score_plot_path, feature_outlier_image=feature_outlier_image_path)

    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
                img = Image.open(file.stream).convert('RGB')
                img = img.resize((64, 64))
                img = np.asarray(img)
                img = img.astype(np.float32) / 255.
                img = img.reshape(1, 64, 64, 3)

                # Reconstruction
                x_recon = od.vae(img).numpy()

                # Prediction
                od_preds = od.predict(img, outlier_type='instance', return_feature_score=True, return_instance_score=True)
                instance_score = od_preds['data']['instance_score'][0]
                return jsonify(prediction="Anomaly Score: {:.4f}".format(instance_score))

if __name__ == '__main__':
    app.run(debug=True)
