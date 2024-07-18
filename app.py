import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정
from flask import Flask, request, jsonify
from alibi_detect.utils.saving import load_detector
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model
od = load_detector('outlier_detection_model')

current_dir = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(current_dir, 'results')

# 결과 폴더가 없는 경우 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 파일로 결과를 저장하는 함수
def save_result_image(fig, filename_prefix, original_filename):
    if fig is None:
        print("Error: Figure is None")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = os.path.splitext(original_filename)[0]  # Extract base filename without extension
    img_path = os.path.join(results_folder, f"{base_filename}_{filename_prefix}_{timestamp}.png")
    
    print(f"Saving image to: {img_path}")

    try:
        fig.savefig(img_path)
        print("Image saved successfully")
        plt.close(fig)  # 그림 객체 닫기
        return img_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return jsonify(error="No file provided"), 400

        original_filename = file.filename  # Get the original filename

        try:
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((64, 64))
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.
            img = img.reshape(1, 64, 64, 3)

            # Reconstruction
            x_recon = od.vae(img).numpy()

            # Prediction
            od_preds = od.predict(img, outlier_type='instance', return_feature_score=True, return_instance_score=True)
            
            # For visualization, use x_test which is the original image data received from the client
            x_test = img.reshape(1, 64, 64, 3)

            # Debug information
            print("od_preds:", od_preds)
            print("x_test shape:", x_test.shape)
            print("x_recon shape:", x_recon.shape)

            # Generate instance score plot and save it
            plot_instance_score(od_preds, [0], ['normal', 'outlier'], od.threshold)
            instance_score_plot_path = save_result_image(plt.gcf(), 'instance_score_plot', original_filename)

            # Generate feature outlier image plot
            plot_feature_outlier_image(od_preds, x_test, X_recon=x_recon, max_instances=5, outliers_only=False)
            feature_outlier_image_path = save_result_image(plt.gcf(), 'feature_outlier_image', original_filename)

            if instance_score_plot_path and feature_outlier_image_path:
                return jsonify(
                    prediction="Anomaly Score: {:.4f}".format(od_preds['data']['instance_score'][0]),
                    instance_score_plot_path=instance_score_plot_path,
                    feature_outlier_image_path=feature_outlier_image_path
                )
            else:
                return jsonify(
                    prediction="Anomaly Score: {:.4f}".format(od_preds['data']['instance_score'][0]),
                    error="Error saving visualizations"
                ), 500

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify(error="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True)
