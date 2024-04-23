from flask import Flask, request, render_template
from flask_restful import Resource, Api
import tensorflow as tf
from alibi_detect.models.tensorflow.losses import elbo
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)
api = Api(app)

# Load the pre-trained model
od = OutlierVAE.load('outlier_detection_model')

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

            # Visualization
            instance_score_plot = plot_instance_score(od_preds, np.zeros(1), ['normal', 'outlier'], od.threshold, show_plot=False)
            plt.close()

            x_recon = od.vae(x_val).numpy()
            feature_outlier_image = plot_feature_outlier_image(
                od_preds,
                img,
                X_recon=x_recon,
                max_instances=5,
                outliers_only=False,
                show_plot=False
            )
            plt.close()

            return render_template('index.html', prediction="Anomaly Score: {:.4f}".format(od_preds['data']['instance_score'][0]), instance_score_plot=instance_score_plot, feature_outlier_image=feature_outlier_image)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)