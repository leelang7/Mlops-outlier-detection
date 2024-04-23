from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, InputLayer
from alibi_detect.models.tensorflow.losses import elbo
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob

app = Flask(__name__)

# Load images
img_list = glob('Negative/*.jpg')
train_img_list, val_img_list = train_test_split(img_list, test_size=0.1, random_state=2021)

def img_to_np(fpaths, resize=True):
    img_array = []
    for fname in fpaths:
        try:
            img = Image.open(fname).convert('RGB')
            if(resize): 
                img = img.resize((64, 64))
            img_array.append(np.asarray(img))
        except:
            continue
    images = np.array(img_array)
    return images

x_train = img_to_np(train_img_list[:1000])
x_train = x_train.astype(np.float32) / 255.

x_val = img_to_np(val_img_list[:32])
x_val = x_val.astype(np.float32) / 255.

# Model
latent_dim = 1024

encoder_net = tf.keras.Sequential([
    InputLayer(input_shape=(64, 64, 3)),
    Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu)
])

decoder_net = tf.keras.Sequential([
    InputLayer(input_shape=(latent_dim,)),
    Dense(4 * 4 * 128),
    Reshape(target_shape=(4, 4, 128)),
    Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2DTranspose(32, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
])

od = OutlierVAE(
    threshold=.005,
    score_type='mse',
    encoder_net=encoder_net,
    decoder_net=decoder_net,
    latent_dim=latent_dim,
)

od.fit(x_train, epochs=30, verbose=True)

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
