import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Layer, Reshape, InputLayer
from alibi_detect.models.tensorflow.losses import elbo
from alibi_detect.od import OutlierVAE
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from glob import glob
from alibi_detect.utils.saving import save_detector


# Function to convert images to numpy arrays
def img_to_np(fpaths, resize=True):  
    img_array = []
    for fname in fpaths:
        try:
            img = Image.open(fname).convert('RGB')
            if resize: 
                img = img.resize((64, 64))
            img_array.append(np.asarray(img))
        except:
            continue
    images = np.array(img_array)
    return images

# Load and preprocess training data
img_list = glob('Negative/*.jpg')
train_img_list, val_img_list = train_test_split(img_list, test_size=0.1, random_state=2021)
x_train = img_to_np(train_img_list[:1000])
x_train = x_train.astype(np.float32) / 255.

# Define model architecture
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

# Train the outlier detection model
od = OutlierVAE(
    threshold=.005,
    score_type='mse',
    encoder_net=encoder_net,
    decoder_net=decoder_net,
    latent_dim=latent_dim,
)
od.fit(x_train, epochs=1, verbose=True)

# Save the model
save_detector(od, 'outlier_detection_model')