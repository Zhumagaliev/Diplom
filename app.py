import pickle
import joblib
from IPython.display import display
from PIL import Image
from flask import Flask, jsonify
from tensorflow.keras.applications.inception_v3 import InceptionV3
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
# model = load_model("inceptionv3_fine_tuned.h5")


mm = open('joblib_model.pkl', 'rb')
model = joblib.load(mm)


# model = pickle.load(mm)
# print("Test")
# print(X_ray_model)

# X_ray_model = pickle.load(open('model.pkl', ''))
# modelfile = 'final_prediction.pickle'
# model = pickle.load(open(modelfile, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Размер изображения
    image_size = 299
    # Размер мини-выборки
    batch_size = 32

    # img_path = "C:/Users/Meirman/Datasets/val/Normal/IM-0005-0001.jpeg"
    # img_path = 'Datasets/test/Tuberculosis/00000011_007.png'
    # img = image.load_img(img_path, target_size=(image_size, image_size))
    pretrained_model = InceptionV3(weights='imagenet', include_top=False)
    pretrained_model.trainable = False
    x = pretrained_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=pretrained_model.input, outputs=predictions)

    # model = InceptionV3(weights='imagenet', include_top=False)
    pretrained_model.trainable = False
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=1e-4),
                  metrics=['accuracy'])
    # model.trainable = False
    # int_features = [float(x) for x in request.form.values()]
    # print("Meirman ddad dwad wa + ++dwa d")
    photo = request.files['photo']
    # print(X_ray_model)
    # print(photo)
    # print(photo.filename)

    # img1 = ''.join(img)
    # img2 = int(img1)

    # img_path = "C:/Users/Meirman/Datasets/val/Normal/"+photo.filename
    # img_path = "C:/Users/Meirman/Datasets/val/Normal/IM-0005-0001.jpeg"
    img = image.load_img(photo, target_size=(299, 299))

    # model = image.astype('float32')

    # print("Meirman ddad dwad wa + ++dwa d1")
    x = image.img_to_array(img)
    x /= 255
    # x -= 0.5
    # x *= 2.
    x = np.expand_dims(x, axis=0)

    # x = preprocess_input(x)
    # print(x)
    # x = preprocess_input(x)
    # query_df = pd.DataFrame(x)
    # query = pd.get_dummies(query_df)

    # print("Meirman ddad dwad wa + ++dwa d2")
    prediction = model.predict(x)
    # prediction = decode_predictions(prediction)
    # prediction = model.predict(query)
    # print("Meirman ddad dwad wa + ++dwa d3")
    # rounded = [np.round(x) for x in prediction]
    # print(rounded)
    # print(prediction)
    # print(photo)

    if prediction[[0]] < 0.5:
        return render_template('index.html', prediction_text="Normal")
    else:
        return render_template('index.html', prediction_text="Tuberculocis")

        # return render_template('index.html', prediction=prediction)


# @app.route('/results', methods=['POST'])
# def results():

if __name__ == '__main__':
    app.run(debug=True)
