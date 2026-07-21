import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

NUM_CLASSES = 5
IMAGE_SIZE = (224, 224, 3)

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=IMAGE_SIZE
)

for layer in base_model.layers:
    layer.trainable = False

# ✅ CORRECT
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.save("knee_vgg16.h5")

print("✅ Fixed VGG16 model saved successfully")
