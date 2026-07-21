import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

NUM_CLASSES = 5
IMAGE_SIZE = (224, 224, 3)

# Base model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=IMAGE_SIZE
)

# Freeze base model (optional but recommended)
for layer in base_model.layers:
    layer.trainable = False

# ✅ CORRECT: tensor, NOT list
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

# Save fixed model
model.save("knee_mobilenet.h5")

print("✅ Fixed MobileNetV2 model saved successfully")
