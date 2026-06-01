# %%
import zipfile
import os


zip_path = "C:\\Users\\nathn\\OneDrive\\Desktop\\archive.zip"
extract_path = "C:\\Users\\nathn\\OneDrive\\Desktop\\Brain-Tumor-Detection"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete!")
print(os.listdir(extract_path))

# %%
import os
import matplotlib.pyplot as plt

base_dir = 'C:\\Users\\nathn\\OneDrive\\Desktop\\Brain-Tumor-Detection\\brain_tumor_dataset'
categories = ['yes', 'no']

for cat in categories:
    folder = os.path.join(base_dir, cat)
    count = 1
    for filename in os.listdir(folder):
        if filename.startswith(('Y_', 'N_')):
            continue

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):

            suffix = 'Y' if cat == 'yes' else 'N'
            new_name = f"{suffix}_{count}.jpg"

            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                count += 1

print("Renaming completed")

# Data for Bar Graph
num_yes = len(os.listdir(os.path.join(base_dir, 'yes')))
num_no = len(os.listdir(os.path.join(base_dir, 'no')))

plt.figure(figsize=(6, 6))
plt.bar(['Tumorous', 'Non-Tumorous'], [num_yes, num_no], color=['red', 'green'])
plt.ylabel("No. of Brain Tumor Images")
plt.title("Count of Brain Tumor Images")
plt.show()

print(f"Total images: {num_yes + num_no}")

# %%
import shutil
import numpy as np

processed_dir = '/content/brain_tumor_processed'
train_dir = os.path.join(processed_dir, 'train')
val_dir = os.path.join(processed_dir, 'val')

for folder in [train_dir, val_dir]:
    for sub in ['yes', 'no']:
        os.makedirs(os.path.join(folder, sub), exist_ok=True)

def split_data(cat):
    src = os.path.join(base_dir, cat)
    files = os.listdir(src)
    np.random.shuffle(files)
    split = int(len(files) * 0.8)

    for i, f in enumerate(files):
        dest = train_dir if i < split else val_dir
        shutil.copy(os.path.join(src, f), os.path.join(dest, cat, f))

split_data('yes')
split_data('no')
print("Data Split Successfully (80% Train, 20% Val)")

# %%
import cv2
import imutils

def crop_brain_contour(image):
    # FIX: Ensure image is 8-bit integer for OpenCV
    if image.max() <= 1.0:
        image = (image * 255).astype('uint8')
    else:
        image = image.astype('uint8')

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    # Return resized float image (0-1) for the AI
    image = cv2.resize(image, (224, 224))
    return image.astype('float32') / 255.0

# %%
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    preprocessing_function=crop_brain_contour
)

val_datagen = ImageDataGenerator(preprocessing_function=crop_brain_contour)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Start Training
history = model.fit(train_gen, validation_data=val_gen, epochs=20)

# %%
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Model Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Model Loss')
plt.legend()

plt.show()

# Save final model
model.save('brain_tumor_model.h5')
print("Model saved as brain_tumor_model.h5")

# %%
# Unfreeze the last block of VGG19 for better accuracy
for layer in base_model.layers[-5:]:
    layer.trainable = True

# Re-compile with a very small learning rate
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train for 10 more epochs to "Fine-Tune" the brain textures
print("Fine-tuning the model for better accuracy...")
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save the improved model
model.save('brain_tumor_improved.h5')

# %%
print(train_gen.class_indices)

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 1. Unfreeze more of the VGG19 model to let it learn deeper features
for layer in base_model.layers[-8:]:
    layer.trainable = True

# 2. Use a slightly higher learning rate for a faster "jump" in learning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Add a callback that slows down learning if the model gets stuck
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

# 4. Train for 15 more epochs
print("Training deeper to improve confidence...")
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[reduce_lr]
)

model.save('brain_tumor_final.h5')

# %%
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import imutils

# Load the improved model
model = tf.keras.models.load_model('brain_tumor_improved.h5')

def predict_tumor(input_img):
    # Preprocessing
    img = input_img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Make Prediction
    prediction = model.predict(img)[0]
    result_index = np.argmax(prediction) # Get the index of the highest probability
    confidence = prediction[result_index] * 100

    if result_index == 1:
        return f"RESULT: Tumor Detected (Confidence: {confidence:.2f}%)"
    else:
        return f"RESULT: No Tumor Detected / Healthy (Confidence: {confidence:.2f}%)"

# Launch the Interface with a Textbox output
interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(),
    outputs=gr.Textbox(label="Diagnosis Result"),
    title="AI Brain Tumor Diagnostic System",
    description="Upload an MRI scan. The system will analyze the brain structure to detect abnormalities.",
    theme="soft"
)

interface.launch(share=True)

# %%
model.save('brain_tumor_final.h5')


