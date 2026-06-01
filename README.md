# AI-Based Brain Tumor Detection System

An end-to-end Deep Learning project using **VGG19 Transfer Learning** to detect brain tumors from MRI scans with high confidence.

## Project Overview
This project uses Computer Vision to assist in medical imaging diagnosis. It features an automated image-processing pipeline that crops MRI scans to focus on the brain area, followed by a fine-tuned Convolutional Neural Network (CNN).

## Tools & Technologies
- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Model:** VGG19 (Transfer Learning)
- **Image Processing:** OpenCV & Imutils
- **UI:** Gradio (Web Interface)
- **Environment:** Google Colab

## Results
- **Validation Accuracy:** Over 80% (after fine-tuning)
- **Features:** Automatic skull-stripping/cropping, probability-based diagnosis, and real-time interactive UI.

## How to Use
1. Clone the repository: git clone https://github.com/malon328/brain-tumor-detection.git
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook in Google Colab or Jupyter to launch the Gradio interface.