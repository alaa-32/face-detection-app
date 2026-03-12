# Face Detection App

A simple computer vision project for detecting faces in images using OpenCV and Streamlit.

## Overview

This project allows users to upload an image or take a photo with a camera, then detect faces and visualize the results in a simple web interface.

## Tools Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow

## Method

This project uses a classical computer vision approach for face detection with OpenCV’s Haar cascade classifier. It detects faces in an image and draws bounding boxes around them.

## Features

- Upload an image or use the camera
- Detect faces in real time
- Adjust detection settings
- Change rectangle color and thickness
- Save or download the processed image

## Try It

You can test the app here: [Streamlit App]([https://face-detection-app-8b4ehd8raxbxhzanddafxp.streamlit.app/](https://face-detection-app-8b4ehd8raxbxhzanddafxp.streamlit.app))

## Run Locally

```bash
git clone https://github.com/alaa-32/face-detection-app.git
cd face-detection-app
pip install -r requirements.txt
streamlit run main.py
