# Pneumonia Detector Web App

**The web app for this pneumonia detector: https://.herokuapp.com/**

This is a simple image classification web app to predict whether a chest X-rays image showed pneumonia.

The model for the web app was trained with the [Chest X-Ray Images (Pneumonia) dataset from Paul Mooney](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

<p align="center">
  <img width="300" height="300" src=https://github.com/marcellusruben/rock_paper_scissor_web_app/blob/master/IM-0001-0001.jpeg>
</p>

## Files

There are six files in this repo:

- Kaggle_pneumonia_Inception_v3.ipynb: Google Colab file to load the dataset, build the ML model, and train the model.
- pneumonia_app.py: Python file to build the web-app.
- pneumonia.h5: file contains the trained model architecture and its corresponding weights.
- setup.sh: file to setup your configuration on Heroku.
- requirements.txt: the file to tell Heroku which dependencies you need to build the app.
- Procfile: file to tell Heroku which and how the command and file should be executed.
