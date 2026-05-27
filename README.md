# HandDraw Sketch Classifier

A hand-drawn sketch classification app using a Convolutional Neural Network (CNN) trained on the Google QuickDraw dataset. Draw in the air using your index finger via webcam, then classify your sketch in real time.

---

## Features

- Trains a CNN model to recognize hand-drawn sketches
- Uses the QuickDraw dataset to download and prepare training images
- Supports real-time hand gesture drawing via webcam using MediaPipe
- Classifies drawings using a trained CNN with PyTorch
- Two interfaces: a lightweight OpenCV app and a full PyQt5 GUI

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ideniaa/handdraw-classifier.git
cd handdraw-classifier
```

### 2. Install Dependencies

```bash
pip install torch torchvision numpy opencv-python pillow mediapipe PyQt5
```

---

## Project Structure

```
handdraw-classifier/
 ┣ data_download.py    # Downloads and processes the QuickDraw dataset
 ┣ train.py            # Trains the CNN model
 ┣ model.py            # Defines the CNN architecture and class labels
 ┣ drawing_app.py      # OpenCV-based drawing and classification app
 ┣ gui.py              # PyQt5 GUI version of the drawing app
 ┗ README.md
```

---

## Usage

### 1. Download the Dataset

```bash
python data_download.py
```

Downloads and extracts images for each category into `quickdraw_data/`.

### 2. Train the Model

```bash
python train.py
```

Trains the CNN and saves weights to `quickdraw_model.pth`.

### 3. Run the App

**OpenCV version (keyboard controls):**
```bash
python drawing_app.py
```

| Key | Action |
|-----|--------|
| `D` | Toggle drawing on/off |
| `C` | Clear the canvas |
| `S` | Save drawing and classify |
| `Esc` | Exit |

**PyQt5 GUI version:**
```bash
python gui.py
```

Use the on-screen buttons to start/stop drawing, clear the canvas, and classify your sketch.

---

## Model Architecture

A CNN with the following layers:

- 2 convolutional layers with ReLU activation
- MaxPooling layers to reduce spatial dimensions
- 2 fully connected layers for classification
- CrossEntropyLoss with the Adam optimizer

---

## Classes

The model is trained on 7 categories:

✅ Apple · ✅ Banana · ✅ Cat · ✅ Dog · ✅ Tree · ✅ Car · ✅ Fish

To add more classes, update the `class_labels` list in `model.py` and re-run `data_download.py` and `train.py`.

---

## Future Improvements

- Add data augmentation to improve model accuracy
- Expand to more QuickDraw categories
- Deploy as a web app using Flask or Streamlit

---

## License

This project is open-source and available under the [MIT License](LICENSE).
