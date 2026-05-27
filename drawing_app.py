import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
from model import SketchCNN, class_labels  # Import trained model and labels

# Initialize MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

# Load trained model
num_classes = len(class_labels)
model = SketchCNN(num_classes)
model.load_state_dict(torch.load("quickdraw_model.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Define image transformation for classification
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Flag to track drawing status
drawing = False
previous_point = None

# Helper function for line interpolation
def interpolate_points(start, end, num_points=5):
    x_vals = np.linspace(start[0], end[0], num_points)
    y_vals = np.linspace(start[1], end[1], num_points)
    return list(zip(x_vals, y_vals))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip

            # Convert to pixel coordinates
            x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)

            if drawing and previous_point:
                # Interpolate between previous and current point
                interpolated_points = interpolate_points(previous_point, (x, y), num_points=10)
                for point in interpolated_points:
                    cv2.circle(canvas, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1)

            # Draw the current point if drawing is enabled
            if drawing:
                cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)

            # Update previous point
            previous_point = (x, y)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # FIX: Reset previous_point when hand leaves frame to prevent
        # unintended lines when the hand re-enters
        previous_point = None

    # Combine original frame and canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Hand Drawing", combined)

    key = cv2.waitKey(1)
    if key == ord('d'):  # Press 'd' to toggle drawing mode
        drawing = not drawing
        previous_point = None  # FIX: Reset so no line jumps on re-enable
        print("Drawing mode:", "Enabled" if drawing else "Disabled")

    if key == ord('c'):  # Press 'c' to clear the drawing
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Canvas cleared!")

    if key == ord('s'):  # Save drawing and classify it
        cv2.imwrite("drawing.png", canvas)
        print("Drawing saved. Classifying...")

        # Load and preprocess the saved drawing
        image = Image.open("drawing.png").convert("L")  # Convert to grayscale
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict class
        with torch.no_grad():
            output = model(image)
            predicted_class = output.argmax().item()
            print(f"Predicted drawing: {class_labels[predicted_class]}")

    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
