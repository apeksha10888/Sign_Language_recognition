import cv2
import mediapipe as mp
import joblib


print("Loading model...")

model = joblib.load(
    "models/sign_model.pkl"
)

print("Model loaded")


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Camera not detected")
    exit()


print("Starting prediction...")


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame = cv2.flip(frame,1)


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    result = hands.process(rgb)


    prediction = ""


    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:


            landmarks = []


            for point in hand.landmark:

                landmarks.append(point.x)
                landmarks.append(point.y)
                landmarks.append(point.z)


            prediction = model.predict(
                [landmarks]
            )[0]


            draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


    cv2.putText(
        frame,
        "Sign: " + prediction,
        (30,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0,255,0),
        3
    )


    cv2.imshow(
        "Sign Language Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
