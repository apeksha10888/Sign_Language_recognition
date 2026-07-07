import cv2
import mediapipe as mp
import numpy as np
import os


label = input("Enter sign name (A/B/C/D/E): ")


save_path = "dataset"


os.makedirs(save_path,exist_ok=True)



mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)


mp_draw = mp.solutions.drawing_utils



cap=cv2.VideoCapture(0)



data=[]


count=0


while True:


    ret,frame=cap.read()

    frame=cv2.flip(frame,1)


    rgb=cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    result=hands.process(rgb)



    if result.multi_hand_landmarks:


        for hand in result.multi_hand_landmarks:


            landmarks=[]


            for point in hand.landmark:

                landmarks.append(point.x)
                landmarks.append(point.y)
                landmarks.append(point.z)



            data.append(landmarks)

            count+=1



            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )



    cv2.putText(
        frame,
        f"{label}: {count}/50",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )



    cv2.imshow(
        "Collecting",
        frame
    )



    if count>=50:
        break



    if cv2.waitKey(1)==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()



np.save(
    f"{save_path}/{label}.npy",
    np.array(data)
)


print("Saved:",label)
