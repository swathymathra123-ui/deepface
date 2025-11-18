import cv2
from deepface import DeepFace
import pyttsx3
import threading

# Initialize TTS engine
engine = pyttsx3.init()

# emotion spoken last time
last_spoken = ""

# Thread function to speak emotion (runs separately)
def speak_emotion(emotion):
    engine.say(emotion)
    engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    key, img = cap.read()
    
    if not key:
        break

    # Analyze emotion
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    emotion = results[0]['dominant_emotion']

    # Show emotion on screen
    cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Speak only when emotion changes
    if emotion != last_spoken:
        threading.Thread(target=speak_emotion, args=(emotion,), daemon=True).start()
        last_spoken = emotion

    cv2.imshow("Emotion Recognition", img)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()