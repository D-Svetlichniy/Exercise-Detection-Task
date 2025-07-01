import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def get_elbow_angle(landmarks, side='RIGHT'):
    shoulder = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value]
    elbow = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value]
    wrist = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value]
    return calculate_angle(shoulder, elbow, wrist)


def get_knee_angle(landmarks, side='RIGHT'):
    hip = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value]
    knee = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value]
    ankle = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value]
    return calculate_angle(hip, knee, ankle)


def classify_exercise(landmarks):
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    return "Push-up" if abs(shoulder - hip) < 0.1 else "Squat"


def draw_angle(frame, p1, p2, p3, angle, width, height):
    color = (0, 255, 255)
    thickness = 2

    x1, y1 = int(p1.x * width), int(p1.y * height)
    x2, y2 = int(p2.x * width), int(p2.y * height)
    x3, y3 = int(p3.x * width), int(p3.y * height)

    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.line(frame, (x3, y3), (x2, y2), color, thickness)

    cv2.putText(frame, f'{int(angle)}', (x2 + 10, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


class RepeatCounter:
    def __init__(self, exercise):
        self.count = 0
        self.exercise = exercise
        self.pos = None

    def update(self, angle):
        if self.exercise == "Squat":
            if angle < 150:
                self.pos = "down"
            elif angle > 160 and self.pos == "down":
                self.pos = "up"
                self.count += 1
        if self.exercise == "Push-up":
            if angle < 110:
                self.pos = "down"
            elif angle > 140 and self.pos == "down":
                self.pos = "up"
                self.count += 1


cap = cv2.VideoCapture("Exercises.mp4")
# cap = cv2.VideoCapture("Test_video.mp4")
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

squat_counter = RepeatCounter("Squat")
pushup_counter = RepeatCounter("Push-up")

stable_counter = 0
last_detected = None
current_exercise = None
stable_threshold = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        new_detected = classify_exercise(landmarks)

        if new_detected == last_detected:
            stable_counter += 1
        else:
            stable_counter = 0
            last_detected = new_detected

        if stable_counter >= stable_threshold:
            current_exercise = new_detected

        if current_exercise == "Squat":
            angle = get_knee_angle(landmarks)
            hip, knee, ankle = [landmarks[i] for i in (
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value)]
            draw_angle(image, hip, knee, ankle, angle, width, height)
            squat_counter.update(angle)

        elif current_exercise == "Push-up":
            angle = get_elbow_angle(landmarks)
            shoulder, elbow, wrist = [landmarks[i] for i in (
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                mp_pose.PoseLandmark.RIGHT_WRIST.value)]
            draw_angle(image, shoulder, elbow, wrist, angle, width, height)
            pushup_counter.update(angle)

    cv2.putText(image, f'Exercise: {current_exercise}', (20, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.putText(image, f'Squats: {squat_counter.count}', (20, 80),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(image, f'Push-ups: {pushup_counter.count}', (20, 120),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 255), 2)

    out.write(image)
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

