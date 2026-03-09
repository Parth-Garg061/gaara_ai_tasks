import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_angle(a, b, c):

    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c) 


    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_coords(landmarks, landmark_name, w, h):
    
    lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return [lm.x * w, lm.y * h]
    

def check_pose(landmarks, w, h):
    
    feedback = []
    correct = True
    
    l_shoulder = get_coords(landmarks, "LEFT_SHOULDER", w, h)
    r_shoulder = get_coords(landmarks, "RIGHT_SHOULDER", w, h)

    l_elbow = get_coords(landmarks, "LEFT_ELBOW", w, h)
    r_elbow = get_coords(landmarks, "RIGHT_ELBOW", w, h)

    l_wrist = get_coords(landmarks, "LEFT_WRIST", w, h)
    r_wrist = get_coords(landmarks, "RIGHT_WRIST", w, h)

    l_hip = get_coords(landmarks, "LEFT_HIP", w, h)
    r_hip = get_coords(landmarks, "RIGHT_HIP", w, h)

    l_ankle = get_coords(landmarks, "LEFT_ANKLE", w, h)
    r_ankle = get_coords(landmarks, "RIGHT_ANKLE", w, h)

    l_ear = get_coords(landmarks, "LEFT_EAR", w, h)
    r_ear = get_coords(landmarks, "RIGHT_EAR", w, h)

    nose = get_coords(landmarks, "NOSE", w, h)

    head_size = abs(nose[1] - l_ear[1])

    hands_up = (l_wrist[1] < nose[1] - head_size) and (r_wrist[1] < nose[1] - head_size)

    left_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

    arms_straight = (left_arm_angle > 160) and (right_arm_angle > 160)

    wrist_distance = np.linalg.norm(np.array(l_wrist) - np.array(r_wrist))

    left_arm_to_ear = abs(l_wrist[0] - l_ear[0])
    right_arm_to_ear = abs(r_wrist[0] - r_ear[0])

    arms_near_ears = (left_arm_to_ear < 100) and (right_arm_to_ear < 100)

    feet_distance = abs(l_ankle[0] - r_ankle[0])

    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    
    palms_joined = wrist_distance < (shoulder_width * 0.3)

    feet_apart = (shoulder_width * 0.2) < feet_distance < (shoulder_width * 0.6)
   
    arms_down = (l_wrist[1] > l_hip[1]) and (r_wrist[1] > r_hip[1])


    if not feet_apart:
        feedback.append("Keep feet ~6 inches apart")
        correct = False

    if not arms_down:
        # Only check arm rules if arms are supposed to be UP
        if not hands_up:
            feedback.append("Raise arms fully above head")
            correct = False

        if not arms_straight:
            feedback.append("Straighten your arms fully")
            correct = False

        if not palms_joined:
            feedback.append("Join your palms together")
            correct = False

        if not arms_near_ears:
            feedback.append("Bring arms closer to ears")
            correct = False

    return correct, feedback, arms_down

BREATH_DURATION = 4.0

def get_breath_count(elapsed_seconds):
    return int(elapsed_seconds / BREATH_DURATION)
   
HOLD_BREATHS = 4

REST_BREATHS = 2

TOTAL_SETS = 3

def draw_feedback(frame, feedback, correct, phase, set_num, breath_count, target_breaths):
   
    h, w = frame.shape[:2]

    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    status_color = (0, 255, 0) if correct else (0, 0, 255)

    status_text = "CORRECT POSE" if correct else "INCORRECT POSE"
    cv2.putText(frame, status_text, (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    set_text = f"Set: {set_num}/{TOTAL_SETS}  |  Phase: {phase}"
    cv2.putText(frame, set_text, (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    breath_text = f"Breaths: {breath_count}/{target_breaths}"
    cv2.putText(frame, breath_text, (20, 100),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

    for i, msg in enumerate(feedback):
        cv2.putText(frame, f"  {msg}", (20, 130 + i * 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

    return frame

def main():

    cap = cv2.VideoCapture(0)
    
    current_set = 1         
    phase = "ARMS_UP"       
    phase_start = time.time()  
    exercise_done = False   

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
          
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elapsed = time.time() - phase_start
            if phase == "ARMS_UP":
                target_breaths = HOLD_BREATHS 
            else:
                target_breaths = REST_BREATHS 

            breath_count = min(get_breath_count(elapsed), target_breaths)
            if breath_count >= target_breaths and not exercise_done:
                if phase == "ARMS_UP":
                    phase = "REST"
                    phase_start = time.time()
                else:
                    if current_set < TOTAL_SETS:
                        current_set += 1
                        phase = "ARMS_UP"
                        phase_start = time.time()
                    else:
                        exercise_done = True

            if exercise_done:
                cv2.putText(frame, "EXERCISE COMPLETE! Great job!",
                            (w // 2 - 250, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                correct, feedback, arms_down = check_pose(landmarks, w, h)
                skeleton_color = (0, 255, 0) if correct else (0, 0, 255)
                dot_style = mp_drawing.DrawingSpec(
                    color=skeleton_color, thickness=4, circle_radius=5)
                line_style = mp_drawing.DrawingSpec(
                    color=skeleton_color, thickness=3)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=dot_style,
                    connection_drawing_spec=line_style
                )
                frame = draw_feedback(
                    frame, feedback, correct,
                    phase, current_set, breath_count, target_breaths
                )

            else:
                cv2.putText(frame, "No person detected — step into frame",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 165, 255), 2)

            cv2.imshow("Exercise Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
