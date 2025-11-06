import cv2
import mediapipe as mp
import math
import time

# ====== Settings ======
MIRROR = True          # True = mirror preview
MAX_HANDS = 2
DET_CONF = 0.6
TRK_CONF = 0.5

# ====== MediaPipe setup ======
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DET_CONF,
    min_tracking_confidence=TRK_CONF
)

# Landmark indices
WRIST = 0
TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4
IN_MCP, IN_PIP, IN_DIP, IN_TIP = 5, 6, 7, 8
MI_MCP, MI_PIP, MI_DIP, MI_TIP = 9,10,11,12
RI_MCP, RI_PIP, RI_DIP, RI_TIP = 13,14,15,16
PI_MCP, PI_PIP, PI_DIP, PI_TIP = 17,18,19,20

# ====== Helpers ======
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def angle(a, b, c):
    """Angle ABC in degrees (2D)."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot/(n1*n2)))
    return math.degrees(math.acos(cosang))

def finger_extended(lm, mcp, pip, dip, tip, wrist_idx=WRIST,
                    min_straight_deg=155, min_tip_wrist_margin=0.02):
    """Rotation-invariant finger extension test."""
    d_tip_wrist = dist(lm[tip], lm[wrist_idx])
    d_pip_wrist = dist(lm[pip], lm[wrist_idx])
    straight = angle(lm[mcp], lm[pip], lm[tip]) >= min_straight_deg
    return (d_tip_wrist > d_pip_wrist + min_tip_wrist_margin) and straight

def thumb_extended(lm, cmc=1, mcp=2, ip=3, tip=4, wrist_idx=WRIST,
                   min_straight_deg=150, min_tip_wrist_margin=0.015):
    """Thumb extended irrespective of direction."""
    d_tip_wrist = dist(lm[tip], lm[wrist_idx])
    d_mcp_wrist = dist(lm[mcp], lm[wrist_idx])
    straight = angle(lm[cmc], lm[ip], lm[tip]) >= min_straight_deg
    return (d_tip_wrist > d_mcp_wrist + min_tip_wrist_margin) and straight

def thumb_direction_up_or_down(lm, up_down_eps=0.02):
    """Check thumb pointing up/down (for thumbs up/down)."""
    dy = lm[TH_TIP].y - lm[TH_MCP].y  # image y grows downward
    if dy < -up_down_eps: return "up"
    if dy >  up_down_eps: return "down"
    return "neutral"

# ====== Gesture Classifier ======
def classify_gesture(lm, handed_label):
    index_up  = finger_extended(lm, IN_MCP, IN_PIP, IN_DIP, IN_TIP)
    middle_up = finger_extended(lm, MI_MCP, MI_PIP, MI_DIP, MI_TIP)
    ring_up   = finger_extended(lm, RI_MCP, RI_PIP, RI_DIP, RI_TIP)
    pinky_up  = finger_extended(lm, PI_MCP, PI_PIP, PI_DIP, PI_TIP)
    thumb_up_any = thumb_extended(lm)

    thumb_ud = thumb_direction_up_or_down(lm)

    others_folded = (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up)
    others_all_up = index_up and middle_up and ring_up and pinky_up
    two_up_V      = index_up and middle_up and (not ring_up) and (not pinky_up)

    # "OK" pinch (normalized by hand size)
    size = dist(lm[WRIST], lm[MI_MCP]) + 1e-6
    pinch_ok = math.hypot(lm[TH_TIP].x - lm[IN_TIP].x,
                          lm[TH_TIP].y - lm[IN_TIP].y) < 0.35 * size

    # --- Rules ---
    if others_folded and thumb_ud == "up":
        return "Thumbs Up"
    if others_folded and thumb_ud == "down":
        return "Thumbs Down"
    if two_up_V:
        return "Peace"
    if pinch_ok and not middle_up and not ring_up and not pinky_up:
        return "OK"
    if others_all_up and thumb_up_any:
        return "Open Palm"
    if others_folded and thumb_ud == "neutral":
        return "Fist"

    return "Unknown"

# ====== Video loop ======
cap = cv2.VideoCapture(0)
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if MIRROR:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w = frame.shape[:2]

    if result.multi_hand_landmarks:
        handedness_list = []
        if result.multi_handedness:
            for hnd in result.multi_handedness:
                handedness_list.append(hnd.classification[0].label)  # "Left"/"Right"

        for i, handLms in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            lm = handLms.landmark
            label = handedness_list[i] if i < len(handedness_list) else "Hand"

            gesture = classify_gesture(lm, label)

            # Put label near top of that hand
            min_y = min(int(p.y * h) for p in lm)
            min_x = min(int(p.x * w) for p in lm)
            text = f"{label}: {gesture}"
            cv2.putText(frame, text, (max(10, min_x - 10), max(30, min_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # FPS counter
    now = time.time()
    fps = 1.0 / (now - prev) if now > prev else 0.0
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Gesture Tracker (Improved)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
