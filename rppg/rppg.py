# ================================================================
# rppg.py — Camera se Heart Rate Detect karna
# Features: Live BPM + 3 readings average
# ================================================================


# ================================================================
# SECTION 1 — LIBRARIES
# ================================================================

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_tasks
from scipy.signal import butter, filtfilt
import urllib.request
import os


# ================================================================
# SECTION 2 — MODEL PATH
# ================================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


# ================================================================
# SECTION 3 — MODEL DOWNLOAD
# ================================================================

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model download ho raha hai...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[INFO] Model download complete!")


# ================================================================
# SECTION 4 — BANDPASS FILTER
# ================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


# ================================================================
# SECTION 5 — HEART RATE CALCULATE (FFT)
# ================================================================

def extract_heart_rate(frames, fps=30):
    """
    POS Algorithm — Plane Orthogonal to Skin
    RGB teeno channels use karta hai — green only se zyada accurate
    """
    if len(frames) < 90:
        return None

    # ── Step 1: RGB teeno channels alag karo ──
    r_means, g_means, b_means = [], [], []

    for frame in frames:
        if frame is not None and frame.size > 0:
            r_means.append(np.mean(frame[:, :, 2]))  # Red
            g_means.append(np.mean(frame[:, :, 1]))  # Green
            b_means.append(np.mean(frame[:, :, 0]))  # Blue

    if len(r_means) < 90:
        return None

    r = np.array(r_means)
    g = np.array(g_means)
    b = np.array(b_means)

    # ── Step 2: Normalize (lighting changes hatao) ──
    r = r / (np.mean(r) + 1e-6)
    g = g / (np.mean(g) + 1e-6)
    b = b / (np.mean(b) + 1e-6)

    # ── Step 3: POS Algorithm ──
    S1 = r - g
    S2 = r + g - 2 * b
    alpha = (np.std(S1) + 1e-6) / (np.std(S2) + 1e-6)
    pulse = S1 + alpha * S2

    # ── Step 4: Signal normalize karo ──
    pulse = (pulse - np.mean(pulse)) / (np.std(pulse) + 1e-6)

    # ── Step 5: Bandpass filter ──
    b_coef, a_coef = butter_bandpass(0.5, 4.0, fps)
    filtered = filtfilt(b_coef, a_coef, pulse)

    # ── Step 6: FFT ──
    fft_vals = np.abs(np.fft.rfft(filtered))
    fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)

    # ── Step 7: Realistic range (42-210 BPM) ──
    valid_idx = np.where((fft_freqs >= 0.7) & (fft_freqs <= 3.5))[0]
    if len(valid_idx) == 0:
        return None

    # ── Step 8: Peak frequency → BPM ──
    peak_freq = fft_freqs[valid_idx[np.argmax(fft_vals[valid_idx])]]
    return round(peak_freq * 60, 1)


# ================================================================
# SECTION 6 — EK READING LENA (5 seconds)
# ================================================================

def single_scan(detector, duration=5, fps=30):
    """
    Ek baar scan karo — 5 sec
    Return: BPM ya None
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Camera open nahi hua"

    forehead_frames = []
    total_needed = duration * fps
    forehead_points = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
    live_bpm = None  # Real-time BPM

    while len(forehead_frames) < total_needed:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            fx_coords = [int(landmarks[i].x * w) for i in forehead_points]
            fy_coords = [int(landmarks[i].y * h) for i in forehead_points]

            fx_min = max(0, min(fx_coords) - 10)
            fx_max = min(w, max(fx_coords) + 10)
            fy_min = max(0, min(fy_coords) - 20)
            fy_max = max(0, max(fy_coords) + 10)

            forehead = frame[fy_min:fy_max, fx_min:fx_max]
            if forehead.size > 0:
                forehead_frames.append(forehead)
                cv2.rectangle(frame, (fx_min, fy_min), (fx_max, fy_max), (0, 255, 0), 2)
                cv2.putText(frame, "Forehead detected ✓",
                            (fx_min, fy_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── LIVE BPM: Har 30 frames pe update karo ──
            if len(forehead_frames) >= 60 and len(forehead_frames) % 30 == 0:
                live_bpm = extract_heart_rate(forehead_frames[-90:], fps)

        else:
            cv2.putText(frame, "Chehra nahi dikh raha — paas aao!",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # ── PROGRESS BAR ──
        progress = int((len(forehead_frames) / total_needed) * 100)
        bar_w = int((progress / 100) * 400)
        cv2.rectangle(frame, (10, 15), (410, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 15), (10 + bar_w, 40), (0, 200, 100), -1)
        cv2.putText(frame, f"Scanning: {progress}%",
                    (420, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        # ── LIVE BPM DISPLAY ──
        if live_bpm:
            cv2.rectangle(frame, (10, 50), (260, 85), (0, 0, 0), -1)
            cv2.putText(frame, f"Live BPM: {live_bpm}",
                        (15, 78), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)
            # Cyan color mein live BPM dikhao

        cv2.namedWindow('Heart Rate Scan', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Heart Rate Scan', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Heart Rate Scan', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    return extract_heart_rate(forehead_frames, fps), None


# ================================================================
# SECTION 7 — MAIN FUNCTION: 3 READINGS AVERAGE
# ================================================================

def capture_heart_rate(duration=15, fps=30):
    """
    3 baar 5-second scan karo
    Teeno ka average nikalo
    Return: {"success": True, "heart_rate": 72.5, "readings": [70, 73, 74]}
    """
    download_model()

    # MediaPipe setup
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    readings = []      # Teeno readings yahan store hongi
    total_rounds = 3   # 3 baar scan karenge

    for round_num in range(1, total_rounds + 1):
        print(f"[INFO] Reading {round_num}/{total_rounds} shuru...")

        # Ek reading lo
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            detector.close()
            return {"success": False, "error": "Camera open nahi hua"}

        forehead_frames = []
        total_needed = 5 * fps  # 5 seconds per reading
        forehead_points = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
        live_bpm = None

        while len(forehead_frames) < total_needed:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                fx_coords = [int(landmarks[i].x * w) for i in forehead_points]
                fy_coords = [int(landmarks[i].y * h) for i in forehead_points]

                fx_min = max(0, min(fx_coords) - 10)
                fx_max = min(w, max(fx_coords) + 10)
                fy_min = max(0, min(fy_coords) - 20)
                fy_max = max(0, max(fy_coords) + 10)

                forehead = frame[fy_min:fy_max, fx_min:fx_max]
                if forehead.size > 0:
                    forehead_frames.append(forehead)
                    cv2.rectangle(frame, (fx_min, fy_min), (fx_max, fy_max), (0, 255, 0), 2)
                    cv2.putText(frame, "Forehead detected",
                                (fx_min, fy_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Live BPM update
                if len(forehead_frames) >= 60 and len(forehead_frames) % 30 == 0:
                    live_bpm = extract_heart_rate(forehead_frames[-90:], fps)

            else:
                cv2.putText(frame, "Chehra nahi dikh raha!",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            # Round number dikhao
            cv2.putText(frame, f"Reading {round_num}/3",
                        (w - 160, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 200, 0), 2)

            # Progress bar
            progress = int((len(forehead_frames) / total_needed) * 100)
            bar_w = int((progress / 100) * 400)
            cv2.rectangle(frame, (10, 15), (410, 40), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 15), (10 + bar_w, 40), (0, 200, 100), -1)
            cv2.putText(frame, f"Scanning: {progress}%",
                        (420, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

            # Live BPM display
            if live_bpm:
                cv2.rectangle(frame, (10, 50), (260, 85), (0, 0, 0), -1)
                cv2.putText(frame, f"Live BPM: {live_bpm}",
                            (15, 78), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)

            # Previous readings dikhao
            for i, r in enumerate(readings):
                cv2.putText(frame, f"Reading {i+1}: {r} BPM",
                            (10, h - 60 + (i * 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (180, 255, 180), 1)

            cv2.namedWindow('Heart Rate Scan', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Heart Rate Scan', cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow('Heart Rate Scan', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                if readings:
                    avg = round(sum(readings) / len(readings), 1)
                    return {"success": True, "heart_rate": avg, "readings": readings}
                return {"success": False, "error": "Q dabake band kar diya"}

        cap.release()

        # Is round ka result nikalo
        hr = extract_heart_rate(forehead_frames, fps)
        if hr:
            readings.append(hr)
            print(f"[INFO] Reading {round_num}: {hr} BPM")

        # 2 second break dikhao readings ke beech
        if round_num < total_rounds:
            blank = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(blank, f"Reading {round_num}: {hr} BPM done!",
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 100), 2)
            cv2.putText(blank, f"Next reading in 2 sec...",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1)
            cv2.namedWindow('Heart Rate Scan', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Heart Rate Scan', cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow('Heart Rate Scan', blank)
            cv2.waitKey(2000)  # 2 second wait

    cv2.destroyAllWindows()
    detector.close()

    # ── FINAL AVERAGE CALCULATE KARO ──
    if not readings:
        return {"success": False, "error": "Koi reading nahi mili, dobara try karo"}

    avg_hr = round(sum(readings) / len(readings), 1)
    print(f"[INFO] Final Average: {avg_hr} BPM from readings {readings}")

    return {
        "success": True,
        "heart_rate": avg_hr,
        "readings": readings  # Teeno readings bhi bhejo
    }