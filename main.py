import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GraduateRPPG:
    def __init__(self, buffer_size=150, fs=30):
        self.buffer_size = buffer_size
        self.fs = fs
        self.rgb_buffer = []
        self.bpm = 0
        
        # New Tasks API Setup
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_roi_mean(self, frame, landmarks, indices):
        h, w, _ = frame.shape
        # New Tasks API landmarks are accessed by index directly
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        mean_val = cv2.mean(frame, mask=mask)[:3]
        return mean_val

    def pos_method(self, rgb_data):
        mean_rgb = np.mean(rgb_data, axis=0)
        norm_rgb = rgb_data / mean_rgb
        s1 = norm_rgb[:, 1] - norm_rgb[:, 2]
        s2 = norm_rgb[:, 1] + norm_rgb[:, 2] - 2 * norm_rgb[:, 0]
        
        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        h = s1 + (alpha * s2)
        h = signal.detrend(h)
        b, a = signal.butter(2, [0.7 / (self.fs / 2), 3.0 / (self.fs / 2)], btype='bandpass')
        return signal.filtfilt(b, a, h)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            # Convert to MediaPipe Image object (Required for Tasks API)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect landmarks
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                lm = detection_result.face_landmarks[0]
                
                # ROI Indices
                fh_ids = [10, 67, 103, 109, 151, 338, 297, 332]
                lc_ids = [234, 93, 132, 58, 172, 136, 150, 149]
                rc_ids = [454, 323, 361, 288, 397, 365, 379, 378]
                
                m1 = self.get_roi_mean(frame, lm, fh_ids)
                m2 = self.get_roi_mean(frame, lm, lc_ids)
                m3 = self.get_roi_mean(frame, lm, rc_ids)
                avg_rgb = np.mean([m1, m2, m3], axis=0)
                
                self.rgb_buffer.append(avg_rgb)
                if len(self.rgb_buffer) > self.buffer_size:
                    self.rgb_buffer.pop(0)
                
                if len(self.rgb_buffer) == self.buffer_size:
                    pulse_sig = self.pos_method(np.array(self.rgb_buffer))
                    fft = np.abs(np.fft.rfft(pulse_sig))
                    freqs = np.fft.rfftfreq(len(pulse_sig), 1/self.fs)
                    
                    # Target 0.7Hz to 3.0Hz (42-180 BPM)
                    mask = (freqs >= 0.7) & (freqs <= 3.0)
                    if np.any(mask):
                        self.bpm = freqs[mask][np.argmax(fft[mask])] * 60
                
                # Visualization
                for ids in [fh_ids, lc_ids, rc_ids]:
                    pts = np.array([(int(lm[i].x * frame.shape[1]), int(lm[i].y * frame.shape[0])) for i in ids])
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 1)

                cv2.putText(frame, f"BPM: {self.bpm:.1f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Stable rPPG - Tasks API', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    GraduateRPPG().run()