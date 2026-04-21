import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class RobustRPPG:
    def __init__(self, buffer_size=300, fs=30):
        self.buffer_size = buffer_size
        self.fs = fs
        self.rgb_buffer = []
        self.bpm = 0.0
        self.confidence = 0.0
        self.signal_locked = False
        self.show_waveform = True        # Toggle with 'w' key
        self.last_pulse_sig = None       # For waveform overlay
        self.frame_count = 0

        # CLAHE preprocessor for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # MediaPipe Face Landmarker (Tasks API)
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    # -------------------------------------------------------------------------
    # Preprocessing: CLAHE on luminance channel
    # -------------------------------------------------------------------------
    def preprocess_frame(self, frame):
        """
        Apply CLAHE to the Y (luminance) channel in YCrCb space.
        This locally normalizes brightness so sudden lighting changes
        (e.g. flashlight) don't spike the raw RGB values fed to POS/CHROM.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # -------------------------------------------------------------------------
    # ROI Extraction with skin-pixel masking
    # -------------------------------------------------------------------------
    def get_roi_mean(self, frame, landmarks, indices):
        """
        Extract mean RGB from a landmark polygon, filtered to skin pixels only.
        An HSV skin-color mask is ANDed with the polygon to exclude non-skin
        pixels (hair, eyebrows, specular highlights) that add noise.
        """
        h, w, _ = frame.shape
        points = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        )

        # Polygon mask from facial landmarks
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [points], 255)

        # HSV skin-color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 20, 50), (50, 255, 255))

        # Combined: only skin pixels inside the polygon
        combined_mask = cv2.bitwise_and(poly_mask, skin_mask)

        # Fallback: if skin mask is too aggressive, use polygon alone
        if cv2.countNonZero(combined_mask) < 30:
            combined_mask = poly_mask

        mean_val = cv2.mean(frame, mask=combined_mask)[:3]
        return mean_val

    # -------------------------------------------------------------------------
    # POS Method (Plane-Orthogonal-to-Skin)
    # -------------------------------------------------------------------------
    def pos_method(self, rgb_data):
        """
        POS projects normalized RGB onto a plane orthogonal to the skin-tone
        vector, suppressing specular reflections and enhancing the blood
        volume pulse component.
        """
        mean_rgb = np.mean(rgb_data, axis=0)
        mean_rgb[mean_rgb == 0] = 1e-8
        norm_rgb = rgb_data / mean_rgb

        s1 = norm_rgb[:, 1] - norm_rgb[:, 2]                        # G - B
        s2 = norm_rgb[:, 1] + norm_rgb[:, 2] - 2 * norm_rgb[:, 0]   # G + B - 2R

        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        h = s1 + alpha * s2
        return signal.detrend(h)

    # -------------------------------------------------------------------------
    # CHROM Method (Chrominance-based, De Haan & Jeanne 2013)
    # -------------------------------------------------------------------------
    def chrom_method(self, rgb_data):
        """
        CHROM builds two orthogonal chrominance signals and combines them
        to cancel the illumination-dependent component. Complementary to
        POS under different lighting conditions.
        """
        mean_rgb = np.mean(rgb_data, axis=0)
        mean_rgb[mean_rgb == 0] = 1e-8
        norm_rgb = rgb_data / mean_rgb

        xf = 3.0 * norm_rgb[:, 0] - 2.0 * norm_rgb[:, 1]               # 3R - 2G
        yf = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]  # 1.5R + G - 1.5B

        alpha = np.std(xf) / (np.std(yf) + 1e-8)
        h = xf - alpha * yf
        return signal.detrend(h)

    # -------------------------------------------------------------------------
    # Bandpass filter
    # -------------------------------------------------------------------------
    def bandpass(self, sig):
        """Butterworth bandpass: 0.7–3.0 Hz (42–180 BPM)."""
        nyq = self.fs / 2.0
        low = max(0.7 / nyq, 0.01)
        high = min(3.0 / nyq, 0.99)
        b, a = signal.butter(2, [low, high], btype='bandpass')
        return signal.filtfilt(b, a, sig)

    # -------------------------------------------------------------------------
    # Combined pulse extraction: average of POS and CHROM
    # -------------------------------------------------------------------------
    def fused_pulse(self, rgb_array):
        """
        Run both POS and CHROM, bandpass filter each, then average.
        Averaging two complementary methods reduces noise that affects
        only one method while preserving the common pulse signal.
        """
        pulse_pos = self.bandpass(self.pos_method(rgb_array))
        pulse_chrom = self.bandpass(self.chrom_method(rgb_array))
        return (pulse_pos + pulse_chrom) / 2.0

    # -------------------------------------------------------------------------
    # BPM estimation via FFT with confidence measure
    # -------------------------------------------------------------------------
    def estimate_bpm(self, pulse_sig):
        """
        Standard FFT peak detection in the 0.7–3.0 Hz band.
        Confidence = peak magnitude / mean magnitude in the band.
        High confidence means a clear dominant frequency (strong pulse).
        Low confidence means the spectrum is flat (noisy/unreliable).
        """
        fft_vals = np.abs(np.fft.rfft(pulse_sig))
        freqs = np.fft.rfftfreq(len(pulse_sig), 1.0 / self.fs)

        # Restrict to physiological heart rate range
        mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(mask):
            return self.bpm, 0.0

        masked_fft = fft_vals[mask]
        masked_freqs = freqs[mask]

        peak_idx = np.argmax(masked_fft)
        peak_freq = masked_freqs[peak_idx]
        peak_mag = masked_fft[peak_idx]

        # Confidence: how much the peak stands out from the rest
        mean_mag = np.mean(masked_fft) + 1e-8
        confidence = peak_mag / mean_mag

        return peak_freq * 60.0, confidence

    # -------------------------------------------------------------------------
    # Diagnostic overlay
    # -------------------------------------------------------------------------
    def draw_overlay(self, frame):
        """Draw BPM, confidence bar, buffer progress, and optional waveform."""
        h, w, _ = frame.shape

        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # BPM text (green when locked, orange when acquiring)
        bpm_color = (0, 255, 100) if self.signal_locked else (0, 180, 255)
        cv2.putText(frame, f"{self.bpm:.0f}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, bpm_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (185, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        # Status label
        if len(self.rgb_buffer) < self.buffer_size:
            status = "BUFFERING..."
            status_color = (0, 180, 255)
        elif self.signal_locked:
            status = "LOCKED"
            status_color = (0, 255, 100)
        else:
            status = "ACQUIRING..."
            status_color = (0, 100, 255)

        cv2.putText(frame, status, (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

        # Confidence bar (red → yellow → green)
        conf_normalized = min(self.confidence / 5.0, 1.0)
        bar_w = int(280 * conf_normalized)
        if conf_normalized < 0.33:
            bar_color = (0, 0, 220)       # Red
        elif conf_normalized < 0.66:
            bar_color = (0, 200, 255)     # Yellow
        else:
            bar_color = (0, 220, 100)     # Green

        cv2.rectangle(frame, (25, 125), (25 + bar_w, 140), bar_color, -1)
        cv2.rectangle(frame, (25, 125), (305, 140), (80, 80, 80), 1)
        cv2.putText(frame, f"Signal Quality: {self.confidence:.1f}", (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Buffer progress bar (during warm-up)
        if len(self.rgb_buffer) < self.buffer_size:
            pct = len(self.rgb_buffer) / self.buffer_size
            prog_w = int(280 * pct)
            cv2.rectangle(frame, (25, 165), (25 + prog_w, 175), (200, 150, 50), -1)
            cv2.rectangle(frame, (25, 165), (305, 175), (80, 80, 80), 1)
            cv2.putText(frame, f"Buffer: {int(pct * 100)}%", (25, 192),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Waveform overlay (bottom-left)
        if self.show_waveform and self.last_pulse_sig is not None:
            self.draw_waveform(frame)

    def draw_waveform(self, frame):
        """Render the last 3 seconds of the filtered pulse signal as a mini graph."""
        h, w, _ = frame.shape
        sig = self.last_pulse_sig

        # Take the last 3 seconds
        n_samples = min(len(sig), self.fs * 3)
        sig = sig[-n_samples:]

        box_x, box_y = 10, h - 110
        box_w, box_h = 300, 90

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Normalize signal to fit the box
        sig_min, sig_max = sig.min(), sig.max()
        if sig_max - sig_min < 1e-8:
            return
        sig_norm = (sig - sig_min) / (sig_max - sig_min)

        # Generate polyline points
        points = []
        for i, val in enumerate(sig_norm):
            x = box_x + int(i * box_w / len(sig_norm))
            y = box_y + box_h - int(val * (box_h - 10)) - 5
            points.append((x, y))

        if len(points) > 1:
            color = (0, 255, 100) if self.signal_locked else (0, 180, 255)
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, 1, cv2.LINE_AA)

        cv2.putText(frame, "PULSE", (box_x + 5, box_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(0)

        # ROI landmark indices: forehead, left cheek, right cheek
        fh_ids = [10, 67, 103, 109, 151, 338, 297, 332]
        lc_ids = [234, 93, 132, 58, 172, 136, 150, 149]
        rc_ids = [454, 323, 361, 288, 397, 365, 379, 378]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.frame_count += 1

            # --- Preprocessing: CLAHE lighting normalization ---
            processed = self.preprocess_frame(frame)

            # --- Face detection ---
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            )
            detection_result = self.detector.detect(mp_image)

            if detection_result.face_landmarks:
                lm = detection_result.face_landmarks[0]

                # --- Skin-masked ROI means ---
                m1 = self.get_roi_mean(processed, lm, fh_ids)
                m2 = self.get_roi_mean(processed, lm, lc_ids)
                m3 = self.get_roi_mean(processed, lm, rc_ids)
                avg_rgb = np.mean([m1, m2, m3], axis=0)

                self.rgb_buffer.append(avg_rgb)
                if len(self.rgb_buffer) > self.buffer_size:
                    self.rgb_buffer.pop(0)

                if len(self.rgb_buffer) >= self.buffer_size:
                    rgb_array = np.array(self.rgb_buffer)

                    # --- POS + CHROM fused pulse signal ---
                    pulse_sig = self.fused_pulse(rgb_array)
                    self.last_pulse_sig = pulse_sig

                    # --- FFT-based BPM estimation ---
                    new_bpm, new_conf = self.estimate_bpm(pulse_sig)
                    self.confidence = new_conf

                    # --- Stabilization ---
                    CONF_THRESHOLD = 2.0
                    if new_conf >= CONF_THRESHOLD:
                        # Physiological clamping: reject jumps > 20 BPM
                        if self.bpm > 0 and abs(new_bpm - self.bpm) > 20:
                            pass  # Keep current BPM
                        else:
                            # EMA smoothing
                            EMA_ALPHA = 0.15
                            if self.bpm == 0:
                                self.bpm = new_bpm
                            else:
                                self.bpm = EMA_ALPHA * new_bpm + (1 - EMA_ALPHA) * self.bpm

                        self.signal_locked = True
                    else:
                        # Low confidence — freeze display at last good value
                        self.signal_locked = False

                # --- ROI visualization ---
                for ids in [fh_ids, lc_ids, rc_ids]:
                    pts = np.array(
                        [(int(lm[i].x * frame.shape[1]), int(lm[i].y * frame.shape[0])) for i in ids]
                    )
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 1)

            # --- Overlay ---
            self.draw_overlay(frame)

            cv2.imshow('Robust rPPG', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                self.show_waveform = not self.show_waveform

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RobustRPPG().run()