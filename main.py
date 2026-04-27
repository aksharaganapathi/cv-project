import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class RobustRPPG:
    def __init__(self, buffer_size=300, fs=30):
        self.buffer_size = buffer_size
        self.fs = fs

        # --- Per-ROI RGB buffers (forehead, left cheek, right cheek) ---
        # Keeping separate buffers lets us compute per-ROI signal quality and
        # blend the three pulse signals by quality rather than averaging blindly.
        self.roi_buffers = [[], [], []]

        self.bpm = 0.0
        self.confidence = 0.0
        self.signal_locked = False
        self.show_waveform = True
        self.last_pulse_sig = None
        self.frame_count = 0

        # Current quality weight for each ROI (updated every analysis cycle)
        self.roi_weights = [1 / 3, 1 / 3, 1 / 3]
        self.roi_names = ["FH", "LC", "RC"]

        # Rolling history of accepted BPM estimates for robust jump rejection.
        # Using a median of recent values is much more stable than a fixed ±20 BPM
        # threshold because it adapts to the actual signal trend.
        self.bpm_history = deque(maxlen=10)

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
        Returns (mean_bgr, pixel_count) so callers can use pixel_count as a
        proxy for region reliability (larger skin area → more stable mean).
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

        pixel_count = cv2.countNonZero(combined_mask)

        # Fallback: if skin mask is too aggressive, use polygon alone
        if pixel_count < 30:
            combined_mask = poly_mask
            pixel_count = cv2.countNonZero(combined_mask)

        mean_val = cv2.mean(frame, mask=combined_mask)[:3]
        return mean_val, pixel_count

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
    # Per-ROI spectral SNR (for weighting)
    # -------------------------------------------------------------------------
    def roi_spectral_snr(self, roi_pulse):
        """
        Compute a spectral SNR score for a single ROI's bandpassed pulse.

        SNR = peak_magnitude / noise_floor, where the noise floor is the
        mean FFT magnitude in the heart-rate band *excluding* a small window
        of bins around the peak.  This is much more discriminating than the
        old peak/mean metric because the mean was pulled up by the peak
        itself, which compressed the dynamic range of the score.

        Returns 0.0 for degenerate signals.
        """
        fft_vals = np.abs(np.fft.rfft(roi_pulse))
        freqs = np.fft.rfftfreq(len(roi_pulse), 1.0 / self.fs)

        band_mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(band_mask):
            return 0.0

        masked_fft = fft_vals[band_mask]
        peak_idx = int(np.argmax(masked_fft))
        peak_mag = masked_fft[peak_idx]

        # Exclude ±3 bins around the peak for the noise estimate
        noise_mask = np.ones(len(masked_fft), dtype=bool)
        lo = max(0, peak_idx - 3)
        hi = min(len(masked_fft), peak_idx + 4)
        noise_mask[lo:hi] = False

        if noise_mask.any():
            noise_floor = np.mean(masked_fft[noise_mask]) + 1e-8
        else:
            noise_floor = np.mean(masked_fft) + 1e-8

        return float(peak_mag / noise_floor)

    # -------------------------------------------------------------------------
    # Weighted per-ROI pulse fusion
    # -------------------------------------------------------------------------
    def fused_pulse_weighted(self, roi_arrays, roi_pixel_counts):
        """
        Compute a POS+CHROM fused pulse signal for each ROI independently,
        score each by spectral SNR, and return a quality-weighted blend.

        The pixel-count bonus rewards larger, more stable skin regions.
        Weights are stored in self.roi_weights for the overlay display.
        """
        roi_pulses = []
        roi_qualities = []

        for rgb_data, px_count in zip(roi_arrays, roi_pixel_counts):
            p_pos = self.bandpass(self.pos_method(rgb_data))
            p_chrom = self.bandpass(self.chrom_method(rgb_data))
            pulse = (p_pos + p_chrom) / 2.0

            snr = self.roi_spectral_snr(pulse)

            # Small log-scale pixel-count bonus so larger regions are preferred
            # when SNR is equal, but SNR differences dominate.
            px_bonus = np.log1p(px_count) / np.log1p(500)   # normalized 0–1
            quality = snr * (0.8 + 0.2 * px_bonus)

            roi_pulses.append(pulse)
            roi_qualities.append(max(quality, 1e-8))

        # Softmax-style normalization so weights always sum to 1
        total = sum(roi_qualities)
        self.roi_weights = [q / total for q in roi_qualities]

        combined = sum(w * p for w, p in zip(self.roi_weights, roi_pulses))
        return combined

    # -------------------------------------------------------------------------
    # BPM estimation via FFT with improved spectral SNR confidence
    # -------------------------------------------------------------------------
    def estimate_bpm(self, pulse_sig):
        """
        FFT peak detection in 0.7–3.0 Hz.

        Confidence is now a true spectral SNR: peak vs. noise floor
        excluding bins around the peak.  The old peak/mean metric was
        artificially compressed because the dominant peak always inflated
        the mean it was divided by.
        """
        fft_vals = np.abs(np.fft.rfft(pulse_sig))
        freqs = np.fft.rfftfreq(len(pulse_sig), 1.0 / self.fs)

        band_mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(band_mask):
            return self.bpm, 0.0

        masked_fft = fft_vals[band_mask]
        masked_freqs = freqs[band_mask]

        peak_idx = int(np.argmax(masked_fft))
        peak_freq = masked_freqs[peak_idx]
        peak_mag = masked_fft[peak_idx]

        # Exclude ±3 bins around the peak for a cleaner noise-floor estimate
        noise_mask = np.ones(len(masked_fft), dtype=bool)
        lo = max(0, peak_idx - 3)
        hi = min(len(masked_fft), peak_idx + 4)
        noise_mask[lo:hi] = False

        if noise_mask.any():
            noise_floor = np.mean(masked_fft[noise_mask]) + 1e-8
        else:
            noise_floor = np.mean(masked_fft) + 1e-8

        confidence = peak_mag / noise_floor

        return float(peak_freq * 60.0), float(confidence)

    # -------------------------------------------------------------------------
    # Diagnostic overlay
    # -------------------------------------------------------------------------
    def draw_overlay(self, frame):
        """Draw BPM, confidence bar, buffer progress, ROI weights, and waveform."""
        h, w, _ = frame.shape

        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # BPM text (green when locked, orange when acquiring)
        bpm_color = (0, 255, 100) if self.signal_locked else (0, 180, 255)
        cv2.putText(frame, f"{self.bpm:.0f}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, bpm_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (185, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        # Status label
        if len(self.roi_buffers[0]) < self.buffer_size:
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
        # The new SNR metric has a wider useful range, so normalise against 10.
        conf_normalized = min(self.confidence / 10.0, 1.0)
        bar_w = int(280 * conf_normalized)
        if conf_normalized < 0.33:
            bar_color = (0, 0, 220)
        elif conf_normalized < 0.66:
            bar_color = (0, 200, 255)
        else:
            bar_color = (0, 220, 100)

        cv2.rectangle(frame, (25, 125), (25 + bar_w, 140), bar_color, -1)
        cv2.rectangle(frame, (25, 125), (305, 140), (80, 80, 80), 1)
        cv2.putText(frame, f"Signal Quality: {self.confidence:.1f}", (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # --- Per-ROI weight readout ---
        roi_colors = [(255, 200, 80), (80, 200, 255), (200, 80, 255)]  # FH / LC / RC
        label_x = 25
        for i, (name, w_val, color) in enumerate(
            zip(self.roi_names, self.roi_weights, roi_colors)
        ):
            text = f"{name}:{w_val * 100:.0f}%"
            cv2.putText(frame, text, (label_x, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            label_x += 90

        # Buffer progress bar (during warm-up)
        buf_len = len(self.roi_buffers[0])
        if buf_len < self.buffer_size:
            pct = buf_len / self.buffer_size
            prog_w = int(280 * pct)
            cv2.rectangle(frame, (25, 185), (25 + prog_w, 195), (200, 150, 50), -1)
            cv2.rectangle(frame, (25, 185), (305, 195), (80, 80, 80), 1)
            cv2.putText(frame, f"Buffer: {int(pct * 100)}%", (25, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Waveform overlay (bottom-left)
        if self.show_waveform and self.last_pulse_sig is not None:
            self.draw_waveform(frame)

    def draw_waveform(self, frame):
        """Render the last 3 seconds of the filtered pulse signal as a mini graph."""
        h, w, _ = frame.shape
        sig = self.last_pulse_sig

        n_samples = min(len(sig), self.fs * 3)
        sig = sig[-n_samples:]

        box_x, box_y = 10, h - 110
        box_w, box_h = 300, 90

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        sig_min, sig_max = sig.min(), sig.max()
        if sig_max - sig_min < 1e-8:
            return
        sig_norm = (sig - sig_min) / (sig_max - sig_min)

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
        all_ids = [fh_ids, lc_ids, rc_ids]

        CONF_THRESHOLD = 3.0   # Raised slightly because the new SNR metric is wider-range

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

                # --- Per-ROI skin-masked means + pixel counts ---
                roi_means = []
                roi_pixel_counts = []
                for ids in all_ids:
                    mean_val, px_count = self.get_roi_mean(processed, lm, ids)
                    roi_means.append(mean_val)
                    roi_pixel_counts.append(px_count)

                # Append each ROI mean to its own buffer
                for i, mean_val in enumerate(roi_means):
                    self.roi_buffers[i].append(mean_val)
                    if len(self.roi_buffers[i]) > self.buffer_size:
                        self.roi_buffers[i].pop(0)

                if len(self.roi_buffers[0]) >= self.buffer_size:
                    roi_arrays = [np.array(buf) for buf in self.roi_buffers]

                    # --- Quality-weighted per-ROI pulse fusion ---
                    pulse_sig = self.fused_pulse_weighted(roi_arrays, roi_pixel_counts)
                    self.last_pulse_sig = pulse_sig

                    # --- FFT-based BPM estimation with improved SNR confidence ---
                    new_bpm, new_conf = self.estimate_bpm(pulse_sig)
                    self.confidence = new_conf

                    # --- Adaptive stabilization ---
                    if new_conf >= CONF_THRESHOLD:
                        # Adaptive EMA: higher confidence → faster response.
                        # Linearly scales from 0.08 at threshold to 0.25 at SNR=15.
                        ema_alpha = 0.08 + 0.17 * min((new_conf - CONF_THRESHOLD) / 12.0, 1.0)

                        # Robust jump rejection: compare against median of recent
                        # accepted estimates rather than a fixed ±20 BPM window.
                        # This adapts to the real signal trend and handles resting
                        # vs. exercising heart rates automatically.
                        if len(self.bpm_history) >= 3:
                            ref_bpm = float(np.median(list(self.bpm_history)))
                            jump_threshold = 25.0  # BPM
                            if abs(new_bpm - ref_bpm) > jump_threshold:
                                # Likely a transient spike — freeze current value
                                self.signal_locked = True
                                continue
                        elif self.bpm > 0 and abs(new_bpm - self.bpm) > 30:
                            # Pre-history guard: very large initial jump
                            self.signal_locked = True
                            continue

                        # Accept the estimate
                        self.bpm_history.append(new_bpm)
                        if self.bpm == 0:
                            self.bpm = new_bpm
                        else:
                            self.bpm = ema_alpha * new_bpm + (1 - ema_alpha) * self.bpm

                        self.signal_locked = True
                    else:
                        # Low confidence — freeze display at last good value
                        self.signal_locked = False

                # --- ROI visualization ---
                for ids in all_ids:
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
