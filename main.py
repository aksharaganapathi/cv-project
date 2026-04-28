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

        # Per-ROI RGB buffers (forehead, left cheek, right cheek)
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

        # Rolling history of accepted BPM estimates for robust jump rejection
        self.bpm_history = deque(maxlen=10)

        # Motion tracking: store a sparse subset of landmark positions to
        # detect head movement between frames.  High motion → lower confidence.
        self.prev_lm_pts = None
        self.motion_metric = 0.0   # EMA of mean landmark displacement (normalised)

        # CLAHE preprocessor for lighting normalisation
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
        Locally normalises brightness so sudden lighting changes don't
        spike the raw RGB values fed to POS/CHROM.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # -------------------------------------------------------------------------
    # ROI Extraction with skin-pixel masking
    # -------------------------------------------------------------------------
    def get_roi_mean(self, frame, landmarks, indices):
        """
        Extract mean RGB from a landmark polygon, filtered to skin pixels.
        Returns (mean_bgr, pixel_count) so callers can weight by region size.
        """
        h, w, _ = frame.shape
        points = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        )

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [points], 255)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 20, 50), (50, 255, 255))

        combined_mask = cv2.bitwise_and(poly_mask, skin_mask)
        pixel_count = cv2.countNonZero(combined_mask)

        if pixel_count < 30:
            combined_mask = poly_mask
            pixel_count = cv2.countNonZero(combined_mask)

        mean_val = cv2.mean(frame, mask=combined_mask)[:3]
        return mean_val, pixel_count

    # -------------------------------------------------------------------------
    # POS Method (Plane-Orthogonal-to-Skin)
    # -------------------------------------------------------------------------
    def pos_method(self, rgb_data):
        mean_rgb = np.mean(rgb_data, axis=0)
        mean_rgb[mean_rgb == 0] = 1e-8
        norm_rgb = rgb_data / mean_rgb

        s1 = norm_rgb[:, 1] - norm_rgb[:, 2]
        s2 = norm_rgb[:, 1] + norm_rgb[:, 2] - 2 * norm_rgb[:, 0]

        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        return signal.detrend(s1 + alpha * s2)

    # -------------------------------------------------------------------------
    # CHROM Method (De Haan & Jeanne 2013)
    # -------------------------------------------------------------------------
    def chrom_method(self, rgb_data):
        mean_rgb = np.mean(rgb_data, axis=0)
        mean_rgb[mean_rgb == 0] = 1e-8
        norm_rgb = rgb_data / mean_rgb

        xf = 3.0 * norm_rgb[:, 0] - 2.0 * norm_rgb[:, 1]
        yf = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]

        alpha = np.std(xf) / (np.std(yf) + 1e-8)
        return signal.detrend(xf - alpha * yf)

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
    # Per-ROI spectral SNR helper
    # -------------------------------------------------------------------------
    def roi_spectral_snr(self, roi_pulse):
        """
        Spectral SNR = peak magnitude / noise floor, where the noise floor
        is computed from the HR band *excluding* bins around the peak.
        """
        windowed = roi_pulse * np.hanning(len(roi_pulse))
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(roi_pulse), 1.0 / self.fs)

        band_mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(band_mask):
            return 0.0

        mf = fft_vals[band_mask]
        pi = int(np.argmax(mf))
        noise_mask = np.ones(len(mf), dtype=bool)
        noise_mask[max(0, pi - 3):min(len(mf), pi + 4)] = False
        nf = np.mean(mf[noise_mask]) + 1e-8 if noise_mask.any() else np.mean(mf) + 1e-8
        return float(mf[pi] / nf)

    # -------------------------------------------------------------------------
    # Quality-weighted per-ROI pulse fusion
    # -------------------------------------------------------------------------
    def fused_pulse_weighted(self, roi_arrays, roi_pixel_counts):
        """
        Compute POS+CHROM pulse per ROI, score by spectral SNR × pixel count,
        and return a quality-weighted blend.  Weights are stored for the overlay.
        """
        roi_pulses = []
        roi_qualities = []

        for rgb_data, px_count in zip(roi_arrays, roi_pixel_counts):
            p_pos = self.bandpass(self.pos_method(rgb_data))
            p_chrom = self.bandpass(self.chrom_method(rgb_data))
            pulse = (p_pos + p_chrom) / 2.0

            snr = self.roi_spectral_snr(pulse)
            px_bonus = np.log1p(px_count) / np.log1p(500)
            quality = snr * (0.8 + 0.2 * px_bonus)

            roi_pulses.append(pulse)
            roi_qualities.append(max(quality, 1e-8))

        total = sum(roi_qualities)
        self.roi_weights = [q / total for q in roi_qualities]
        return sum(w * p for w, p in zip(self.roi_weights, roi_pulses))

    # -------------------------------------------------------------------------
    # Core FFT BPM estimator — Hann window + parabolic interpolation
    # -------------------------------------------------------------------------
    def estimate_bpm(self, pulse_sig):
        """
        FFT peak detection with two precision upgrades:

        1. Hann windowing before the FFT reduces spectral leakage — without it,
           energy from the dominant frequency bleeds into neighbouring bins,
           raising the apparent noise floor and making weak peaks harder to find.

        2. Parabolic peak interpolation fits a parabola to the peak bin and
           its two neighbours to find the true peak between bins.  At 30 fps
           over 10 seconds (300 samples) the raw FFT bin width is 0.1 Hz = 6 BPM,
           so sub-bin resolution matters quite a lot.

        Returns (bpm, confidence) where confidence is peak/noise-floor SNR.
        """
        windowed = pulse_sig * np.hanning(len(pulse_sig))
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(pulse_sig), 1.0 / self.fs)

        band_mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(band_mask):
            return self.bpm, 0.0

        mf = fft_vals[band_mask]
        mfreqs = freqs[band_mask]

        pi = int(np.argmax(mf))
        peak_mag = mf[pi]

        # --- Parabolic interpolation for sub-bin precision ---
        if 0 < pi < len(mf) - 1:
            a, b_val, c = mf[pi - 1], mf[pi], mf[pi + 1]
            denom = a - 2 * b_val + c
            p = 0.5 * (a - c) / denom if abs(denom) > 1e-10 else 0.0
            p = max(-0.5, min(0.5, p))          # clamp to ±½ bin
            freq_res = mfreqs[1] - mfreqs[0] if len(mfreqs) > 1 else 0.0
            peak_freq = mfreqs[pi] + p * freq_res
        else:
            peak_freq = mfreqs[pi]

        # --- Noise floor excluding ±3 bins around peak ---
        noise_mask = np.ones(len(mf), dtype=bool)
        noise_mask[max(0, pi - 3):min(len(mf), pi + 4)] = False
        nf = np.mean(mf[noise_mask]) + 1e-8 if noise_mask.any() else np.mean(mf) + 1e-8
        confidence = peak_mag / nf

        return float(peak_freq * 60.0), float(confidence)

    # -------------------------------------------------------------------------
    # Multi-window consensus check
    # -------------------------------------------------------------------------
    def multi_window_estimate(self, pulse_sig):
        """
        Run estimate_bpm on three overlapping sub-windows (full buffer, first
        two-thirds, last two-thirds) and compare results.

        • If all sub-windows agree (within 4 BPM), confidence is boosted and
          the BPM is refined to a weighted average — consensus means the signal
          is stable across time, not just a transient burst.
        • If sub-windows disagree, confidence is penalised — the signal is
          inconsistent, probably motion or noise.

        This catches the common failure mode where a single loud noise spike
        at the right frequency happens to dominate the full-window FFT.
        """
        full_bpm, full_conf = self.estimate_bpm(pulse_sig)

        n = len(pulse_sig)
        seg = (2 * n) // 3     # ~6.7 s at 30 fps

        sub_estimates = []
        for start in [0, n - seg]:
            sub = pulse_sig[start:start + seg]
            if len(sub) >= self.fs * 4:
                sub_estimates.append(self.estimate_bpm(sub))

        if not sub_estimates:
            return full_bpm, full_conf

        agreements = sum(1 for b, _ in sub_estimates if abs(b - full_bpm) < 4.0)
        agreement_ratio = agreements / len(sub_estimates)

        # Scale confidence: 0.65 (all disagree) → 1.25 (all agree)
        conf_scale = 0.65 + 0.60 * agreement_ratio
        adjusted_conf = full_conf * conf_scale

        # Refine BPM with weighted average when sub-windows agree
        if agreements == len(sub_estimates):
            all_bpms = [full_bpm] + [b for b, _ in sub_estimates]
            all_confs = [full_conf] + [c for _, c in sub_estimates]
            total_c = sum(all_confs) + 1e-8
            refined_bpm = sum(b * c for b, c in zip(all_bpms, all_confs)) / total_c
        else:
            refined_bpm = full_bpm

        return refined_bpm, adjusted_conf

    # -------------------------------------------------------------------------
    # Harmonic octave correction
    # -------------------------------------------------------------------------
    def correct_harmonic(self, pulse_sig, candidate_bpm):
        """
        Check whether the detector has locked onto the 2nd harmonic (2× the
        actual heart rate) instead of the fundamental.

        This happens when motion or noise suppresses the fundamental peak and
        amplifies its first harmonic.  If the half-frequency has substantial
        FFT power (>55 % of the candidate peak) and is itself in a plausible
        resting heart rate range (42–100 BPM), we prefer it.
        """
        candidate_hz = candidate_bpm / 60.0
        half_hz = candidate_hz / 2.0

        # Only correct if the half-frequency is physiologically plausible
        if half_hz < 0.7 or half_hz * 60 > 100:
            return candidate_bpm

        windowed = pulse_sig * np.hanning(len(pulse_sig))
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(pulse_sig), 1.0 / self.fs)

        def power_near(target_hz, tol_bins=2):
            idx = int(np.argmin(np.abs(freqs - target_hz)))
            lo = max(0, idx - tol_bins)
            hi = min(len(fft_vals), idx + tol_bins + 1)
            return float(np.max(fft_vals[lo:hi]))

        p_candidate = power_near(candidate_hz)
        p_half = power_near(half_hz)

        if p_half > 0.55 * p_candidate:
            return half_hz * 60.0
        return candidate_bpm

    # -------------------------------------------------------------------------
    # Diagnostic overlay
    # -------------------------------------------------------------------------
    def draw_overlay(self, frame):
        h, w, _ = frame.shape

        # Panel height depends on buffer state
        panel_bottom = 215
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, panel_bottom), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # BPM
        bpm_color = (0, 255, 100) if self.signal_locked else (0, 180, 255)
        cv2.putText(frame, f"{self.bpm:.0f}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, bpm_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (185, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        # Status
        if len(self.roi_buffers[0]) < self.buffer_size:
            status, status_color = "BUFFERING...", (0, 180, 255)
        elif self.signal_locked:
            status, status_color = "LOCKED", (0, 255, 100)
        else:
            status, status_color = "ACQUIRING...", (0, 100, 255)
        cv2.putText(frame, status, (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

        # Confidence bar (normalised against 10 for the wider SNR range)
        conf_n = min(self.confidence / 10.0, 1.0)
        bar_w = int(280 * conf_n)
        bar_color = (0, 0, 220) if conf_n < 0.33 else (0, 200, 255) if conf_n < 0.66 else (0, 220, 100)
        cv2.rectangle(frame, (25, 125), (25 + bar_w, 140), bar_color, -1)
        cv2.rectangle(frame, (25, 125), (305, 140), (80, 80, 80), 1)
        cv2.putText(frame, f"Signal: {self.confidence:.1f}  Motion: {self.motion_metric:.3f}",
                    (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

        # Per-ROI weight readout
        roi_colors = [(255, 200, 80), (80, 200, 255), (200, 80, 255)]
        lx = 25
        for name, wt, col in zip(self.roi_names, self.roi_weights, roi_colors):
            cv2.putText(frame, f"{name}:{wt * 100:.0f}%", (lx, 173),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            lx += 90

        # Buffer progress (warm-up only)
        buf_len = len(self.roi_buffers[0])
        if buf_len < self.buffer_size:
            pct = buf_len / self.buffer_size
            prog_w = int(280 * pct)
            cv2.rectangle(frame, (25, 185), (25 + prog_w, 197), (200, 150, 50), -1)
            cv2.rectangle(frame, (25, 185), (305, 197), (80, 80, 80), 1)
            cv2.putText(frame, f"Buffer: {int(pct * 100)}%", (25, 213),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        if self.show_waveform and self.last_pulse_sig is not None:
            self.draw_waveform(frame)

    def draw_waveform(self, frame):
        h, w, _ = frame.shape
        sig = self.last_pulse_sig[-min(len(self.last_pulse_sig), self.fs * 3):]

        box_x, box_y, box_w, box_h = 10, h - 110, 300, 90
        ov = frame.copy()
        cv2.rectangle(ov, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        sig_min, sig_max = sig.min(), sig.max()
        if sig_max - sig_min < 1e-8:
            return
        sig_norm = (sig - sig_min) / (sig_max - sig_min)

        pts = [(box_x + int(i * box_w / len(sig_norm)),
                box_y + box_h - int(v * (box_h - 10)) - 5)
               for i, v in enumerate(sig_norm)]
        color = (0, 255, 100) if self.signal_locked else (0, 180, 255)
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

        cv2.putText(frame, "PULSE", (box_x + 5, box_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(0)

        fh_ids = [10, 67, 103, 109, 151, 338, 297, 332]
        lc_ids = [234, 93, 132, 58, 172, 136, 150, 149]
        rc_ids = [454, 323, 361, 288, 397, 365, 379, 378]
        all_ids = [fh_ids, lc_ids, rc_ids]

        # Sparse landmark indices for motion estimation (every 20th point)
        motion_ids = list(range(0, 468, 20))

        # Confidence threshold.  The improved spectral SNR metric typically
        # yields 3–8 for clean webcam footage and 10–30 for clean synthetic
        # data, so 2.5 is a reliable but not overly strict lock-on point.
        CONF_THRESHOLD = 2.5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.frame_count += 1

            processed = self.preprocess_frame(frame)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            )
            detection_result = self.detector.detect(mp_image)

            if detection_result.face_landmarks:
                lm = detection_result.face_landmarks[0]

                # --- Motion metric (landmark displacement EMA) ---
                # Using normalised [0,1] coordinates so the metric is
                # resolution-independent.  Typical still-face displacement
                # is <0.001; talking/nodding is 0.003–0.01; large turns >0.01.
                curr_pts = np.array([(lm[i].x, lm[i].y) for i in motion_ids])
                if self.prev_lm_pts is not None:
                    disp = np.mean(np.linalg.norm(curr_pts - self.prev_lm_pts, axis=1))
                    self.motion_metric = 0.8 * self.motion_metric + 0.2 * disp
                self.prev_lm_pts = curr_pts

                # --- Per-ROI extraction ---
                roi_means, roi_pixel_counts = [], []
                for ids in all_ids:
                    mean_val, px_count = self.get_roi_mean(processed, lm, ids)
                    roi_means.append(mean_val)
                    roi_pixel_counts.append(px_count)

                for i, mean_val in enumerate(roi_means):
                    self.roi_buffers[i].append(mean_val)
                    if len(self.roi_buffers[i]) > self.buffer_size:
                        self.roi_buffers[i].pop(0)

                if len(self.roi_buffers[0]) >= self.buffer_size:
                    roi_arrays = [np.array(buf) for buf in self.roi_buffers]

                    # Quality-weighted per-ROI pulse fusion
                    pulse_sig = self.fused_pulse_weighted(roi_arrays, roi_pixel_counts)
                    self.last_pulse_sig = pulse_sig

                    # Multi-window consensus BPM + confidence
                    new_bpm, new_conf = self.multi_window_estimate(pulse_sig)

                    # Harmonic octave correction (must happen before acceptance)
                    new_bpm = self.correct_harmonic(pulse_sig, new_bpm)

                    # Motion-aware confidence gating.
                    # Exponentially penalise confidence during head movement.
                    # Still allows a lock if SNR is very strong despite slight motion.
                    motion_penalty = float(np.exp(-self.motion_metric * 80))
                    effective_conf = new_conf * (0.5 + 0.5 * motion_penalty)
                    self.confidence = effective_conf

                    # --- Stabilisation (no continue — draw_overlay always runs) ---
                    should_accept = False
                    if effective_conf >= CONF_THRESHOLD:
                        # Robust jump rejection via history median
                        if len(self.bpm_history) >= 3:
                            ref_bpm = float(np.median(list(self.bpm_history)))
                            if abs(new_bpm - ref_bpm) <= 25.0:
                                should_accept = True
                        elif self.bpm == 0 or abs(new_bpm - self.bpm) <= 30.0:
                            should_accept = True

                    if should_accept:
                        # Adaptive EMA: scales from α=0.08 at threshold → 0.25 at SNR=15
                        ema_alpha = 0.08 + 0.17 * min(
                            (effective_conf - CONF_THRESHOLD) / 12.0, 1.0
                        )
                        self.bpm_history.append(new_bpm)
                        if self.bpm == 0:
                            self.bpm = new_bpm
                        else:
                            self.bpm = ema_alpha * new_bpm + (1 - ema_alpha) * self.bpm
                        self.signal_locked = True
                    else:
                        self.signal_locked = False

                # ROI polygon visualisation
                for ids in all_ids:
                    pts = np.array(
                        [(int(lm[i].x * frame.shape[1]), int(lm[i].y * frame.shape[0]))
                         for i in ids]
                    )
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 1)

            # Overlay always drawn — even when a jump is rejected
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
