"""Preprocessing untuk eksperimen CNN CSI. Dataset tidak disertakan."""
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_lab_windows(dataset_root, mode="combined", window_size=100):
    if mode not in {"combined", "butterworth"}:
        raise ValueError("mode harus combined atau butterworth")
    lab = Path(dataset_root) / "Lab Dataset"
    imag_dir = lab / "imaginary_part"
    if not lab.is_dir():
        raise FileNotFoundError(f"Folder tidak ditemukan: {lab}")
    real_files = sorted(p for p in lab.rglob("coordinate*.mat")
                        if re.fullmatch(r"coordinate\d+", p.stem))
    if len(real_files) != 317:
        raise ValueError(f"Diharapkan 317 lokasi Lab Dataset; ditemukan {len(real_files)}")
    sos = butter(4, 0.2, btype="lowpass", output="sos")
    windows, labels, groups = [], [], []
    for real_file in real_files:
        location_id = real_file.stem[len("coordinate"):]
        imag_file = imag_dir / f"imaginary{location_id}.mat"
        if not imag_file.exists():
            raise FileNotFoundError(imag_file)
        real = np.asarray(loadmat(real_file)["myData"], dtype=np.float64)
        imag = np.asarray(loadmat(imag_file)["myData"], dtype=np.float64)
        if real.shape != (3, 30, 1500) or imag.shape != real.shape:
            raise ValueError(f"CSI shape tidak sesuai: {location_id}")
        real = np.nan_to_num(real, nan=0., posinf=0., neginf=0.)
        imag = np.nan_to_num(imag, nan=0., posinf=0., neginf=0.)
        amp = sosfiltfilt(sos, np.hypot(real, imag), axis=-1)
        features = amp.reshape(90, 1500).T
        if mode == "combined":
            phase = np.unwrap(np.angle(real + 1j * imag), axis=1)
            idx = np.arange(30, dtype=np.float64)
            centered = idx - idx.mean()
            slope = (phase * centered[None, :, None]).sum(axis=1, keepdims=True) / (centered ** 2).sum()
            phase = phase - slope * centered[None, :, None] - phase.mean(axis=1, keepdims=True)
            features = np.concatenate([features, phase.reshape(90, 1500).T], axis=1)
        for group, start in enumerate(range(0, 1500, window_size)):
            windows.append(features[start:start + window_size].astype(np.float32))
            labels.append(location_id)
            groups.append(group)
    return np.stack(windows), np.asarray(labels), np.asarray(groups)

def split_scale(X, labels, groups):
    masks = (groups < 10, (groups >= 10) & (groups < 12), groups >= 12)
    encoder = LabelEncoder().fit(labels[masks[0]])
    scaler = StandardScaler()
    results = []
    for index, mask in enumerate(masks):
        x = X[mask]
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        scaled = scaler.fit_transform(flat) if index == 0 else scaler.transform(flat)
        results.append((scaled.reshape(x.shape).astype(np.float32),
                        encoder.transform(labels[mask])))
    return results, encoder, scaler
