from __future__ import annotations
import cv2, numpy as np

FEATURE_NAMES = [
    "mean_h","mean_s","mean_v","mean_l","mean_a","mean_b",
    "motion","edge_density","local_contrast","turbidity_proxy",
    "solid_proxy","phase_boundary_proxy","brightness_std"
]

def extract_frame_features(frame: np.ndarray, prev_gray: np.ndarray | None = None):
    small=cv2.resize(frame,(320,240),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV).astype(np.float32)
    lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_h,mean_s,mean_v=hsv.reshape(-1,3).mean(0)
    mean_l,mean_a,mean_b=lab.reshape(-1,3).mean(0)
    if prev_gray is None: motion=0.0
    else: motion=float(cv2.absdiff(gray,prev_gray).mean()/255.0)
    edges=cv2.Canny(gray,50,150)
    edge_density=float((edges>0).mean())
    blur=cv2.GaussianBlur(gray,(0,0),5)
    local_contrast=float(cv2.absdiff(gray,blur).mean()/255.0)
    brightness_std=float(gray.std()/255.0)
    # Proxies only; deliberately named as proxies, not chemical truths.
    turbidity_proxy=float(np.clip(0.65*local_contrast + 0.35*edge_density,0,1))
    solid_proxy=float(np.clip(0.55*edge_density + 0.25*local_contrast + 0.20*max(motion-0.01,0),0,1))
    sobel_y=np.abs(cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,0,1,ksize=3))
    row_energy=sobel_y.mean(axis=1)
    phase_boundary_proxy=float(np.clip(np.percentile(row_energy,95)/128.0,0,1))
    vec=np.array([mean_h/180,mean_s/255,mean_v/255,mean_l/255,(mean_a-128)/128,(mean_b-128)/128,
                  motion,edge_density,local_contrast,turbidity_proxy,solid_proxy,phase_boundary_proxy,brightness_std],dtype=np.float32)
    return vec, gray
