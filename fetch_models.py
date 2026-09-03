# -*- coding: utf-8 -*-
"""下载 MediaPipe 官方人体分割 + 人脸检测模型到工具目录的 models/ 下。"""
import os, sys, urllib.request

MODELS = {
    "selfie_multiclass_256x256.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "blaze_face_full_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
}

out_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
os.makedirs(out_dir, exist_ok=True)
for name, url in MODELS.items():
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        print("skip (exists):", dst)
        continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(url, dst)
    print("  ->", dst, os.path.getsize(dst), "bytes")

# verify loadable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(out_dir, "selfie_multiclass_256x256.tflite")),
    output_category_mask=True))
print("labels:", seg.get_labels())
det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(out_dir, "blaze_face_short_range.tflite"))))
print("models load OK:", seg, det is not None)
seg.close(); det.close()
