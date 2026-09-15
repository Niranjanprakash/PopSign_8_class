"""
Multi-video sanity check: pick N test videos per class, run prediction, report pass/fail.
"""
from pathlib import Path
from backend.inference.predict_video import predict_single_video

TEST_ROOT = Path("dataset/popsign/test")
CLASSES = ["after", "airplane", "bird", "cloud", "cry", "dog", "drink", "elephant"]
N = 15  # videos per class (dog has only 19 — takes all available up to N)

all_results = []  # (cls, video_name, pred, conf, status)

for cls in CLASSES:
    videos = sorted((TEST_ROOT / cls).glob("*.mp4"))[:N]
    for video in videos:
        try:
            res = predict_single_video(str(video))
            pred = res["prediction"]
            conf = res["confidence"] * 100
            status = "PASS" if pred.lower() == cls.lower() else "FAIL"
        except Exception as e:
            pred, conf, status = None, None, f"ERROR: {e}"
        all_results.append((cls, video.name[:40], pred, conf, status))

print("\n" + "="*85)
print(f"{'Class':<12} {'Video':<42} {'Predicted':<12} {'Conf%':>6}  Status")
print("="*85)

current_cls = None
cls_pass = cls_total = 0
total_pass = total = 0

for cls, vname, pred, conf, status in all_results:
    if cls != current_cls:
        if current_cls is not None:
            print(f"  >> {current_cls}: {cls_pass}/{cls_total} correct")
            print("-"*85)
        current_cls, cls_pass, cls_total = cls, 0, 0
    conf_str = f"{conf:.1f}%" if conf is not None else "  —"
    ok = status == "PASS"
    cls_pass += ok; cls_total += 1
    total_pass += ok; total += 1
    print(f"{cls:<12} {vname:<42} {str(pred):<12} {conf_str:>6}  {status}")

if current_cls:
    print(f"  >> {current_cls}: {cls_pass}/{cls_total} correct")

print("="*85)
print(f"TOTAL: {total_pass}/{total} correct ({100*total_pass/total:.1f}%)\n")
