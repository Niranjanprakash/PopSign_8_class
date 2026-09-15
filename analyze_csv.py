from collections import defaultdict
import os
from pathlib import Path

# Load predictions
with open('outputs/predictions/test_predictions.csv','r',encoding='utf-8') as f:
    f.readline()
    rows = []
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split(',')
        if len(parts) >= 5:
            rows.append((parts[2], parts[4], parts[2]==parts[4]))

class_stats = defaultdict(lambda: [0,0])
for true, pred, correct in rows:
    class_stats[true][1] += 1
    if correct: class_stats[true][0] += 1

# Check what's available in popsign dataset (full 250 class version if exists)
# The test_predictions.csv has 100 classes - these are what was tested
# Find classes with >= 40% accuracy and >= 10 test samples
good_classes = []
for cls, (correct, total) in class_stats.items():
    acc = correct/total if total > 0 else 0
    if acc >= 0.35 and total >= 10:
        good_classes.append((cls, acc, correct, total))

good_classes.sort(key=lambda x: x[1], reverse=True)
print(f'Classes with >=35% accuracy and >=10 test samples: {len(good_classes)}')
print()
for cls, acc, c, t in good_classes:
    print(f'  {cls}: {c}/{t} = {acc*100:.1f}%')

print()
print('--- Classes with 20-35% accuracy (medium) ---')
medium = [(cls, c/t, c, t) for cls,(c,t) in class_stats.items() if t>=10 and 0.20<=c/t<0.35]
medium.sort(key=lambda x: x[1], reverse=True)
for cls, acc, c, t in medium:
    print(f'  {cls}: {c}/{t} = {acc*100:.1f}%')
