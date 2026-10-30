import cv2
from pathlib import Path

# 1. Rasmni ochamiz
rasm_path = Path("palov.jpg")
rasm = cv2.imread(str(rasm_path))

if rasm is None:
    raise FileNotFoundError(f"Rasm ochilmadi: {rasm_path}")

# 2. Sichqoncha bilan taom atrofiga ramka chizamiz
x, y, w, h = cv2.selectROI(
    "Taomni belgilang — ENTER bilan tasdiqlang",
    rasm,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise ValueError("Ramka belgilanmadi.")

# 3. Piksel koordinatalarini YOLO formatiga o‘tkazamiz
rasm_h, rasm_w = rasm.shape[:2]

x_center = (x + w / 2) / rasm_w
y_center = (y + h / 2) / rasm_h
box_width = w / rasm_w
box_height = h / rasm_h

class_id = 2  # Bizdagi sinf tartibida 2 = Palov

# 4. Annotatsiyani rasm bilan bir xil nomdagi TXT faylga yozamiz
label_path = rasm_path.with_suffix(".txt")

label_path.write_text(
    f"{class_id} {x_center:.6f} {y_center:.6f} "
    f"{box_width:.6f} {box_height:.6f}\n",
    encoding="utf-8"
)

# 5. Belgilangan ramkani rasmga chizib, tekshirish uchun saqlaymiz
natija = rasm.copy()

cv2.rectangle(
    natija,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

cv2.putText(
    natija,
    "Palov",
    (x, max(y - 10, 25)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

cv2.imwrite("palov_belgilangan.jpg", natija)

print("Annotatsiya:", label_path)
print("Tekshirish rasmi: palov_belgilangan.jpg")