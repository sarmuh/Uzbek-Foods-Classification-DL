from pathlib import Path

import cv2
import numpy as np


# ==================== SOZLAMALAR ====================

LOYIHA_PAPKA = Path(__file__).resolve().parent

RASM_PAPKA = LOYIHA_PAPKA / "qayerdan olinsin rasmlar"
LABEL_PAPKA = LOYIHA_PAPKA / "qayerga saqlansin"

CLASS_ID = 1  # Bizning sinf tartibimizda: Palov = 2

OYNA_NOMI = "Palov | ENTER: ramkani qabul qilish | ESC: keyingi rasm"


# ==================== PAPKALARNI TEKSHIRISH ====================

if not RASM_PAPKA.is_dir():
    raise FileNotFoundError(
        f"Rasmlar papkasi topilmadi: {RASM_PAPKA}"
    )

LABEL_PAPKA.mkdir(parents=True, exist_ok=True)

formatlar = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rasmlar = sorted(
    fayl
    for fayl in RASM_PAPKA.iterdir()
    if fayl.is_file() and fayl.suffix.lower() in formatlar
)

# Masalan, plov1.jpg va plov1.png bir xil TXT nomini beradi.
nomlar = [fayl.stem.lower() for fayl in rasmlar]

if len(nomlar) != len(set(nomlar)):
    raise ValueError(
        "Rasmlarning asosiy nomlari takrorlangan. "
        "Masalan, plov1.jpg va plov1.png nomlarini farqlang."
    )

print(f"Jami rasmlar: {len(rasmlar)}")
print(f"Annotatsiyalar saqlanadigan joy: {LABEL_PAPKA}")


# ==================== RASMLARNI BELGILASH ====================

try:
    for tartib, rasm_path in enumerate(rasmlar, start=1):

        label_path = LABEL_PAPKA / f"{rasm_path.stem}.txt"

        # Oldin belgilangan rasmlarni o‘tkazamiz.
        if label_path.exists():
            print(f"Oldin saqlangan: {rasm_path.name}")
            continue

        # Unicode/kirill belgili yo‘llarni ham o‘qiymiz.
        try:
            fayl_baytlari = rasm_path.read_bytes()

            if not fayl_baytlari:
                print(f"Bo‘sh fayl: {rasm_path.name}")
                continue

            bufer = np.frombuffer(fayl_baytlari, dtype=np.uint8)
            rasm = cv2.imdecode(bufer, cv2.IMREAD_COLOR)

        except (OSError, cv2.error) as xato:
            print(f"Rasm ochilmadi: {rasm_path.name}")
            print(xato)
            continue

        if rasm is None:
            print(f"Format ochilmadi yoki fayl buzilgan: {rasm_path.name}")
            continue

        # Katta rasmlarni ekranga sig‘diramiz.
        balandlik, kenglik = rasm.shape[:2]

        masshtab = min(
            1000 / kenglik,
            700 / balandlik,
            1.0
        )

        ekran_rasm = cv2.resize(
            rasm,
            (
                max(1, round(kenglik * masshtab)),
                max(1, round(balandlik * masshtab))
            ),
            interpolation=cv2.INTER_AREA
        )

        print(f"\n[{tartib}/{len(rasmlar)}] {rasm_path.name}")
        print("1. Palov atrofiga ramka chizing.")
        print("2. ENTER bosing — ramka qabul qilinadi.")
        print("3. Boshqa Palov porsiyasi bo‘lsa, yana ramka chizing.")
        print("4. Shu rasm tayyor bo‘lsa, ESC bosing.")

        ramkalar = cv2.selectROIs(
            OYNA_NOMI,
            ekran_rasm,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyAllWindows()

        if len(ramkalar) == 0:
            print("Ramka belgilanmadi. TXT yaratilmagan.")
            continue

        ekran_h, ekran_w = ekran_rasm.shape[:2]
        qatorlar = []

        for x, y, w, h in ramkalar:

            if w <= 0 or h <= 0:
                continue

            # Koordinatalarni YOLO formatiga o‘tkazamiz.
            x_center = (x + w / 2) / ekran_w
            y_center = (y + h / 2) / ekran_h
            box_width = w / ekran_w
            box_height = h / ekran_h

            qator = (
                f"{CLASS_ID} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            qatorlar.append(qator)

        if qatorlar:
            label_path.write_text(
                "\n".join(qatorlar) + "\n",
                encoding="utf-8"
            )

            print(
                f"Saqlandi: {label_path.name} "
                f"— {len(qatorlar)} ta ramka"
            )

        # Rasmlar orasida ishni to‘xtatish imkoniyati.
        javob = input(
            "Davom etish: ENTER | To‘xtatish: q + ENTER: "
        ).strip().lower()

        if javob == "q":
            print("To‘xtatildi. Saqlangan annotatsiyalar saqlanib qoladi.")
            break

except KeyboardInterrupt:
    print("\nJarayon to‘xtatildi. Oldingi annotatsiyalar saqlanib qoladi.")

finally:
    cv2.destroyAllWindows()

print("\nBelgilash jarayoni yakunlandi.")