# 🍽️ O‘zbek taomlarini CNN yordamida tasniflash

### ResNet50 · Transfer Learning · PyTorch · Google Colab

O‘zbek milliy taomlarini rasmlar asosida **7 ta sinfga tasniflovchi** deep learning loyihasi. ImageNet’da oldindan o‘qitilgan **ResNet50** modelining asosiy qismi muzlatilib, oxirgi klassifikatsiya qatlami o‘zbek taomlari datasetida o‘qitildi.

Loyiha notebook’i: **`FineTuning_ResNET50.ipynb`**

> 🎯 **Test natijasi:** 1058 ta rasmdan 987 tasi to‘g‘ri tasniflandi — **93.29% accuracy**.

---

## 📌 Loyiha haqida

Loyihaning maqsadi — transfer learning yordamida cheklangan hisoblash resurslarida taom rasmlarini tasniflovchi model yaratish va model tayyorlash jarayonini amaliy o‘rganish.

Notebook quyidagi bosqichlarni qamrab oladi:

- pretrained ResNet50 modelini yuklash;
- backbone parametrlarini muzlatish;
- 7 ta sinf uchun yangi classifier yaratish;
- datasetni yuklash va rasmlarni tayyorlash;
- train, validation va test to‘plamlariga ajratish;
- modelni o‘qitish;
- scheduler va early stopping orqali jarayonni boshqarish;
- loss grafiklarini chizish;
- test metrikalari va confusion matrix orqali baholash;
- model vaznlari va kerakli metama’lumotlarni saqlash.

**Model butun rasmni tasniflaydi.** U taom atrofiga ramka chizmaydi va rasmdagi bir nechta taomning joylashuvini alohida aniqlamaydi.

---

## 🛠️ Texnologiyalar

| Vosita       | Loyihadagi vazifasi                              |
| ------------ | ------------------------------------------------ |
| Python       | Dasturlash tili                                  |
| PyTorch      | Modelni o‘qitish va baholash                     |
| TorchVision  | ResNet50, transforms va ImageFolder              |
| scikit-learn | Datasetni ajratish va classification metrikalari |
| NumPy        | Yordamchi sonli amallar                          |
| Matplotlib   | Rasmlar va loss grafiklari                       |
| Seaborn      | Confusion matrix vizualizatsiyasi                |
| Pillow       | Rasm bilan ishlash                               |
| gdown        | Google Drive’dan dataset yuklash                 |
| Google Colab | Notebook’ni GPU’da bajarish                      |

**Notebook’da qayd etilgan GPU:** NVIDIA Tesla T4.

OpenCV ham import qilingan, ammo notebook’da jonli kamera yoki detection funksiyasi amalga oshirilmagan.

---

## 📂 Dataset

Dataset rasmlari quyidagi usullar orqali yig‘ilgan:

- veb-saytlardan web scraping;
- internetdan qo‘lda yuklab olish;
- ochiq e’lon qilingan datasetlardan foydalanish.

Rasmlar taom nomiga mos papkalarda saqlangan.

### Sinflar va rasmlar soni

| ID | Taom     | Rasmlar soni |
| -: | -------- | -----------: |
|  0 | Lagmon   |         1116 |
|  1 | Manti    |          711 |
|  2 | Palov    |          996 |
|  3 | Perashka |         1031 |
|  4 | Shashlik |         1691 |
|  5 | Shorva   |          520 |
|  6 | Somsa    |          983 |
|    | **Jami** |     **7048** |

> ℹ️ Notebook’dagi dastlabki tavsifda 7085 ta rasm yozilgan. Ushbu README’da `ImageFolder` orqali amalda yuklangan va chiqarilgan **7048 ta rasm** asos qilib olindi.

### Datasetni yuklab olish

📥 [Dataset — Google Drive](https://drive.google.com/file/d/14CkkOGjQV17wh7x0pmc5woJU9IzRdrr9/view?usp=sharing)

Notebook datasetni `gdown` orqali `uzbek_foods.zip` nomi bilan yuklaydi va `Data` papkasiga ochadi.

### Papkalar tuzilishi

```text
Data/
└── Uzbek-Foods/
    ├── Lagmon/
    ├── Manti/
    ├── Palov/
    ├── Perashka/
    ├── Shashlik/
    ├── Shorva/
    └── Somsa/
```

Colab’dagi dataset yo‘li:

```python
"/content/Data/Uzbek-Foods"
```

Dataset quyidagicha yaratiladi:

```python
dataset = ImageFolder(
    root="/content/Data/Uzbek-Foods",
    transform=preprocess
)
```

`ImageFolder` sinf nomlarini papkalardan oladi. Rasm so‘ralganda fayl ochilib, preprocessing qo‘llanadi.

### Datasetni ajratish

Rasmlar sinf nisbatlarini saqlagan holda uch qismga ajratilgan:

| To‘plam    | Ulush | Rasmlar soni | Vazifasi                               |
| ---------- | ----: | -----------: | -------------------------------------- |
| Train      |   70% |         4933 | Modelni o‘qitish                       |
| Validation |   15% |         1057 | O‘qitishni kuzatish va modelni tanlash |
| Test       |   15% |         1058 | Yakuniy baholash                       |

Ajratishda `train_test_split`, `stratify`, `shuffle=True` va **`RANDOM_SEED = 69`** ishlatilgan.

`Subset` orqali tegishli indekslarga asoslangan datasetlar yaratilgan.

---

## 🧠 Model arxitekturasi

### Pretrained ResNet50

```python
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).to(device)
```

ImageNet’da o‘rganilgan belgilarni ajratish qobiliyatidan foydalanish uchun model pretrained vaznlar bilan yuklangan.

### Backbone’ni muzlatish

```python
for param in model.parameters():
    param.requires_grad = False
```

Bu parametrlar uchun gradient hisoblanmaydi va ularning qiymatlari optimizer tomonidan yangilanmaydi.

### Yangi klassifikatsiya qatlami

Original modeldagi 1000 ta chiqishli qatlam 7 ta chiqishli qatlamga almashtirilgan:

```python
model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=7
).to(device)
```

Yangi qatlam:

```text
Linear(in_features=2048, out_features=7, bias=True)
```

- **2048** — ResNet ajratib bergan belgilar soni.
- **7** — taom sinflari soni.

Yangi qatlamning vazn va biaslari o‘qitiladi:

```text
2048 × 7 + 7 = 14 343 parametr
```

> 🔎 Notebook nomida “FineTuning” yozilgan bo‘lsa-da, amaldagi usul — **muzlatilgan backbone bilan transfer learning**, ya’ni feature extraction va classifier’ni o‘qitish. Backbone qatlamlarini ochib, to‘liq fine-tuning qilish bajarilmagan.

---

## 🖼️ Rasmlarni tayyorlash

Preprocessing `torchvision.transforms.v2` yordamida yozilgan:

```python
preprocess = v2.Compose([
    v2.Resize(
        size=232,
        interpolation=v2.InterpolationMode.BILINEAR,
        antialias=True
    ),
    v2.CenterCrop(size=224),
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

| Bosqich    | Bajariladigan amal                                       |
| ---------- | -------------------------------------------------------- |
| Resize     | Qisqa tomonini 232 pikselga keltiradi                    |
| CenterCrop | Markazdan 224 × 224 qismni oladi                         |
| ToImage    | Rasmni tensor ko‘rinishiga o‘tkazadi                     |
| ToDtype    | `float32` formatiga o‘tkazib, qiymatlarni masshtablaydi  |
| Normalize  | ImageNet mean/std qiymatlari bilan normalizatsiya qiladi |

Train, validation va test uchun **bir xil preprocessing** ishlatilgan. Alohida tasodifiy train augmentation qo‘shilmagan.

Notebook’da namunaviy rasmni ko‘rsatish uchun normalizatsiyani teskari aylantiruvchi `denormalize()` funksiyasi ham mavjud.

---

## ⚙️ DataLoader va o‘qitish sozlamalari

### DataLoader

| Sozlama            | Qiymat           |
| ------------------ | ---------------- |
| Batch size         | 64               |
| Train shuffle      | `True`           |
| Validation shuffle | `False`          |
| Test shuffle       | `False`          |
| Workerlar soni     | `os.cpu_count()` |

Bir batch uchun kuzatilgan shakllar:

```text
Rasmlar: [64, 3, 224, 224]
Sinflar: [64]
```

### Asosiy hyperparameterlar

| Parametr                   | Qiymat           |
| -------------------------- | ---------------- |
| Loss                       | CrossEntropyLoss |
| Optimizer                  | Adam             |
| O‘qitiladigan qism         | `model.fc`       |
| Boshlang‘ich learning rate | 0.001            |
| Maksimal epoch             | 100              |
| Random seed                | 69               |
| Early stopping patience    | 5                |

`CrossEntropyLoss`ga modelning xom ballari — logits beriladi. O‘qitishdan oldin alohida Softmax qo‘llanmagan.

### Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
```

Scheduler har epochdan keyin validation loss orqali boshqariladi:

```python
scheduler.step(val_loss)
```

U validation loss yetarlicha yaxshilanmay qolsa, learning rate’ni kamaytiradi. Saqlangan run davomida LR **0.001** bo‘lib qolgan.

---

## 🔄 O‘qitish jarayoni

Har epochda quyidagi ishlar bajariladi:

1. Train batchlari modelga beriladi.
2. Loss hisoblanadi.
3. Gradientlar hisoblanadi.
4. Classifier parametrlari yangilanadi.
5. Validation natijalari hisoblanadi.
6. Loss qiymatlari grafik uchun ro‘yxatlarga qo‘shiladi.
7. Modelni saqlash sharti tekshiriladi.
8. Scheduler va early stopping ishlaydi.

Train paytida:

```python
model.eval()
model.fc.train()
```

Bu usul backbone’dagi BatchNorm statistikalarini o‘zgartirmaydi. `eval()` gradient hisoblashni o‘chirmaydi; yangi `fc` qatlami o‘rganishda davom etadi.

Validation va testda:

```python
with torch.no_grad():
```

ishlatilib, parametrlar yangilanmaydi.

### Early stopping va checkpoint mezoni

Notebook’da model quyidagi **ikkala shart birga bajarilganda** saqlanadi:

```python
val_loss < best_val_score and \
val_accuracy >= best_val_acc + 0.0005
```

Ya’ni oxirgi qabul qilingan modelga nisbatan:

- validation loss kichikroq bo‘lishi;
- validation accuracy kamida **0.05 foiz punkt** yuqoriroq bo‘lishi kerak.

Ketma-ket 5 epochda bu birgalikdagi shart bajarilmasa, o‘qitish to‘xtaydi.

> 📌 Bu qoida faqat eng kichik loss yoki faqat eng yuqori accuracy’ni tanlashdan farq qiladi. Koddagi `best_val_score` va `best_val_acc` oxirgi qabul qilingan checkpoint qiymatlaridir.

---

## 📈 O‘qitish natijalari

Notebook’da saqlangan run **14-epochda to‘xtagan**. Shartga mos eng yaxshi checkpoint — **9-epoch**.

| Epoch | Train loss | Train accuracy |   Val loss | Val accuracy |
| ----: | ---------: | -------------: | ---------: | -----------: |
|     1 |     1.0582 |         75.53% |     0.5847 |       91.67% |
|     2 |     0.4681 |         92.28% |     0.3723 |       93.28% |
|     4 |     0.2690 |         94.59% |     0.2575 |       94.80% |
|     7 |     0.1861 |         95.62% |     0.2066 |       95.08% |
|     8 |     0.1720 |         95.84% |     0.1977 |       95.18% |
| **9** | **0.1588** |     **96.19%** | **0.1903** |   **95.46%** |
|    10 |     0.1481 |         96.45% |     0.1848 |       95.18% |
|    14 |     0.1187 |         97.08% |     0.1711 |       95.46% |

14-epochda loss 9-epochdagidan kichik bo‘lgan, ammo accuracy belgilangan miqdorda oshmagan. Shu sababli checkpoint yangilanmagan.

O‘qitish tugagach, saqlangan parametrlar modelga qaytarilgan:

```python
model.load_state_dict(best_model_state)
model.eval()
```

### Loss grafigi

`train_losses` va `val_losses` ro‘yxatlaridan barcha bajarilgan epochlar uchun grafik chizilgan.

Grafik **o‘qitish jarayonini** aks ettiradi. Modelga qaytarilgan checkpoint esa 9-epochga tegishli.

---

## 🧪 Test natijalari

Tanlangan model alohida **1058 ta test rasmi** orqali baholangan.

| Ko‘rsatkich        |     Natija |
| ------------------ | ---------: |
| Test rasmlari      |       1058 |
| To‘g‘ri javoblar   |        987 |
| Noto‘g‘ri javoblar |         71 |
| Test loss          |     0.2254 |
| **Test accuracy**  | **93.29%** |

Test bosqichida gradientlar hisoblanmagan va model parametrlari yangilanmagan.

### Sinflar bo‘yicha classification report

Quyidagi qiymatlar notebook chiqishidagi ikki xonagacha yaxlitlangan natijalardir:

| Taom             | Precision |   Recall | F1-score | Test rasmlari |
| ---------------- | --------: | -------: | -------: | ------------: |
| Lagmon           |      0.94 |     0.90 |     0.92 |           168 |
| Manti            |      0.95 |     0.97 |     0.96 |           106 |
| Palov            |      0.85 |     0.93 |     0.89 |           150 |
| Perashka         |      0.99 |     0.95 |     0.97 |           155 |
| Shashlik         |      0.94 |     0.91 |     0.92 |           254 |
| Shorva           |      0.92 |     0.87 |     0.89 |            78 |
| Somsa            |      0.95 |     0.98 |     0.96 |           147 |
| **Macro avg**    |  **0.93** | **0.93** | **0.93** |      **1058** |
| **Weighted avg** |  **0.93** | **0.93** | **0.93** |      **1058** |

- **Precision:** model biror taom deb bergan javoblarning qanchasi to‘g‘ri?
- **Recall:** shu taomning haqiqiy rasmlaridan qanchasi topildi?
- **F1-score:** precision va recall’ni birgalikda hisobga oladi.

### Natijalarning qisqa tahlili

- **Perashka** eng yuqori qayd etilgan F1-score’ga ega: **0.97**.
- **Somsa** recall’i **0.98** — testdagi Somsa rasmlarining katta qismi to‘g‘ri topilgan.
- **Palov** precision’i **0.85** — ayrim boshqa taomlar Palov deb tasniflangan.
- **Shorva** recall’i **0.87** — ushbu sinf bo‘yicha yaxshilash imkoniyati bor.

### Confusion matrix

Seaborn orqali sinf nomlari yozilgan, ko‘k rangdagi confusion matrix chizilgan.

- Qatorlar — haqiqiy sinflar.
- Ustunlar — model taxminlari.
- Diagonal — to‘g‘ri javoblar.
- Diagonaldan tashqaridagi kataklar — adashishlar.

> ℹ️ Test kodida haqiqiy javoblar ro‘yxati `y_train` deb nomlangan. Nomiga qaramay, u **test javoblarini** saqlaydi, chunki ro‘yxat `test_loader` ichida to‘ldirilgan. Aniqlik uchun keyinchalik `y_true` deb nomlash mumkin.

---

## 💾 Modelni saqlash

Model quyidagi faylga saqlangan:

```text
uzbek_food_resnet50.pth
```

Checkpoint tarkibi:

| Kalit              | Mazmuni                                             |
| ------------------ | --------------------------------------------------- |
| `model_state_dict` | Vaznlar va saqlangan statistikalar                  |
| `classes`          | Sinf nomlari                                        |
| `class_to_idx`     | Sinf nomi va ID mosligi                             |
| `architecture`     | `resnet50`                                          |
| `num_classes`      | `7`                                                 |
| `preprocess`       | Resize, crop, mean/std va interpolation sozlamalari |

Tensorlar saqlashdan oldin CPU’ga o‘tkazilgan. Model fayli Colab’dan yuklab olish uchun chiqarilgan.

📥 [Tayyor model — Google Drive](https://drive.google.com/file/d/1rUDS-MjLw0OsYWq9iFTVBdxgsiXIMpkC/view?usp=sharing)

Bu checkpoint inference uchun kerakli asosiy ma’lumotlarni saqlaydi. Unda optimizer va scheduler holatlari yo‘q; o‘qitishni aynan o‘sha holatdan davom ettirish uchun ular ham saqlanishi kerak.

---

## 🚀 Notebook’ni ishga tushirish

1. `FineTuning_ResNET50.ipynb` faylini Google Colab’da oching.
2. Runtime sozlamalaridan GPU muhitini tanlang.
3. Kerakli kutubxonalar mavjudligini tekshiring.
4. Datasetni yuklab, arxivni oching.
5. Dataset yo‘li `/content/Data/Uzbek-Foods` ekanini tekshiring.
6. Kataklarni yuqoridan pastga tartib bilan bajaring.
7. O‘qitish, loss grafigi va test natijalarini ko‘ring.
8. Model checkpoint’ini yuklab oling.

### Ishga tushirishga oid eslatmalar

- `gdown` o‘rnatilmagan bo‘lsa, uning o‘rnatish katagini birinchi importlardan oldin bajaring.
- Notebook’dagi GPU nomini chiqarish qatori CUDA mavjudligini talab qiladi. CPU’da ishga tushirish uchun shu qatorni shart bilan himoyalash kerak.
- `os.cpu_count()` bergan workerlar soni muhitga bog‘liq. Resurs cheklansa, workerlar sonini kamaytirish mumkin.
- Kutubxonalarning aniq versiyalari notebook’da qayd etilmagan.

---

## ⚠️ Cheklovlar

- Model faqat mavjud **7 ta sinf** orasidan javob tanlaydi.
- Taom bo‘lmagan rasm uchun ham shu sinflardan birini chiqarishi mumkin.
- Bitta rasm uchun bitta sinf beradi; bounding box va segmentation yo‘q.
- Train va testdagi o‘xshash yoki takror rasmlar umumlashtirish bahosiga ta’sir qilishi mumkin.
- Sinflar hajmi teng emas; umumiy accuracy bilan birga har sinf natijalarini ham kuzatish kerak.
- Center crop chetda joylashgan taomning bir qismini kesib yuborishi mumkin.
- Internet rasmlaridagi natija haqiqiy kamera tasvirlaridagi natijani kafolatlamaydi.
- `torch.manual_seed(69)` va split seed’i ishlatilgan, ammo barcha tasodifiy manbalar va deterministik sozlamalar to‘liq boshqarilmagan.

---

## 🔜 CNN loyihasini yaxshilash yo‘nalishlari

- [ ] Xato tasniflangan 71 ta test rasmini tahlil qilish.
- [ ] Train uchun mos data augmentation sinash.
- [ ] Palov va Shorva sinflaridagi xatolar sababini tekshirish.
- [ ] Checkpoint tanlash va early stopping mezonlarini tajribalar bo‘yicha solishtirish.
- [ ] Backbone’ning oxirgi bloklarini kichik LR bilan fine-tuning qilish.
- [ ] Bitta yangi rasm uchun inference funksiyasi yozish.
- [ ] Saqlangan modelni qayta yuklab, natijalarini tekshirish.
- [ ] Kutubxona versiyalari va to‘liq tajriba sozlamalarini qayd etish.

Model va sozlamalarni tanlashda validation to‘plamidan foydalaniladi. Test to‘plami yakuniy baholash uchun saqlanadi.

---

## 👨‍💻 Muallif

**Safarov Muhammad**

Sun’iy intellekt bo‘yicha amaliy o‘quv loyihasi.

**Notebook’da qayd etilgan yakuniy natija: 7 sinfli o‘zbek taomlari classification — 93.29% test accuracy.**
