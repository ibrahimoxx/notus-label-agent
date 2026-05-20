# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

<!-- project-spec: notus-label-agent -->

> **Projet** : Notus Label Agent
> **Description** : Agent IA de vision par ordinateur qui extrait automatiquement le **numéro de LOT** et la **date d'expiration** depuis une photo d'étiquette pharmaceutique, à partir d'un nom de produit déjà connu dans le catalogue Notus Pharma.
> **Stack** : Python 3.11 · FastAPI · OpenCV · Tesseract + PaddleOCR · PostgreSQL · Redis · Docker
> **Auteur** : Ibrahim Stouri
> **Contexte** : Test technique de recrutement pour Notus Pharma (SaaS officine, Maroc)

---

## 0. OVERVIEW

`notus-label-agent` est un service backend qui automatise la saisie des **lots** et **dates de péremption** dans un SaaS de gestion d'officine. L'agent en pharmacie tape le nom du médicament (catalogue déjà existant), prend une photo de l'étiquette, et le pipeline IA en extrait LOT + expiration via un double moteur OCR (Tesseract + PaddleOCR), un préprocessing OpenCV avancé (CLAHE, adaptive threshold, deskew) et un parser regex multi-pattern hiérarchisé. Le projet est livré production-ready : Dockerisé, testé sur 9 images réelles d'étiquettes marocaines, instrumenté Prometheus, conforme mypy strict.

**80% du focus = pipeline IA (preprocessing → OCR → parsing).**
**20% du focus = API + infra + UI minimale.**

---

## 1. ARCHITECTURE DU PROJET

```
notus-label-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                        # Entrée FastAPI, mount static, instrumentator
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py            # DI : DB session, services singletons
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── products.py            # GET /api/products (paginated)
│   │       ├── batches.py             # CRUD batches
│   │       └── analyze.py             # POST /api/analyze (cœur du projet)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                  # Pydantic Settings, .env
│   │   └── logging.py                 # Loguru config JSON structuré
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_preprocessor.py      # Pipeline OpenCV (CLAHE, threshold, deskew)
│   │   ├── ocr_service.py             # Multi-passes Tesseract + PaddleOCR + fusion
│   │   ├── label_parser.py            # Regex LOT + date, scoring, validation
│   │   └── camera_service.py          # Helpers upload image (validation MIME, taille)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── domain.py                  # SQLAlchemy ORM : Product, ProductBatch
│   │   └── schemas.py                 # Pydantic : OCRResult, ParseResult, AnalyzeResponse
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── product_repo.py            # Accès Product + cache Redis
│   │   └── batch_repo.py              # Accès ProductBatch
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py             # Lecture/écriture image, conversion formats
│       └── date_normalizer.py         # Normalisation date → ISO YYYY-MM
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures pytest : client, db, services
│   ├── unit/
│   │   ├── test_label_parser.py       # Test chaque regex isolément
│   │   ├── test_image_preprocessor.py # Test chaque étape du pipeline
│   │   └── test_date_normalizer.py    # Tous formats date → ISO
│   ├── integration/
│   │   ├── test_pipeline.py           # Pipeline complet sur 9 images réelles
│   │   └── test_api.py                # TestClient sur tous endpoints
│   └── fixtures/
│       ├── images/
│       │   ├── lot_001_white_cap.jpeg
│       │   ├── lot_002_barcode_box.jpeg
│       │   ├── lot_003_embossed.jpeg
│       │   ├── lot_004_yellow_box.jpeg
│       │   ├── lot_005_melatonine.jpeg
│       │   ├── lot_006_red_label.jpeg
│       │   ├── lot_007_qrcode.jpeg
│       │   ├── lot_008_arabic.jpeg
│       │   └── lot_009_white_box.jpeg
│       └── ground_truth.json
├── scripts/
│   ├── benchmark.py                   # Précision LOT/date sur 9 images + métriques
│   └── seed_products.py               # Insertion catalogue exemple
├── docker/
│   ├── Dockerfile                     # Multi-stage Python 3.11-slim + Tesseract
│   └── docker-compose.yml             # API + Postgres + Redis
├── migrations/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
├── static/
│   └── index.html                     # UI minimale single-page (caméra + résultat)
├── uploads/                           # Stockage images scannées (gitignored)
├── .github/
│   └── workflows/
│       └── ci.yml                     # Lint + tests sur push
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

**Principes architecturaux :**
- **Clean Architecture** : `domain` (models) / `infrastructure` (repositories, services externes) / `interface` (api/routes).
- **SOLID** : un service = une responsabilité. `ImagePreprocessor` ne fait QUE du traitement image ; `OCRService` ne fait QUE de l'OCR ; `LabelParser` ne fait QUE du parsing texte.
- **Injection de dépendances** via `app/api/dependencies.py` (FastAPI `Depends`).
- **Async-first** : SQLAlchemy 2.0 async, `asyncpg`. PaddleOCR/Tesseract sont CPU-bound → `run_in_executor`.
- **Zéro import circulaire** : `schemas.py` ne dépend que de `domain.py` (jamais l'inverse).

---

## 2. PHASES DE DÉVELOPPEMENT

### **Phase 1 — Setup & Infrastructure (1 jour)**

**Objectifs :** projet bootstrappé, Docker fonctionnel, CI verte, linting strict.

**Tâches :**
1. Init `pyproject.toml` Poetry avec toutes les dépendances (voir section 4).
2. `app/main.py` minimal :
   ```python
   from fastapi import FastAPI
   from app.core.config import settings
   from app.core.logging import configure_logging

   configure_logging()
   app = FastAPI(title=settings.app_name, version=settings.app_version)

   @app.get("/api/health")
   async def health() -> dict:
       return {"status": "ok"}
   ```
3. `app/core/config.py` via Pydantic Settings (toute config depuis `.env`, zéro secret en dur).
4. `docker/Dockerfile` multi-stage :
   ```dockerfile
   FROM python:3.11-slim AS base
   RUN apt-get update && apt-get install -y --no-install-recommends \
       tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara tesseract-ocr-eng \
       libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
       && rm -rf /var/lib/apt/lists/*
   # ... poetry install --no-dev
   ```
5. `docker/docker-compose.yml` : services `api`, `postgres:16`, `redis:7-alpine`.
6. `.github/workflows/ci.yml` : checkout → setup Python → poetry install → `ruff check` → `mypy --strict` → `pytest`.
7. `.pre-commit-config.yaml` : ruff, mypy, trailing-whitespace.

**Dépendances Phase 1 :**
`fastapi==0.111.0`, `uvicorn[standard]==0.29.0`, `pydantic-settings==2.2.1`, `python-dotenv==1.0.1`.

**Critère de sortie :** `docker-compose up` lance l'API ; `GET /api/health` retourne `{"status":"ok"}` ; CI verte.

---

### **Phase 2 — Modèle de données & API CRUD (1 jour)**

**Modèles SQLAlchemy (`app/models/domain.py`) :**

```python
from datetime import date, datetime
from sqlalchemy import ForeignKey, Numeric, String, Date, DateTime, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): pass

class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    price: Mapped[float] = mapped_column(Numeric(10, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    batches: Mapped[list["ProductBatch"]] = relationship(back_populates="product")

class ProductBatch(Base):
    __tablename__ = "product_batches"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    lot_number: Mapped[str] = mapped_column(String(50))
    expiration_date: Mapped[date] = mapped_column(Date)
    scan_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    product: Mapped["Product"] = relationship(back_populates="batches")
```

**Migrations Alembic** : `alembic init migrations` → configurer `env.py` pour `Base.metadata` → première révision.

**Endpoints REST :**
- `GET /api/products?cursor=&limit=20` — pagination cursor-based (cursor = dernier `id` vu).
- `GET /api/products/{id}` — détail.
- `POST /api/batches` — création manuelle (corps : `product_id`, `lot_number`, `expiration_date`).
- `GET /api/batches?product_id=` — liste filtrable.
- `GET /api/batches/{id}` — détail.

**Dépendances Phase 2 :**
`sqlalchemy[asyncio]==2.0.30`, `alembic==1.13.1`, `asyncpg==0.29.0`, `psycopg2-binary==2.9.9`.

**Critère de sortie :** Swagger `/docs` montre tous les endpoints fonctionnels, migrations passent, tests `test_api.py` (CRUD basique) verts.

---

### **Phase 3 — Pipeline de préprocessing image (2 jours) — CRITIQUE**

> **Plus le préprocessing est bon, plus l'OCR sera précis.** C'est ici qu'on gagne 20 points de précision.

**Service `app/services/image_preprocessor.py` :**

```python
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class PreprocessedVariants:
    """Toutes les variantes générées, à essayer en cascade par l'OCR."""
    original_gray: np.ndarray
    thresholded: np.ndarray
    inverted: np.ndarray
    clahe_enhanced: np.ndarray
    deskewed: np.ndarray

class ImagePreprocessor:
    """Pipeline OpenCV ordonné pour étiquettes pharmaceutiques."""

    def process(self, image_bgr: np.ndarray) -> PreprocessedVariants:
        # 1. Conversion BGR → RGB → grayscale
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 2. Redressement perspective (si contour rectangulaire détecté)
        gray = self._deskew(gray)

        # 3. Adaptive thresholding — JAMAIS cv2.threshold simple
        thresholded = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # 4. Débruitage (préserve les bords du texte)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 5. CLAHE — essentiel pour texte gravé blanc sur blanc
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(denoised)

        # 6. Dilatation légère pour reconnecter caractères fragmentés
        kernel = np.ones((2, 2), np.uint8)
        clahe_enhanced = cv2.dilate(clahe_enhanced, kernel, iterations=1)

        # 7. Inverted (pour texte blanc sur fond foncé)
        inverted = cv2.bitwise_not(thresholded)

        # 8. Unsharp masking si flou détecté
        if self._is_blurry(gray):
            gray = self._unsharp_mask(gray)

        return PreprocessedVariants(
            original_gray=gray,
            thresholded=thresholded,
            inverted=inverted,
            clahe_enhanced=clahe_enhanced,
            deskewed=gray,
        )

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Redresse l'image si angle > 1° (via moments)."""
        coords = np.column_stack(np.where(gray < 200))
        if len(coords) == 0:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 1.0:
            return gray
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _is_blurry(self, gray: np.ndarray, threshold: float = 100.0) -> bool:
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
```

**Cas spéciaux à gérer dans le pipeline :**
- Texte blanc sur fond blanc → variante `inverted`.
- Texte gravé en relief → CLAHE agressif (clipLimit=2.0).
- Image floue → unsharp masking auto.
- Angle > 1° → deskew via moments.

**Dépendances Phase 3 :**
`opencv-python==4.9.0.80`, `Pillow==10.3.0`, `numpy==1.26.4`.

**Critère de sortie :** `test_image_preprocessor.py` vérifie que chaque variante est une `np.ndarray` valide non-nulle, et que la rotation deskew se déclenche sur image tournée 15°.

---

### **Phase 4 — OCR & extraction (3 jours)**

**Stratégie multi-passes** (ordre de priorité strict, on s'arrête au premier succès confident) :

| # | Engine | Image input | Quand |
|---|--------|-------------|-------|
| 1 | PaddleOCR | `original_gray` | Première passe par défaut |
| 2 | Tesseract PSM 6 | `thresholded` | Si Paddle confidence < 0.7 |
| 3 | Tesseract PSM 11 | `inverted` | Si étiquette à texte sparse / blanc sur foncé |
| 4 | PaddleOCR | `clahe_enhanced` | Si texte gravé peu visible |
| 5 | Fusion | — | Garder le résultat avec meilleur score global |

**Configs OCR :**

```python
# app/services/ocr_service.py

TESSERACT_CONFIGS = {
    "standard": "--oem 3 --psm 6 -l fra+eng",
    "sparse":   "--oem 3 --psm 11 -l fra+eng",
    "whitelist": '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/:.- " -l fra',
    "arabic":   "--oem 3 --psm 6 -l ara+fra",
}

PADDLE_CONFIG = {
    "use_angle_cls": True,
    "lang": "fr",
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.5,
    "rec_batch_num": 6,
    "use_gpu": False,
    "show_log": False,
}
```

**Schéma OCRResult (`app/models/schemas.py`) :**

```python
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    text: str
    confidence: float
    coordinates: list[tuple[float, float]]  # 4 points (polygon)

class OCRResult(BaseModel):
    raw_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    engine_used: str  # "paddleocr" | "tesseract_psm6" | "tesseract_psm11" | "fusion"
    processing_time_ms: int
    bounding_boxes: list[BoundingBox] = []
```

**Squelette OCRService :**

```python
import asyncio
import time
import pytesseract
from paddleocr import PaddleOCR

class OCRService:
    def __init__(self) -> None:
        self._paddle = PaddleOCR(**PADDLE_CONFIG)

    async def extract(self, variants: PreprocessedVariants) -> OCRResult:
        """Multi-passes, retourne le meilleur résultat global."""
        loop = asyncio.get_running_loop()
        results: list[OCRResult] = []

        # Pass 1: PaddleOCR sur gray
        results.append(await loop.run_in_executor(None, self._paddle_extract,
                                                  variants.original_gray, "paddle_gray"))
        # Pass 2: Tesseract PSM 6 thresholded
        results.append(await loop.run_in_executor(None, self._tesseract_extract,
                                                  variants.thresholded, "standard", "tesseract_psm6"))
        # Pass 3: Tesseract PSM 11 inverted
        results.append(await loop.run_in_executor(None, self._tesseract_extract,
                                                  variants.inverted, "sparse", "tesseract_psm11"))
        # Pass 4: PaddleOCR CLAHE
        results.append(await loop.run_in_executor(None, self._paddle_extract,
                                                  variants.clahe_enhanced, "paddle_clahe"))

        return self._fuse(results)

    def _fuse(self, results: list[OCRResult]) -> OCRResult:
        """Fusion : on garde le résultat avec la meilleure confidence,
        mais on concatène les textes uniques pour maximiser le recall du parser."""
        best = max(results, key=lambda r: r.confidence)
        all_text = "\n".join(dict.fromkeys(r.raw_text for r in results))  # dédup ordre
        return OCRResult(
            raw_text=all_text,
            confidence=best.confidence,
            engine_used=f"fusion({best.engine_used})",
            processing_time_ms=sum(r.processing_time_ms for r in results),
            bounding_boxes=best.bounding_boxes,
        )
```

> ⚠️ **PIÈGE** : ne jamais appeler Paddle/Tesseract directement dans une route async sans `run_in_executor` — sinon tout le event loop bloque.

**Dépendances Phase 4 :**
`pytesseract==0.3.10`, `paddleocr==2.7.3`, `paddlepaddle==2.6.1`.

**Critère de sortie :** `OCRService.extract()` retourne un `OCRResult` valide sur les 9 images réelles, temps moyen < 3s/image en CPU.

---

### **Phase 5 — Parser intelligent LOT + Date (2 jours) — CŒUR ALGORITHMIQUE**

**Patterns LOT** (ordre priorité décroissante, on s'arrête au premier match) :

```python
LOT_PATTERNS: list[tuple[str, float]] = [
    # (pattern, confidence_si_match)
    (r'LOT\s*[:\s]\s*([A-Z]{1,3}\d{3,}[A-Z]?\d*)',     1.0),  # LOT: RN620, LOT: PF1122001
    (r'LOT\s*[:\s]\s*([A-Z0-9]{4,12})',                 1.0),  # LOT: CIX31, LOT: N3661
    (r'LOT\s+(\d{4,}\s?\d*)',                           0.8),  # LOT 251391 1, LOT 24040 51
    (r'\bLOT[:\s]+([A-Z0-9\s]{3,15})',                  0.6),  # générique
    (r'^([A-Z]{1,2}\d{3,}[A-Z]?\d*)\s+\d{2}[/\-]',     0.4),  # K4S598 05/24 sans préfixe
]
```

**Patterns DATE EXPIRATION** :

```python
DATE_PATTERNS: list[tuple[str, float]] = [
    (r'(?:EXP|PER|PEREMPTION)[:\s]+(\d{2}[/\s\-]\d{4})',  1.0),  # EXP: 04/2027, PER: 10/2025
    (r'(?:EXP|PER)[:\s]+(\d{2}[/\s\-]\d{2})\b',           1.0),  # EXP 05/27, PER: 10/27
    (r'(?:EXP|PER)[:\s]+(\d{2}\s\d{4})',                  1.0),  # EXP 04 2028
    (r'(?:EXP|PER)[:\s]+(\d{2}[/\-]\d{2,4})',             0.8),  # générique
    (r'\b(\d{2}[/\-]\d{4})\b(?!.*LOT)',                   0.6),  # fallback date isolée
]
```

**Normalisation date (`app/utils/date_normalizer.py`) :**

```python
import re
from datetime import date

def normalize_to_iso(raw: str) -> str | None:
    """
    '10/2025' → '2025-10'
    '05/27'   → '2027-05'  (année 2 chiffres : <50 → 20XX, sinon 19XX)
    '10 2026' → '2026-10'
    '04-2028' → '2028-04'
    """
    cleaned = re.sub(r'[\s\-]', '/', raw.strip())
    parts = cleaned.split('/')
    if len(parts) != 2:
        return None
    month, year = parts
    if not month.isdigit() or not year.isdigit():
        return None
    m = int(month)
    if not 1 <= m <= 12:
        return None
    if len(year) == 2:
        y = int(year)
        year = f"20{year}" if y < 50 else f"19{year}"
    return f"{year}-{m:02d}"
```

**Patterns à IGNORER explicitement** (anti-faux-positifs) :
- `PPV`, `PPC`, `PPH` (prix de vente publique / prix public)
- `ACL` (code Article)
- `CE:N°`, `CE N°` (norme européenne)
- `DOM` (date de **fabrication**, pas expiration)
- `FAB` (fabrication)
- Toute séquence qui suit un `€`, `MAD`, `DH`, ou `.XX` final (prix)

**Validation post-parsing :**

```python
from datetime import date

def validate_parsed(lot: str | None, exp_iso: str | None) -> list[str]:
    warnings: list[str] = []
    if lot:
        if not (3 <= len(lot.replace(" ", "")) <= 15):
            warnings.append(f"LOT longueur suspecte: {lot}")
        if re.match(r'^\d+\.\d+$', lot):
            warnings.append(f"LOT ressemble à un prix: {lot}")
    if exp_iso:
        y, m = exp_iso.split("-")
        exp_date = date(int(y), int(m), 1)
        if exp_date < date.today():
            warnings.append(f"Date expiration dans le passé: {exp_iso}")
    return warnings
```

**ParseResult schema :**

```python
class ParseResult(BaseModel):
    lot_number: str | None
    expiration_date: str | None  # YYYY-MM
    confidence_lot: float
    confidence_date: float
    confidence_global: float
    raw_matches: dict[str, list[str]] = {}  # debug : tous les matches trouvés
    warnings: list[str] = []
```

**Service LabelParser :**

```python
class LabelParser:
    def parse(self, raw_text: str) -> ParseResult:
        text = raw_text.upper()
        lot, conf_lot, lot_matches = self._extract_lot(text)
        exp_iso, conf_date, date_matches = self._extract_date(text)
        warnings = validate_parsed(lot, exp_iso)
        return ParseResult(
            lot_number=lot,
            expiration_date=exp_iso,
            confidence_lot=conf_lot,
            confidence_date=conf_date,
            confidence_global=(conf_lot + conf_date) / 2,
            raw_matches={"lot": lot_matches, "date": date_matches},
            warnings=warnings,
        )

    def _extract_lot(self, text: str) -> tuple[str | None, float, list[str]]:
        # Filtrer les lignes contenant PPV/PPC/DOM/FAB/ACL
        lines = [l for l in text.splitlines()
                 if not any(bad in l for bad in ("PPV", "PPC", "PPH", "ACL", "DOM", "FAB"))]
        clean_text = "\n".join(lines)
        all_matches = []
        for pattern, conf in LOT_PATTERNS:
            matches = re.findall(pattern, clean_text, re.MULTILINE)
            all_matches.extend(matches)
            if matches:
                return matches[0].strip(), conf, all_matches
        return None, 0.0, all_matches

    def _extract_date(self, text: str) -> tuple[str | None, float, list[str]]:
        # Si plusieurs dates : garder la PLUS TARDIVE (c'est l'expiration vs DOM)
        candidates: list[tuple[str, float]] = []
        for pattern, conf in DATE_PATTERNS:
            for m in re.findall(pattern, text):
                iso = normalize_to_iso(m)
                if iso:
                    candidates.append((iso, conf))
            if candidates:
                break  # on s'arrête au premier pattern qui matche
        if not candidates:
            return None, 0.0, []
        candidates.sort(key=lambda x: x[0], reverse=True)  # plus tardive = expiration
        best_iso, best_conf = candidates[0]
        return best_iso, best_conf, [c[0] for c in candidates]
```

**Critère de sortie :** `test_label_parser.py` couvre les 9 formats de LOT et 5 formats de date, tous verts.

---

### **Phase 6 — Endpoint d'analyse vision (1 jour)**

**`POST /api/analyze`** — pipeline complet end-to-end.

```python
# app/api/routes/analyze.py
import uuid
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile, HTTPException
from loguru import logger

router = APIRouter()

@router.post("/analyze")
async def analyze(
    background: BackgroundTasks,
    image: UploadFile = File(...),
    product_name: str = Form(...),
    preprocessor: ImagePreprocessor = Depends(get_preprocessor),
    ocr: OCRService = Depends(get_ocr_service),
    parser: LabelParser = Depends(get_parser),
    batch_repo: BatchRepository = Depends(get_batch_repo),
) -> AnalyzeResponse:
    start = time.perf_counter()
    logger.info("analyze_start", product_name=product_name, filename=image.filename)

    # 1. Validation MIME + sauvegarde fichier
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Format non supporté")
    filename = f"scan_{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpeg"
    path = Path("uploads") / filename
    async with aiofiles.open(path, "wb") as f:
        await f.write(await image.read())
    logger.info("image_saved", path=str(path))

    # 2. Preprocessing
    img_array = read_image(path)
    variants = preprocessor.process(img_array)
    logger.info("preprocessing_done", duration_ms=int((time.perf_counter() - start) * 1000))

    # 3. OCR multi-passes
    ocr_result = await ocr.extract(variants)
    logger.info("ocr_done", engine=ocr_result.engine_used, confidence=ocr_result.confidence)

    # 4. Parsing
    parsed = parser.parse(ocr_result.raw_text)
    logger.info("parsing_done", lot=parsed.lot_number, exp=parsed.expiration_date,
                confidence=parsed.confidence_global)

    # 5. Warning si confidence faible
    if parsed.confidence_global < 0.5:
        parsed.warnings.append("Résultat incertain — vérification manuelle recommandée")

    # 6. Sauvegarde DB en background (ne bloque pas la réponse)
    if parsed.lot_number and parsed.expiration_date:
        background.add_task(batch_repo.save_from_scan,
                            product_name, parsed, str(path))

    total_ms = int((time.perf_counter() - start) * 1000)
    return AnalyzeResponse(
        product_name=product_name,
        lot_number=parsed.lot_number,
        expiration_date=parsed.expiration_date,
        confidence_global=parsed.confidence_global,
        confidence_lot=parsed.confidence_lot,
        confidence_date=parsed.confidence_date,
        raw_ocr_text=ocr_result.raw_text,
        engine_used=ocr_result.engine_used,
        processing_time_ms=total_ms,
        image_path=str(path),
        warnings=parsed.warnings,
    )
```

**Format de réponse :**

```json
{
  "product_name": "Doliprane 1000mg",
  "lot_number": "RN620",
  "expiration_date": "2026-10",
  "confidence_global": 0.85,
  "confidence_lot": 0.90,
  "confidence_date": 0.80,
  "raw_ocr_text": "...",
  "engine_used": "fusion(paddleocr)",
  "processing_time_ms": 1240,
  "image_path": "uploads/scan_20250520_143022_a3f1b2c4.jpeg",
  "warnings": []
}
```

**Logging structuré (`app/core/logging.py`) :**

```python
import sys
from loguru import logger

def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stdout, serialize=True, level="INFO")  # JSON structuré
```

**Dépendances Phase 6 :**
`python-multipart==0.0.9`, `loguru==0.7.2`, `aiofiles==23.2.1`.

---

### **Phase 7 — Tests complets (2 jours)**

**Tests unitaires (`tests/unit/`) :**

`test_label_parser.py` — tester chaque regex sur strings brutes :

```python
import pytest
from app.services.label_parser import LabelParser

parser = LabelParser()

@pytest.mark.parametrize("text,expected_lot", [
    ("LOT: RN620 EXP: 10/2026", "RN620"),
    ("LOT: N3661\nPER: 10/2027", "N3661"),
    ("LOT: CIX31\nEXP 07/25", "CIX31"),
    ("LOT 251391 1\nEXP 04 2028", "251391 1"),
    ("LOT 24040 51 PER: 04/2027", "24040 51"),
    ("LOT: PF1122001 PER: 10/2025", "PF1122001"),
    ("LOT: 2JV0951 EXP: 09/2025 FAB: 09/2023", "2JV0951"),
    ("K4S598 05/24", "K4S598"),
])
def test_extract_lot(text: str, expected_lot: str) -> None:
    result = parser.parse(text)
    assert result.lot_number == expected_lot

@pytest.mark.parametrize("text,expected_iso", [
    ("EXP: 10/2026", "2026-10"),
    ("PER: 10/27", "2027-10"),
    ("EXP 04 2028", "2028-04"),
    ("EXP: 10-2022", "2022-10"),
    ("EXP 05/27", "2027-05"),
])
def test_extract_date(text: str, expected_iso: str) -> None:
    result = parser.parse(text)
    assert result.expiration_date == expected_iso

def test_ignore_dom_keep_exp() -> None:
    text = "DOM: 01/2023 EXP: 10/2026"
    result = parser.parse(text)
    assert result.expiration_date == "2026-10"

def test_ignore_ppv_keep_lot() -> None:
    text = "PPV: 45.50 MAD\nLOT: RN620"
    result = parser.parse(text)
    assert result.lot_number == "RN620"
```

`test_image_preprocessor.py` — chaque étape retourne une image valide.
`test_date_normalizer.py` — couverture exhaustive de tous les formats.

**Coverage minimum : 80%** (configuré dans `pyproject.toml`).

**Tests d'intégration sur 9 images réelles (`tests/integration/test_pipeline.py`) :**

```python
import json
import pytest
from pathlib import Path
from app.services.image_preprocessor import ImagePreprocessor
from app.services.ocr_service import OCRService
from app.services.label_parser import LabelParser
from app.utils.image_utils import read_image

FIXTURES_DIR = Path("tests/fixtures/images")
GROUND_TRUTH = json.loads(Path("tests/fixtures/ground_truth.json").read_text())

@pytest.mark.parametrize("sample", GROUND_TRUTH)
@pytest.mark.asyncio
async def test_full_pipeline_real_images(sample: dict) -> None:
    image_path = FIXTURES_DIR / sample["image"]
    img = read_image(image_path)
    variants = ImagePreprocessor().process(img)
    ocr_result = await OCRService().extract(variants)
    parsed = LabelParser().parse(ocr_result.raw_text)
    assert parsed.lot_number == sample["expected"]["lot"], (
        f"LOT mismatch on {sample['image']}: got {parsed.lot_number}"
    )
    assert parsed.expiration_date == sample["expected"]["expiration_date"], (
        f"Date mismatch on {sample['image']}: got {parsed.expiration_date}"
    )
```

**Ground truth (`tests/fixtures/ground_truth.json`) :**

```json
[
  {"image": "lot_001_white_cap.jpeg",    "expected": {"lot": "PF1122001", "expiration_date": "2025-10"}, "difficulty": "medium", "notes": "texte gravé relief sur couvercle blanc"},
  {"image": "lot_002_barcode_box.jpeg",  "expected": {"lot": "K4S598",    "expiration_date": "2027-05"}, "difficulty": "hard",   "notes": "LOT sans préfixe, bruit code-barres"},
  {"image": "lot_003_embossed.jpeg",     "expected": {"lot": "251391 1",  "expiration_date": "2028-04"}, "difficulty": "hard",   "notes": "embossé faible contraste, PPV à ignorer"},
  {"image": "lot_004_yellow_box.jpeg",   "expected": {"lot": "N3661",     "expiration_date": "2027-10"}, "difficulty": "easy",   "notes": "fond jaune bon contraste"},
  {"image": "lot_005_melatonine.jpeg",   "expected": {"lot": "24040 51",  "expiration_date": "2027-04"}, "difficulty": "easy",   "notes": "étiquette claire fond blanc"},
  {"image": "lot_006_red_label.jpeg",    "expected": {"lot": "CIX31",     "expiration_date": "2025-07"}, "difficulty": "medium", "notes": "étiquette collée fond rouge, PPC à ignorer"},
  {"image": "lot_007_qrcode.jpeg",       "expected": {"lot": "2713076",   "expiration_date": "2025-08"}, "difficulty": "medium", "notes": "QR code à ignorer"},
  {"image": "lot_008_arabic.jpeg",       "expected": {"lot": "RN620",     "expiration_date": "2026-10"}, "difficulty": "hard",   "notes": "bilingue arabe/français, DOM différent de EXP"},
  {"image": "lot_009_white_box.jpeg",    "expected": {"lot": "2JV0951",   "expiration_date": "2025-09"}, "difficulty": "easy",   "notes": "FAB à ignorer, EXP clairement labelisé"}
]
```

**Tests API (`tests/integration/test_api.py`) :** `TestClient` sur tous endpoints, upload d'une fixture, cas d'erreur (image invalide, format non supporté).

**Script benchmark (`scripts/benchmark.py`) :**

```python
"""
Benchmark sur les 9 images réelles.
Sortie :
  - Précision LOT (X/9)
  - Précision date (X/9)
  - Précision par difficulté
  - Temps moyen ms/image
  - Rapport JSON dans benchmark_report.json
"""
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

async def main() -> None:
    truth = json.loads(Path("tests/fixtures/ground_truth.json").read_text())
    fixtures = Path("tests/fixtures/images")
    pre = ImagePreprocessor()
    ocr = OCRService()
    parser = LabelParser()

    by_diff: dict[str, dict] = defaultdict(lambda: {"lot_ok": 0, "date_ok": 0, "total": 0})
    times: list[int] = []
    details: list[dict] = []

    for sample in truth:
        start = time.perf_counter()
        img = read_image(fixtures / sample["image"])
        variants = pre.process(img)
        ocr_result = await ocr.extract(variants)
        parsed = parser.parse(ocr_result.raw_text)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        times.append(elapsed_ms)

        lot_ok  = parsed.lot_number      == sample["expected"]["lot"]
        date_ok = parsed.expiration_date == sample["expected"]["expiration_date"]
        diff = sample["difficulty"]
        by_diff[diff]["total"] += 1
        by_diff[diff]["lot_ok"] += int(lot_ok)
        by_diff[diff]["date_ok"] += int(date_ok)

        details.append({
            "image": sample["image"], "difficulty": diff,
            "lot_ok": lot_ok, "date_ok": date_ok,
            "expected": sample["expected"],
            "got": {"lot": parsed.lot_number, "exp": parsed.expiration_date},
            "confidence": parsed.confidence_global, "time_ms": elapsed_ms,
        })

    total_lot  = sum(d["lot_ok"]  for d in by_diff.values())
    total_date = sum(d["date_ok"] for d in by_diff.values())
    total      = sum(d["total"]   for d in by_diff.values())

    report = {
        "summary": {
            "precision_lot": f"{total_lot}/{total}",
            "precision_date": f"{total_date}/{total}",
            "avg_time_ms": sum(times) // len(times),
        },
        "by_difficulty": dict(by_diff),
        "details": details,
    }
    Path("benchmark_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

---

### **Phase 8 — Interface minimaliste (1 jour)**

**`static/index.html`** — single page, vanilla JS, mobile-first :

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Notus Label Agent</title>
  <style>
    body { font-family: system-ui; max-width: 480px; margin: auto; padding: 16px; }
    video, canvas { width: 100%; border-radius: 8px; background: #000; }
    button { padding: 12px; width: 100%; font-size: 16px; margin: 8px 0; }
    .result { background: #f4f4f4; padding: 12px; border-radius: 8px; margin-top: 16px; }
    .warn { color: #c00; }
  </style>
</head>
<body>
  <h1>Notus Label Agent</h1>
  <input id="productName" placeholder="Nom du produit (ex: Doliprane 1000mg)" />
  <button id="openCam">Ouvrir caméra</button>
  <video id="video" autoplay playsinline hidden></video>
  <button id="capture" hidden>Capturer & analyser</button>
  <canvas id="canvas" hidden></canvas>
  <div id="result" class="result" hidden></div>

  <h2>Derniers scans</h2>
  <div id="history"></div>

  <script>
    const $ = id => document.getElementById(id);
    let stream;

    $('openCam').onclick = async () => {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      });
      $('video').srcObject = stream;
      $('video').hidden = false;
      $('capture').hidden = false;
    };

    $('capture').onclick = async () => {
      const v = $('video'), c = $('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.9));
      const fd = new FormData();
      fd.append('image', blob, 'scan.jpeg');
      fd.append('product_name', $('productName').value || 'Inconnu');
      const res = await fetch('/api/analyze', { method: 'POST', body: fd });
      const data = await res.json();
      $('result').hidden = false;
      $('result').innerHTML = `
        <strong>${data.product_name}</strong><br>
        LOT: <b>${data.lot_number || '—'}</b><br>
        Expiration: <b>${data.expiration_date || '—'}</b><br>
        Confiance: ${(data.confidence_global * 100).toFixed(0)}%<br>
        ${data.warnings.map(w => `<div class="warn">⚠ ${w}</div>`).join('')}
      `;
      loadHistory();
    };

    async function loadHistory() {
      const res = await fetch('/api/batches?limit=10');
      const list = await res.json();
      $('history').innerHTML = list.map(b =>
        `<div>${b.lot_number} — ${b.expiration_date}</div>`
      ).join('');
    }
    loadHistory();
  </script>
</body>
</html>
```

Servir via `app.mount("/", StaticFiles(directory="static", html=True), name="static")` dans `main.py`.

---

### **Phase 9 — Optimisation & monitoring (1 jour)**

- **Cache Redis** : `GET /api/products` cache TTL 300s, invalidation sur `POST /api/products`.
- **Prometheus** : middleware `prometheus-fastapi-instrumentator` exposant `/metrics`. Métriques custom :
  - `ocr_processing_seconds{engine=...}` (histogram)
  - `ocr_confidence_score{engine=...}` (gauge)
  - `analyze_success_total` / `analyze_failure_total` (counter)
- **Health check** : `GET /api/health` retourne `{db, redis, tesseract}`.
- **Swagger** : descriptions détaillées sur chaque endpoint (`summary`, `description`, exemples).
- **README.md** complet : prérequis, install, commandes, exemples curl, captures screenshots du benchmark.

**Dépendances Phase 9 :**
`redis[hiredis]==5.0.4`, `prometheus-fastapi-instrumentator==6.1.0`.

---

## 3. SPÉCIFICATIONS TECHNIQUES

### `pyproject.toml` complet

```toml
[tool.poetry]
name = "notus-label-agent"
version = "0.1.0"
description = "Agent IA d'extraction LOT + date d'expiration sur étiquettes pharmaceutiques"
authors = ["Ibrahim Stouri"]
readme = "README.md"
packages = [{include = "app"}]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "0.111.0"
uvicorn = {extras = ["standard"], version = "0.29.0"}
sqlalchemy = {extras = ["asyncio"], version = "2.0.30"}
alembic = "1.13.1"
asyncpg = "0.29.0"
psycopg2-binary = "2.9.9"
pydantic-settings = "2.2.1"
python-dotenv = "1.0.1"
python-multipart = "0.0.9"
opencv-python = "4.9.0.80"
Pillow = "10.3.0"
numpy = "1.26.4"
pytesseract = "0.3.10"
paddleocr = "2.7.3"
paddlepaddle = "2.6.1"
loguru = "0.7.2"
aiofiles = "23.2.1"
redis = {extras = ["hiredis"], version = "5.0.4"}
prometheus-fastapi-instrumentator = "6.1.0"

[tool.poetry.group.dev.dependencies]
pytest = "8.2.0"
pytest-asyncio = "0.23.6"
pytest-cov = "5.0.0"
httpx = "0.27.0"
ruff = "0.4.4"
mypy = "1.10.0"
pre-commit = "3.7.1"

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "slow: tests lents (intégration sur images)",
    "integration: tests bout-en-bout"
]

[tool.coverage.report]
fail_under = 80
exclude_lines = ["pragma: no cover", "if __name__"]
```

### `.env.example`

```env
APP_NAME=notus-label-agent
APP_VERSION=0.1.0
DEBUG=false

# Database
DATABASE_URL=postgresql+asyncpg://notus:notus@postgres:5432/notus
DATABASE_URL_SYNC=postgresql://notus:notus@postgres:5432/notus

# Redis
REDIS_URL=redis://redis:6379/0
CACHE_TTL_SECONDS=300

# OCR
TESSERACT_CMD=/usr/bin/tesseract
PADDLE_USE_GPU=false

# Storage
UPLOADS_DIR=uploads
MAX_UPLOAD_SIZE_MB=10

# Logging
LOG_LEVEL=INFO
```

### `docker-compose.yml`

```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./app:/app/app
      - ./uploads:/app/uploads
    depends_on: [postgres, redis]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: notus
      POSTGRES_PASSWORD: notus
      POSTGRES_DB: notus
    volumes: ["pgdata:/var/lib/postgresql/data"]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

volumes:
  pgdata:
```

---

## 4. CRITÈRES DE QUALITÉ (NON NÉGOCIABLES)

- ✅ Code **100% typé** (`mypy --strict` passe sans erreur).
- ✅ **Architecture clean** : `domain` / `infrastructure` / `interface` séparés.
- ✅ **SOLID** : chaque service a une responsabilité unique.
- ✅ **Logs structurés JSON** (loguru `serialize=True`).
- ✅ Toutes les variables via **Pydantic Settings** depuis `.env`.
- ✅ **Zéro secret en dur** dans le code.
- ✅ **Zéro import circulaire**.
- ✅ **Docstrings** sur toutes les méthodes publiques.
- ✅ **Coverage ≥ 80%**.
- ✅ Ruff + mypy + pre-commit en place.

---

## 5. COMMANDES UTILES

```bash
# Installation
poetry install

# Dev server local
uvicorn app.main:app --reload --port 8000

# Tests complets avec coverage
pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Tests intégration uniquement
pytest tests/integration/ -v -m "not slow"

# Benchmark sur 9 images réelles
python scripts/benchmark.py

# Seed du catalogue
python scripts/seed_products.py

# Docker (build + run complet)
docker-compose up --build

# Migrations Alembic
alembic upgrade head
alembic revision --autogenerate -m "description"

# Lint + type-check
ruff check app/ tests/
ruff format app/ tests/
mypy app/ --strict

# Pre-commit (à installer une fois)
pre-commit install
pre-commit run --all-files
```

---

## 6. PIÈGES À ÉVITER

| Piège | Pourquoi c'est mauvais | Solution |
|-------|----------------------|----------|
| `cv2.threshold` simple | Échoue dès qu'il y a variation d'éclairage | Toujours `cv2.adaptiveThreshold` |
| PaddleOCR appelé en sync dans une route async | Bloque tout l'event loop | `await loop.run_in_executor(None, ...)` |
| Stockage image en base64 dans la DB | Explose la taille, ralentit les queries | Stocker `scan_image_path` (chemin fichier) |
| Un seul moteur OCR | Échec sur 30% des étiquettes difficiles | Multi-passes Tesseract + PaddleOCR + fusion |
| Confondre `DOM` (fabrication) et `EXP` | Donne une date dans le passé | Filtrer `DOM`/`FAB` AVANT le regex date |
| Matcher un prix `PPV: 45.50` comme un LOT | Faux positifs catastrophiques | Filtrer les lignes contenant `PPV`/`PPC`/`PPH`/`ACL` |
| Tesseract pas installé dans le conteneur | Crash au runtime | `apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara` dans le Dockerfile |
| Année 2 chiffres mal normalisée | `05/27` → `1927-05` | `if y < 50: 20XX else 19XX` |
| Plusieurs dates trouvées → garder la première | Risque de prendre `DOM` au lieu de `EXP` | Garder la **plus tardive** (sort desc) |
| Lancer OCR sans variantes preprocessing | Échec sur texte gravé / blanc sur blanc | Toujours essayer `clahe_enhanced` ET `inverted` |

---

## 7. ROADMAP POST-MVP

- 🎯 Fine-tuning modèle OCR custom (PaddleOCR rec model) si volume > 10 000 scans.
- 🎯 **Active learning** : queue des cas `confidence_global < 0.5` corrigés à la main, réinjectés pour améliorer les patterns.
- 🎯 **App mobile native React Native** (certification Meta de l'auteur), avec capture optimisée et offline-first.
- 🎯 Intégration directe dans le catalogue Notus via API (webhook sur création de batch).
- 🎯 **Mode batch** : scanner plusieurs produits en série (queue Redis + worker Celery).
- 🎯 Détection automatique de lots dupliqués (alerte si même LOT scanné > 1 fois).
- 🎯 Dashboard analytique : taux d'erreur par fournisseur, temps moyen de saisie économisé.

---

## 8. RÉSUMÉ — PRIORITÉS D'IMPLÉMENTATION

Si tu dois prioriser dans le temps :

1. **🔥 Cœur du projet (80% du temps)** :
   - Phase 3 (preprocessing) + Phase 4 (OCR multi-passes) + Phase 5 (parser regex) + Phase 7 (tests sur 9 images réelles).
   - C'est ce que le recruteur évalue.

2. **⚙️ Infra solide (15% du temps)** :
   - Phase 1 (Docker, CI, lint), Phase 2 (DB, CRUD), Phase 6 (endpoint analyze).

3. **🎨 Interface (5% du temps)** :
   - Phase 8 (single HTML page).

4. **📊 Bonus différenciants (si temps)** :
   - Phase 9 (Prometheus, Redis cache, health checks).
   - Script `benchmark.py` avec rapport JSON formaté.

---

**Le critère ultime de réussite : `python scripts/benchmark.py` affiche ≥ 7/9 sur LOT et ≥ 7/9 sur date.**

---

## 9. SKILLS ROUTING — Python / FastAPI / PostgreSQL / Redis Project

| Phase | Skills / Agents |
|-------|----------------|
| Design/UI | `frontend-design` (static/index.html only — no framework) |
| Backend/API | `python-patterns`, `api-design`, `backend-patterns`, `silent-failure-hunter` agent |
| Database | `postgres-patterns`, `database-migrations` |
| Testing | `tdd-guide` agent, `tdd-workflow`, `pr-test-analyzer` agent, `python-testing` |
| Security | `security-reviewer` agent, `security-review`, `silent-failure-hunter` agent |
| Performance | `performance-optimizer` agent, `cost-aware-llm-pipeline` (OCR inference cost) |
| Refactor | `refactor-cleaner` agent, `code-simplifier` agent |
| PR/Commit | `finishing-a-development-branch`, `pr-review-toolkit`, `commit-commands` |
| Docs lookup | `docs-lookup` agent — FastAPI / SQLAlchemy / PaddleOCR / Alembic live docs |
| Docker/Infra | `docker-patterns` |
| Type checking | `pyright-lsp` plugin (auto-active on .py files) |
| Code review | `python-reviewer` agent (mandatory on every .py change) |
| Build errors | `build-error-resolver` agent → `pytorch-build-resolver` for PaddleOCR/paddle issues |

> Global routing rules in `~/.claude/CLAUDE.md` always active on top of these.
