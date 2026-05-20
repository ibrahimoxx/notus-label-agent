"""Benchmark pipeline on 9 real label images. Outputs precision + timing report."""
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from app.services.image_preprocessor import ImagePreprocessor
from app.services.label_parser import LabelParser
from app.services.ocr_service import OCRService
from app.utils.image_utils import read_image

GROUND_TRUTH = json.loads(
    (Path(__file__).parent.parent / "tests" / "fixtures" / "ground_truth.json").read_text()
)
FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "images"


async def main() -> None:
    pre = ImagePreprocessor()
    ocr = OCRService()
    parser = LabelParser()

    by_diff: dict[str, dict[str, int]] = defaultdict(
        lambda: {"lot_ok": 0, "date_ok": 0, "total": 0}
    )
    times: list[int] = []
    details: list[dict] = []

    for sample in GROUND_TRUTH:
        t0 = time.perf_counter()
        img = read_image(FIXTURES / sample["image"])
        variants = pre.process(img)
        ocr_result = await ocr.extract(variants)
        parsed = parser.parse(ocr_result.raw_text)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        times.append(elapsed_ms)

        lot_ok = parsed.lot_number == sample["expected"]["lot"]
        date_ok = parsed.expiration_date == sample["expected"]["expiration_date"]
        diff = sample["difficulty"]
        by_diff[diff]["total"] += 1
        by_diff[diff]["lot_ok"] += int(lot_ok)
        by_diff[diff]["date_ok"] += int(date_ok)

        status = "OK" if lot_ok and date_ok else "FAIL"
        print(
            f"[{status}] {sample['image']:<35} "
            f"LOT={('OK' if lot_ok else f'FAIL got={parsed.lot_number!r}'):<20} "
            f"DATE={('OK' if date_ok else f'FAIL got={parsed.expiration_date!r}'):<20} "
            f"{elapsed_ms}ms"
        )

        details.append(
            {
                "image": sample["image"],
                "difficulty": diff,
                "lot_ok": lot_ok,
                "date_ok": date_ok,
                "expected": sample["expected"],
                "got": {
                    "lot": parsed.lot_number,
                    "exp": parsed.expiration_date,
                },
                "confidence": parsed.confidence_global,
                "time_ms": elapsed_ms,
            }
        )

    total_lot = sum(d["lot_ok"] for d in by_diff.values())
    total_date = sum(d["date_ok"] for d in by_diff.values())
    total = sum(d["total"] for d in by_diff.values())

    report = {
        "summary": {
            "precision_lot": f"{total_lot}/{total}",
            "precision_date": f"{total_date}/{total}",
            "avg_time_ms": sum(times) // len(times),
        },
        "by_difficulty": {k: dict(v) for k, v in by_diff.items()},
        "details": details,
    }

    Path("benchmark_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(json.dumps(report["summary"], indent=2))
    print("=" * 60)

    lot_ok_count = int(report["summary"]["precision_lot"].split("/")[0])
    date_ok_count = int(report["summary"]["precision_date"].split("/")[0])
    if lot_ok_count >= 7 and date_ok_count >= 7:
        print("TARGET MET: >=7/9 on both LOT and date")
    else:
        print(f"TARGET NOT MET: need >=7/9, got LOT={lot_ok_count}/9 DATE={date_ok_count}/9")


if __name__ == "__main__":
    asyncio.run(main())
