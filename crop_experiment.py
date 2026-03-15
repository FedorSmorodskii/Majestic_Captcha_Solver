from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


# === НАСТРОЙКИ ДЛЯ ЭКСПЕРИМЕНТОВ ===

# Капча, с которой экспериментируем
INPUT_IMAGE: Path = Path("captchas") / "captcha_004.png"

# Сколько пикселей обрезать с каждой стороны
CROP_TOP: int = 18
CROP_BOTTOM: int = 18
CROP_LEFT: int = 46
CROP_RIGHT: int = 50

# На сколько частей делим по горизонтали (цифры)
NUM_SLICES: int = 6

# Куда складывать результаты
OUTPUT_DIR: Path = Path("crop_debug")

# Открывать ли превьюшки в просмотрщике
OPEN_PREVIEW: bool = False


@dataclass(frozen=True)
class Crop:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def crop_image(img: Image.Image, crop: Crop) -> Image.Image:
    w, h = img.size
    left = clamp_int(crop.left, 0, w)
    right = clamp_int(w - crop.right, 0, w)
    top = clamp_int(crop.top, 0, h)
    bottom = clamp_int(h - crop.bottom, 0, h)
    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop: got box=({left},{top},{right},{bottom}) for image size {(w, h)}"
        )
    return img.crop((left, top, right, bottom))


def split_into_slices(img: Image.Image, n: int) -> list[Image.Image]:
    if n <= 0:
        raise ValueError("n must be > 0")
    w, h = img.size
    base = w // n
    rem = w % n

    slices: list[Image.Image] = []
    x = 0
    for i in range(n):
        slice_w = base + (1 if i < rem else 0)
        slices.append(img.crop((x, 0, x + slice_w, h)))
        x += slice_w
    return slices


def draw_crop_box_preview(img: Image.Image, crop: Crop) -> Image.Image:
    preview = img.convert("RGBA")
    w, h = preview.size

    left = clamp_int(crop.left, 0, w)
    right = clamp_int(w - crop.right, 0, w)
    top = clamp_int(crop.top, 0, h)
    bottom = clamp_int(h - crop.bottom, 0, h)

    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    d.rectangle([0, 0, w, top], fill=(255, 0, 0, 60))
    d.rectangle([0, bottom, w, h], fill=(255, 0, 0, 60))
    d.rectangle([0, top, left, bottom], fill=(255, 0, 0, 60))
    d.rectangle([right, top, w, bottom], fill=(255, 0, 0, 60))

    d.rectangle([left, top, right - 1, bottom - 1], outline=(0, 255, 0, 220), width=2)

    return Image.alpha_composite(preview, overlay).convert("RGB")


def draw_slice_grid_preview(img: Image.Image, n: int) -> Image.Image:
    preview = img.convert("RGB").copy()
    w, h = preview.size
    d = ImageDraw.Draw(preview)

    base = w // n
    rem = w % n
    x = 0
    for i in range(n):
        slice_w = base + (1 if i < rem else 0)
        x_next = x + slice_w
        if i != 0:
            d.line([(x, 0), (x, h)], fill=(0, 255, 0), width=2)
        d.text((x + 3, 3), str(i + 1), fill=(0, 0, 0))
        d.text((x + 2, 2), str(i + 1), fill=(255, 255, 255))
        x = x_next
    d.rectangle([0, 0, w - 1, h - 1], outline=(0, 255, 0), width=2)
    return preview


def main() -> int:
    crop = Crop(
        top=CROP_TOP,
        bottom=CROP_BOTTOM,
        left=CROP_LEFT,
        right=CROP_RIGHT,
    )
    out_dir: Path = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(INPUT_IMAGE).convert("RGB")

    crop_box_preview = draw_crop_box_preview(img, crop)
    crop_box_preview_path = out_dir / "00_crop_box_preview.png"
    crop_box_preview.save(crop_box_preview_path)

    cropped = crop_image(img, crop)
    cropped_path = out_dir / "01_cropped.png"
    cropped.save(cropped_path)

    grid_preview = draw_slice_grid_preview(cropped, NUM_SLICES)
    grid_preview_path = out_dir / "02_slice_grid_preview.png"
    grid_preview.save(grid_preview_path)

    slices = split_into_slices(cropped, NUM_SLICES)
    for i, sl in enumerate(slices, start=1):
        sl.save(out_dir / f"slice_{i:02d}.png")

    print(f"Input:  {INPUT_IMAGE}  size={img.size}")
    print(f"Crop:   top={crop.top} bottom={crop.bottom} left={crop.left} right={crop.right}")
    print(f"Cropped size={cropped.size}")
    print(f"Saved:  {out_dir.resolve()}")
    print(f" - {crop_box_preview_path.name}")
    print(f" - {cropped_path.name}")
    print(f" - {grid_preview_path.name}")
    print(" - slice_01.png .. slice_%02d.png" % NUM_SLICES)

    if OPEN_PREVIEW:
        crop_box_preview.show(title="Crop box preview")
        grid_preview.show(title="Slice grid preview")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

