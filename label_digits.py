from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageTk
import tkinter as tk
import os


# === НАСТРОЙКИ ОБРЕЗКИ / НАРЕЗКИ ===

INPUT_DIR: Path = Path("captchas")          # откуда брать сырые капчи
OUTPUT_DIR: Path = Path("digits")          # сюда будут складываться папки 0..9

# Параметры обрезки (как ты подобрал)
CROP_TOP: int = 18
CROP_BOTTOM: int = 18
CROP_LEFT: int = 46
CROP_RIGHT: int = 50

# Сколько цифр в одной капче
NUM_SLICES: int = 6


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


def split_into_slices(img: Image.Image, n: int) -> List[Image.Image]:
    if n <= 0:
        raise ValueError("n must be > 0")
    w, h = img.size
    base = w // n
    rem = w % n

    slices: List[Image.Image] = []
    x = 0
    for i in range(n):
        slice_w = base + (1 if i < rem else 0)
        slices.append(img.crop((x, 0, x + slice_w, h)))
        x += slice_w
    return slices


def build_dataset() -> list[Tuple[Path, int]]:
    """
    Возвращает список (путь к исходной капче, индекс среза 0..NUM_SLICES-1)
    в порядке обхода файлов и срезов.
    """
    items: list[Tuple[Path, int]] = []
    for img_path in sorted(INPUT_DIR.glob("*.png")):
        for slice_idx in range(NUM_SLICES):
            items.append((img_path, slice_idx))
    return items


class LabelApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Captcha digit labeling")

        # Подготовка выходных папок 0..9
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for d in range(10):
            (OUTPUT_DIR / str(d)).mkdir(parents=True, exist_ok=True)

        self.crop = Crop(
            top=CROP_TOP,
            bottom=CROP_BOTTOM,
            left=CROP_LEFT,
            right=CROP_RIGHT,
        )

        self.items = build_dataset()
        self.index = 0  # текущий индекс в self.items

        # history: список (путь_сохранённого_файла, индекс_в_dataset)
        self.history: list[Tuple[Path, int]] = []

        # GUI элементы
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.info_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.info_label.pack()

        # сюда нужно сохранить ссылку на объект изображения, иначе Tk его "съест"
        self._tk_image = None

        # биндим клавиши
        self.root.bind("<Key>", self.on_key)
        self.root.bind("<Left>", self.on_left)
        self.root.bind("<Right>", self.on_right)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        if not self.items:
            self.info_label.config(text=f"Нет файлов в {INPUT_DIR}")
        else:
            self.show_current()

    # === Навигация по датасету ===

    def show_current(self) -> None:
        if not self.items:
            return
        img_path, slice_idx = self.items[self.index]

        img = Image.open(img_path).convert("RGB")
        cropped = crop_image(img, self.crop)
        slices = split_into_slices(cropped, NUM_SLICES)
        digit_img = slices[slice_idx]

        # Масштабируем, если слишком большая/маленькая (чисто для удобства)
        max_h = 160
        w, h = digit_img.size
        if h < max_h:
            scale = max_h / h
            digit_img = digit_img.resize((int(w * scale), int(h * scale)), Image.NEAREST)

        self._tk_image = ImageTk.PhotoImage(digit_img)
        self.image_label.config(image=self._tk_image)

        self.info_label.config(
            text=(
                f"Файл: {img_path.name}  "
                f"срез: {slice_idx + 1}/{NUM_SLICES}  "
                f"{self.index + 1}/{len(self.items)}\n"
                f"Цифра 0-9 = сохранить, "
                f"→ = пропустить, ← = отменить последнюю метку, Esc = выход"
            )
        )

    def next_item(self) -> None:
        if self.index + 1 < len(self.items):
            self.index += 1
            self.show_current()
        else:
            self.info_label.config(text="Готово, больше изображений нет.")
            self.image_label.config(image="")

    # === Обработка клавиш ===

    def on_key(self, event: tk.Event) -> None:
        ch = event.char
        if ch and ch.isdigit() and len(ch) == 1:
            self.save_current_digit(int(ch))

    def on_left(self, event: tk.Event) -> None:
        self.undo_last()

    def on_right(self, event: tk.Event) -> None:
        self.next_item()

    # === Логика сохранения / отката ===

    def save_current_digit(self, digit: int) -> None:
        if not (0 <= digit <= 9):
            return
        if not self.items:
            return

        img_path, slice_idx = self.items[self.index]

        img = Image.open(img_path).convert("RGB")
        cropped = crop_image(img, self.crop)
        slices = split_into_slices(cropped, NUM_SLICES)
        digit_img = slices[slice_idx]

        out_dir = OUTPUT_DIR / str(digit)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{img_path.stem}_s{slice_idx}.png"
        out_path = out_dir / out_name

        # если что‑то с таким именем уже есть — не перетираем, а добавляем суффикс
        counter = 1
        final_path = out_path
        while final_path.exists():
            final_path = out_dir / f"{img_path.stem}_s{slice_idx}_{counter}.png"
            counter += 1

        digit_img.save(final_path)

        # записываем в историю для возможного Undo
        self.history.append((final_path, self.index))

        self.next_item()

    def undo_last(self) -> None:
        if not self.history:
            return
        last_path, last_index = self.history.pop()

        try:
            if last_path.exists():
                os.remove(last_path)
        except OSError:
            pass

        self.index = last_index
        self.show_current()


def main() -> int:
    root = tk.Tk()
    app = LabelApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

