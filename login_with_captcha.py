from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tensorflow.keras.models import load_model
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

from digit_classifier import MODEL_PATH


LOGIN_URL = "https://majestic.com/account/login"
EMAIL = ""
PASSWORD = ""

# Параметры обрезки / нарезки такие же, как в crop_experiment.py и label_digits.py
CROP_TOP: int = 18
CROP_BOTTOM: int = 18
CROP_LEFT: int = 46
CROP_RIGHT: int = 50
NUM_SLICES: int = 6

# Куда сохранять неудачные попытки
FAILED_DIR = Path("failed_captchas")


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


def create_driver(headless: bool = False) -> webdriver.Chrome:
    options = ChromeOptions()
    # Не ждать полной загрузки тяжёлых страниц — сами ждём только нужные элементы.
    options.page_load_strategy = "none"
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,720")

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def load_digit_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}. Сначала обучите модель digit_classifier.py."
        )
    model = load_model(MODEL_PATH)
    return model


def predict_digits(model, digit_images: List[Image.Image]) -> str:
    if not digit_images:
        raise ValueError("digit_images is empty")

    img_size = (64, 64)
    batch = []
    for img in digit_images:
        img_resized = img.convert("RGB").resize(img_size)
        arr = np.array(img_resized, dtype=np.float32)
        batch.append(arr)

    batch_array = np.stack(batch, axis=0)
    predictions = model.predict(batch_array)

    # В digit_classifier модель обучена на классах '1'..'9', поэтому индекс + 1 = цифра.
    digit_indices = np.argmax(predictions, axis=1).astype(int)
    digits = [str(idx + 1) for idx in digit_indices]
    return "".join(digits)


def solve_captcha(driver, model) -> tuple[str, Image.Image]:
    """
    Возвращает (код_нейронки, исходная_картинка_капчи).
    """
    wait = WebDriverWait(driver, 10)

    captcha_element = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "img[alt='captcha']"))
    )

    png_bytes = captcha_element.screenshot_as_png

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    crop = Crop(
        top=CROP_TOP,
        bottom=CROP_BOTTOM,
        left=CROP_LEFT,
        right=CROP_RIGHT,
    )
    cropped = crop_image(img, crop)
    digit_images = split_into_slices(cropped, NUM_SLICES)

    code = predict_digits(model, digit_images)
    print(f"Распознанная капча: {code}")
    return code, img


def main() -> None:
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    model = load_digit_model()
    driver = create_driver(headless=False)
    wait = WebDriverWait(driver, 15)

    try:
        attempt = 0

        while True:
            attempt += 1
            print(f"\n=== Попытка #{attempt} ===")
            driver.get(LOGIN_URL)

            email_input = wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='email']"))
            )
            password_input = wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='password']"))
            )

            email_input.clear()
            email_input.send_keys(EMAIL)

            password_input.clear()
            password_input.send_keys(PASSWORD)

            captcha_code, captcha_img = solve_captcha(driver, model)

            captcha_input = wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@name='Captcha']"))
            )
            captcha_input.clear()
            captcha_input.send_keys(captcha_code)

            login_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//input[@class='mob-block-input']"))
            )
            login_button.click()

            # Ждём 3 секунды и проверяем, остались ли на странице логина.
            time.sleep(3.0)
            current_url = driver.current_url
            print(f"Текущий URL после попытки: {current_url}")

            if current_url.startswith(LOGIN_URL):
                # Возможный случай "too many active sessions" — считаем попытку успешной.
                too_many_elems = driver.find_elements(
                    By.XPATH, "//h2[text()='This account has too many active sessions']"
                )
                if too_many_elems:
                    print(
                        "Сообщение 'This account has too many active sessions' найдено. "
                        "Считаем капчу решённой и переходим к следующему входу."
                    )
                    driver.get(LOGIN_URL)
                    continue

                # Обычная неудача: капчу не прошли — сохраняем капчу и ответ нейронки.
                filename = FAILED_DIR / f"attempt_{attempt:04d}_{captcha_code}.png"
                captcha_img.save(filename)
                print(f"Неудача, капча сохранена: {filename}")

                # Переходим к следующей попытке (страница уже логина, просто продолжаем цикл).
                continue
            else:
                # Капчу прошли — можно анализировать, но по задаче просто перезагружаем браузер
                # (обновляем страницу логина) и идём на следующую попытку.
                print("Капча пройдена, переходим к следующему входу.")
                driver.get(LOGIN_URL)

    finally:
        input("Скрипт закончил работу цикла (или вы его остановили). Нажмите Enter, чтобы закрыть браузер...")
        driver.quit()


if __name__ == "__main__":
    main()

