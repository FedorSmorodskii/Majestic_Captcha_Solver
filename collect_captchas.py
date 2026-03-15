import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions


LOGIN_URL = "https://majestic.com/account/login"
OUTPUT_DIR = Path("captchas")
NUM_CAPTCHAS = 300
TIMEOUT = 0.5
SLEEP_BETWEEN_REFRESH = 0.1  # небольшая пауза, чтобы не спамить


def create_driver(headless: bool = False) -> webdriver.Chrome:
    """Инициализация ChromeDriver с webdriver-manager."""
    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,720")

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_captchas():
    ensure_output_dir(OUTPUT_DIR)

    driver = create_driver(headless=False)
    wait = WebDriverWait(driver, TIMEOUT)

    try:
        driver.get(LOGIN_URL)

        last_src = None

        for i in range(NUM_CAPTCHAS):
            # Ждём появления элемента капчи
            captcha_element = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img[alt='captcha']"))
            )

            # Иногда src может обновляться не сразу после refresh — подождём, пока картинка прогрузится
            wait.until(lambda d: captcha_element.get_attribute("src") is not None)

            src = captcha_element.get_attribute("src")

            # Имя файла
            filename = OUTPUT_DIR / f"captcha_{i:03d}.png"

            # Сохраняем скриншот именно элемента капчи
            captcha_element.screenshot(str(filename))
            print(f"[{i + 1}/{NUM_CAPTCHAS}] Saved {filename} (src={src})")

            # Обновляем страницу, чтобы получить новую капчу
            last_src = src
            time.sleep(SLEEP_BETWEEN_REFRESH)
            driver.refresh()

            # Ждём, пока src у новой капчи (или сам элемент) обновится, чтобы уменьшить шанс дубликатов
            try:
                wait.until(
                    lambda d: d.find_element(By.CSS_SELECTOR, "img[alt='captcha']").get_attribute("src") != last_src
                )
            except Exception:
                # Если src не поменялся по таймауту — всё равно идём дальше, просто логируем
                print(f"Warning: captcha src may not have changed on iteration {i + 1}")

    finally:
        driver.quit()


if __name__ == "__main__":
    collect_captchas()

