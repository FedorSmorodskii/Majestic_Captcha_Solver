import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Глобальные настройки обучения по умолчанию
EPOCHS_DEFAULT = 10
BATCH_SIZE_DEFAULT = 32

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "digits"
MODEL_PATH = BASE_DIR / "digit_model.keras"


def build_model(input_shape=(64, 64, 3), num_classes=9) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=input_shape),

            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets(
    data_dir: Path,
    img_size=(64, 64),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 123,
):
    # Сначала создаём датасеты, считываем class_names,
    # а уже потом навешиваем cache/prefetch, чтобы не потерять атрибут.
    train_ds_raw = image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
    )

    val_ds_raw = image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
    )

    class_names = train_ds_raw.class_names

    # Performance optimizations
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def train(
    epochs: int = EPOCHS_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
    img_size=(64, 64),
):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Папка с данными не найдена: {DATA_DIR}")

    train_ds, val_ds, class_names = load_datasets(
        DATA_DIR,
        img_size=img_size,
        batch_size=batch_size,
    )

    num_classes = len(class_names)
    model = build_model(input_shape=(*img_size, 3), num_classes=num_classes)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    model.save(MODEL_PATH)
    print(f"Модель сохранена в: {MODEL_PATH}")
    print(f"Классы (папки): {class_names}")
    print("Важно: цифра = int(class_name). Например, класс '1' -> цифра 1.")


def predict_image(image_path: Path, img_size=(64, 64)):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}. Сначала запустите обучение."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.utils.load_img(
        image_path,
        target_size=img_size,
        color_mode="rgb",
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

    predictions = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions[0]))

    # Здесь предполагаем, что имена папок — строки "1", "2", ..., "9"
    # Поэтому можно восстановить цифру как (predicted_class_index + 1),
    # если классы отсортированы по имени.
    # Для надежности загрузим class_names из датасета.
    _, _, class_names = load_datasets(DATA_DIR, img_size=img_size, batch_size=32)
    predicted_class_name = class_names[predicted_class_index]

    predicted_digit = int(predicted_class_name)
    confidence = float(np.max(predictions[0]))

    print(f"Путь к картинке: {image_path}")
    print(f"Предсказанная цифра: {predicted_digit}")
    print(f"Класс (имя папки): {predicted_class_name}")
    print(f"Уверенность: {confidence:.4f}")

    return predicted_digit, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Нейронка для распознавания цифр (1–9) из капчи."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Команда train
    train_parser = subparsers.add_parser("train", help="Обучить модель на папке digits")
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT,
        help=f"Количество эпох обучения (по умолчанию {EPOCHS_DEFAULT})",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"Размер батча (по умолчанию {BATCH_SIZE_DEFAULT})",
    )

    # Команда predict
    predict_parser = subparsers.add_parser(
        "predict", help="Предсказать цифру по одной картинке"
    )
    predict_parser.add_argument(
        "image_path",
        type=str,
        help="Путь к картинке с одной цифрой из капчи",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(epochs=args.epochs, batch_size=args.batch_size)
    elif args.command == "predict":
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Картинка не найдена: {image_path}")
        predict_image(image_path)


if __name__ == "__main__":
    # Не занимать всю GPU-память, если есть GPU
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    main()

