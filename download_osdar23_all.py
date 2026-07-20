from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests
from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    sync_playwright,
)


# =====================================================================
# НАСТРОЙКИ
# =====================================================================

PROJECT_DIR = Path(__file__).resolve().parent

# ZIP-архивы сохраняются сюда.
ARCHIVES_DIR = PROJECT_DIR / "data" / "archives"

# Распакованные данные сохраняются сюда.
RAW_DIR = PROJECT_DIR / "data" / "raw"

# Профиль Chrome с cookies сайта.
BROWSER_PROFILE_DIR = (
    PROJECT_DIR
    / "data"
    / ".osdar_browser_profile"
)

DATASET_PAGE_URL = (
    "https://data.fid-move.de/dataset/osdar23"
)

DOWNLOAD_HOST_URL = (
    "https://download.data.fid-move.de/"
)

DOWNLOAD_BASE_URL = (
    "https://download.data.fid-move.de/"
    "dzsf/osdar23"
)

# Используем только центральную цветную камеру 12 Мп.
CAMERA_FOLDER = "rgb_highres_center"

# Оставлять ZIP после распаковки.
# True: удалить ZIP.
# False: сохранить ZIP в data/archives.
DELETE_ARCHIVE_AFTER_EXTRACTION = False

# Размер блока при загрузке.
CHUNK_SIZE = 8 * 1024 * 1024

# Число попыток скачать одну последовательность.
MAX_RETRIES = 3

# Пауза между попытками.
RETRY_DELAY_SECONDS = 10


# =====================================================================
# ПОСЛЕДОВАТЕЛЬНОСТИ
# =====================================================================

# Калибровочные последовательности 1.1 и 1.2 исключены.
SEQUENCES = [
    "2_station_berliner_tor_2.1",

    "3_fire_site_3.1",
    "3_fire_site_3.2",
    "3_fire_site_3.3",
    "3_fire_site_3.4",

    "4_station_pedestrian_bridge_4.1",
    "4_station_pedestrian_bridge_4.2",
    "4_station_pedestrian_bridge_4.3",
    "4_station_pedestrian_bridge_4.4",
    "4_station_pedestrian_bridge_4.5",

    "5_station_bergedorf_5.1",
    "5_station_bergedorf_5.2",

    "6_station_klein_flottbek_6.1",
    "6_station_klein_flottbek_6.2",

    "7_approach_underground_station_7.1",
    "7_approach_underground_station_7.2",
    "7_approach_underground_station_7.3",

    "8_station_altona_8.1",
    "8_station_altona_8.2",
    "8_station_altona_8.3",

    "9_station_ruebenkamp_9.1",
    "9_station_ruebenkamp_9.2",
    "9_station_ruebenkamp_9.3",
    "9_station_ruebenkamp_9.4",
    "9_station_ruebenkamp_9.5",
    "9_station_ruebenkamp_9.6",
    "9_station_ruebenkamp_9.7",

    "10_station_suelldorf_10.1",
    "11_main_station_11.1",
    "12_vegetation_steady_12.1",
    "13_station_ohlsdorf_13.1",

    "14_signals_station_14.1",
    "14_signals_station_14.2",
    "14_signals_station_14.3",

    "15_construction_vehicle_15.1",

    "16_under_bridge_16.1",
    "17_signal_bridge_17.1",
    "18_vegetation_switch_18.1",
    "19_vegetation_curve_19.1",
    "20_vegetation_squirrel_20.1",

    "21_station_wedel_21.1",
    "21_station_wedel_21.2",
    "21_station_wedel_21.3",
]


# =====================================================================
# ИСКЛЮЧЕНИЯ
# =====================================================================

class AuthenticationRequiredError(RuntimeError):
    """
    Сервер вернул страницу проверки вместо ZIP.
    """


# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================

def format_size(size_bytes: int) -> str:
    """
    Преобразует байты в удобный формат.
    """

    size = float(size_bytes)

    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} ПБ"


def prepare_directories() -> None:
    """
    Создаёт рабочие папки.
    """

    ARCHIVES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BROWSER_PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def sequence_directory(sequence: str) -> Path:
    return RAW_DIR / sequence


def archive_path(sequence: str) -> Path:
    return ARCHIVES_DIR / f"{sequence}.zip"


def partial_archive_path(sequence: str) -> Path:
    return ARCHIVES_DIR / f"{sequence}.zip.part"


def labels_path(sequence: str) -> Path:
    return (
        sequence_directory(sequence)
        / f"{sequence}_labels.json"
    )


def camera_directory(sequence: str) -> Path:
    return (
        sequence_directory(sequence)
        / CAMERA_FOLDER
    )


def sequence_is_complete(sequence: str) -> bool:
    """
    Проверяет наличие JSON-разметки и изображений.
    """

    label_file = labels_path(sequence)
    image_dir = camera_directory(sequence)

    if not label_file.is_file():
        return False

    if not image_dir.is_dir():
        return False

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
    }

    return any(
        path.is_file()
        and path.suffix.lower() in image_extensions
        for path in image_dir.iterdir()
    )


def response_is_html(
    response: requests.Response,
) -> bool:
    """
    Проверяет, вернул ли сервер HTML вместо архива.
    """

    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    return "text/html" in content_type


def show_progress(
    downloaded: int,
    total_size: int,
) -> None:
    """
    Показывает прогресс скачивания.
    """

    if total_size > 0:
        percentage = downloaded / total_size * 100

        message = (
            f"\rСкачано: {format_size(downloaded)} / "
            f"{format_size(total_size)} "
            f"({percentage:.1f}%)"
        )
    else:
        message = (
            f"\rСкачано: {format_size(downloaded)}"
        )

    print(
        message,
        end="",
        flush=True,
    )


# =====================================================================
# АВТОРИЗАЦИЯ ЧЕРЕЗ CHROME
# =====================================================================

def wait_for_browser_check(
    description: str,
) -> None:
    """
    Ожидает ручного прохождения проверки сайта.
    """

    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)

    print(
        "В открывшемся Chrome дождись окончания "
        "проверки 'Making sure you're not a bot'."
    )

    print(
        "После появления обычной страницы "
        "вернись в терминал."
    )

    input("Нажми Enter после прохождения проверки: ")


def open_and_authenticate(
    context: BrowserContext,
) -> Page:
    """
    Открывает оба домена сайта для получения cookies.
    """

    pages = context.pages

    if pages:
        page = pages[0]
    else:
        page = context.new_page()

    page.set_default_timeout(120_000)
    page.set_default_navigation_timeout(120_000)

    print("\nОткрываем страницу OSDaR23...")

    try:
        page.goto(
            DATASET_PAGE_URL,
            wait_until="domcontentloaded",
            timeout=120_000,
        )
    except PlaywrightError as error:
        print(
            "Сообщение браузера при открытии страницы:"
        )
        print(error)

    wait_for_browser_check(
        "Проверка доступа к data.fid-move.de"
    )

    print("\nОткрываем сервер загрузки...")

    try:
        page.goto(
            DOWNLOAD_HOST_URL,
            wait_until="domcontentloaded",
            timeout=120_000,
        )
    except PlaywrightError as error:
        print(
            "Сообщение браузера при открытии "
            "сервера загрузки:"
        )
        print(error)

    wait_for_browser_check(
        "Проверка доступа к download.data.fid-move.de"
    )

    return page


def create_requests_session(
    context: BrowserContext,
    page: Page,
) -> requests.Session:
    """
    Переносит cookies и User-Agent из Chrome
    в requests.Session.
    """

    session = requests.Session()

    user_agent = page.evaluate(
        "navigator.userAgent"
    )

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": (
                "application/zip,"
                "application/octet-stream,"
                "*/*"
            ),
            "Accept-Language": (
                "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"
            ),
            "Referer": DATASET_PAGE_URL,
            "Connection": "keep-alive",
        }
    )

    cookies = context.cookies()

    for cookie in cookies:
        cookie_arguments = {
            "name": cookie["name"],
            "value": cookie["value"],
            "path": cookie.get("path", "/"),
        }

        domain = cookie.get("domain")

        if domain:
            cookie_arguments["domain"] = domain

        session.cookies.set(
            **cookie_arguments,
        )

    print(
        f"\nПолучено cookies из Chrome: "
        f"{len(cookies)}"
    )

    return session


# =====================================================================
# СКАЧИВАНИЕ
# =====================================================================

def download_archive(
    session: requests.Session,
    sequence: str,
) -> Path:
    """
    Скачивает ZIP с поддержкой продолжения.
    """

    url = f"{DOWNLOAD_BASE_URL}/{sequence}.zip"

    final_path = archive_path(sequence)
    partial_path = partial_archive_path(sequence)

    if final_path.is_file():
        if zipfile.is_zipfile(final_path):
            print(
                "ZIP уже скачан и прошёл проверку."
            )
            return final_path

        print(
            "Существующий ZIP повреждён. Удаляем."
        )
        final_path.unlink()

    existing_size = (
        partial_path.stat().st_size
        if partial_path.exists()
        else 0
    )

    headers: dict[str, str] = {}

    if existing_size > 0:
        headers["Range"] = (
            f"bytes={existing_size}-"
        )

        print(
            "Продолжаем загрузку с позиции "
            f"{format_size(existing_size)}."
        )
    else:
        print("Начинаем новую загрузку.")

    print(f"URL: {url}")
    print(f"Файл: {final_path}")

    try:
        response = session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(60, 600),
            allow_redirects=True,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"Ошибка подключения: {error}"
        ) from error

    with response:
        if response_is_html(response):
            preview = response.raw.read(
                800,
                decode_content=True,
            ).decode(
                "utf-8",
                errors="replace",
            )

            raise AuthenticationRequiredError(
                "Сервер вернул HTML вместо ZIP.\n"
                "Cookies проверки отсутствуют или истекли.\n\n"
                f"Начало ответа:\n{preview}"
            )

        if response.status_code not in {
            200,
            206,
        }:
            raise RuntimeError(
                f"HTTP {response.status_code}: "
                f"{response.reason}"
            )

        # Сервер проигнорировал Range.
        if (
            existing_size > 0
            and response.status_code != 206
        ):
            print(
                "Сервер не поддержал продолжение. "
                "Файл будет загружен заново."
            )

            existing_size = 0
            partial_path.unlink(
                missing_ok=True,
            )

        content_length = response.headers.get(
            "Content-Length"
        )

        if (
            content_length is not None
            and content_length.isdigit()
        ):
            total_size = (
                existing_size
                + int(content_length)
            )
        else:
            total_size = 0

        mode = (
            "ab"
            if existing_size > 0
            else "wb"
        )

        downloaded = existing_size

        try:
            with partial_path.open(mode) as file:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE,
                ):
                    if not chunk:
                        continue

                    file.write(chunk)
                    downloaded += len(chunk)

                    show_progress(
                        downloaded=downloaded,
                        total_size=total_size,
                    )

        except KeyboardInterrupt:
            print(
                "\nЗагрузка остановлена."
            )
            print(
                "Повторный запуск продолжит "
                "скачивание этого файла."
            )
            raise

    print()

    if not partial_path.is_file():
        raise RuntimeError(
            "Временный файл не был создан."
        )

    if not zipfile.is_zipfile(partial_path):
        with partial_path.open("rb") as file:
            preview_bytes = file.read(1000)

        preview = preview_bytes.decode(
            "utf-8",
            errors="replace",
        )

        invalid_path = (
            ARCHIVES_DIR
            / f"{sequence}.invalid"
        )

        invalid_path.unlink(
            missing_ok=True,
        )

        partial_path.replace(
            invalid_path,
        )

        raise RuntimeError(
            "Полученный файл не является ZIP.\n"
            f"Размер: "
            f"{format_size(invalid_path.stat().st_size)}\n"
            f"Файл сохранён как:\n{invalid_path}\n\n"
            f"Начало ответа:\n{preview}"
        )

    partial_path.replace(final_path)

    print("ZIP успешно загружен.")
    print(
        f"Размер: "
        f"{format_size(final_path.stat().st_size)}"
    )

    return final_path


# =====================================================================
# РАСПАКОВКА
# =====================================================================

def relative_archive_path(
    member_name: str,
    sequence: str,
) -> PurePosixPath | None:
    """
    Получает путь относительно папки последовательности.

    Поддерживает архивы вида:

    sequence/rgb_highres_center/image.png

    и архивы вида:

    rgb_highres_center/image.png
    """

    member_path = PurePosixPath(member_name)
    parts = member_path.parts

    if not parts:
        return None

    if sequence in parts:
        sequence_index = parts.index(sequence)
        relative_parts = parts[
            sequence_index + 1:
        ]
    else:
        relative_parts = parts

    if not relative_parts:
        return None

    return PurePosixPath(
        *relative_parts
    )


def member_is_required(
    relative_path: PurePosixPath,
    sequence: str,
) -> bool:
    """
    Выбирает нужные файлы из архива.
    """

    parts = relative_path.parts

    if not parts:
        return False

    # Все изображения центральной 12-Мп камеры.
    if parts[0] == CAMERA_FOLDER:
        return True

    filename = relative_path.name.lower()

    required_files = {
        f"{sequence.lower()}_labels.json",
        "readme.md",
        "license.md",
    }

    return filename in required_files


def extract_required_files(
    zip_path: Path,
    sequence: str,
) -> None:
    """
    Извлекает только нужные изображения и JSON.
    """

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(
            f"Файл не является ZIP:\n{zip_path}"
        )

    output_directory = sequence_directory(
        sequence
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_root = output_directory.resolve()

    extracted_files = 0
    extracted_images = 0

    print(
        "\nИзвлекаем:"
        f"\n  {CAMERA_FOLDER}"
        "\n  JSON-разметку"
        "\n  README"
        "\n  лицензию"
    )

    with zipfile.ZipFile(
        zip_path,
        mode="r",
    ) as archive:
        members = archive.infolist()

        selected_members: list[
            tuple[
                zipfile.ZipInfo,
                PurePosixPath,
            ]
        ] = []

        for member in members:
            relative_path = relative_archive_path(
                member.filename,
                sequence,
            )

            if relative_path is None:
                continue

            if not member_is_required(
                relative_path,
                sequence,
            ):
                continue

            selected_members.append(
                (
                    member,
                    relative_path,
                )
            )

        if not selected_members:
            raise RuntimeError(
                "В архиве не найдены нужные файлы.\n"
                f"Ожидалась папка {CAMERA_FOLDER} "
                "и JSON-разметка."
            )

        total = len(selected_members)

        for index, (
            member,
            relative_path,
        ) in enumerate(
            selected_members,
            start=1,
        ):
            target_path = (
                output_directory
                / Path(*relative_path.parts)
            )

            resolved_target = target_path.resolve()

            try:
                resolved_target.relative_to(
                    output_root
                )
            except ValueError as error:
                raise RuntimeError(
                    "Обнаружен опасный путь "
                    f"в ZIP: {member.filename}"
                ) from error

            if member.is_dir():
                target_path.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                continue

            target_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(
                member,
                mode="r",
            ) as source_file:
                with target_path.open(
                    mode="wb",
                ) as destination_file:
                    shutil.copyfileobj(
                        source_file,
                        destination_file,
                        length=CHUNK_SIZE,
                    )

            extracted_files += 1

            if (
                relative_path.parts[0]
                == CAMERA_FOLDER
            ):
                extracted_images += 1

            print(
                f"\rРаспаковано: "
                f"{index}/{total}",
                end="",
                flush=True,
            )

    print()

    print(
        f"Извлечено файлов: {extracted_files}"
    )

    print(
        f"Извлечено изображений: "
        f"{extracted_images}"
    )

    if not sequence_is_complete(sequence):
        raise RuntimeError(
            "Распаковка завершена, но обязательные "
            "данные не найдены.\n"
            f"Проверь папку:\n{output_directory}"
        )

    print(
        "Последовательность успешно подготовлена."
    )

    print(
        f"Папка:\n{output_directory}"
    )


# =====================================================================
# ОБРАБОТКА ОДНОЙ ПОСЛЕДОВАТЕЛЬНОСТИ
# =====================================================================

def process_sequence(
    session: requests.Session,
    sequence: str,
) -> None:
    """
    Загружает и распаковывает одну последовательность.
    """

    if sequence_is_complete(sequence):
        print(
            "Данные уже распакованы. Пропускаем."
        )
        return

    zip_path = download_archive(
        session=session,
        sequence=sequence,
    )

    extract_required_files(
        zip_path=zip_path,
        sequence=sequence,
    )

    if DELETE_ARCHIVE_AFTER_EXTRACTION:
        zip_path.unlink(
            missing_ok=True,
        )

        print(
            "ZIP удалён после успешной распаковки."
        )


# =====================================================================
# ОСНОВНОЙ ЦИКЛ
# =====================================================================

def main() -> int:
    prepare_directories()

    print("=" * 70)
    print("ЗАГРУЗКА OSDaR23")
    print("=" * 70)

    print(
        f"Всего последовательностей: "
        f"{len(SEQUENCES)}"
    )

    print(
        f"ZIP будут сохранены в:\n"
        f"{ARCHIVES_DIR}"
    )

    print(
        f"\nРаспакованные данные будут сохранены в:\n"
        f"{RAW_DIR}"
    )

    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    with sync_playwright() as playwright:
        print(
            "\nЗапускаем установленный Google Chrome..."
        )

        try:
            context = (
                playwright.chromium
                .launch_persistent_context(
                    user_data_dir=str(
                        BROWSER_PROFILE_DIR
                    ),
                    channel="chrome",
                    headless=False,
                    accept_downloads=True,
                    viewport={
                        "width": 1280,
                        "height": 900,
                    },
                )
            )
        except PlaywrightError as error:
            print(
                "\nНе удалось запустить Chrome."
            )

            print(
                "Убедись, что Google Chrome установлен."
            )

            print(error)
            return 1

        try:
            page = open_and_authenticate(
                context
            )

            session = create_requests_session(
                context=context,
                page=page,
            )

            for index, sequence in enumerate(
                SEQUENCES,
                start=1,
            ):
                print("\n" + "#" * 70)
                print(
                    f"[{index}/{len(SEQUENCES)}] "
                    f"{sequence}"
                )
                print("#" * 70)

                if sequence_is_complete(sequence):
                    print(
                        "Последовательность уже готова."
                    )
                    skipped.append(sequence)
                    continue

                success = False

                for attempt in range(
                    1,
                    MAX_RETRIES + 1,
                ):
                    print(
                        f"\nПопытка "
                        f"{attempt}/{MAX_RETRIES}"
                    )

                    try:
                        process_sequence(
                            session=session,
                            sequence=sequence,
                        )

                        completed.append(sequence)
                        success = True
                        break

                    except AuthenticationRequiredError as error:
                        print(
                            "\nТребуется повторная "
                            "проверка браузера."
                        )

                        print(error)

                        session.close()

                        page = open_and_authenticate(
                            context
                        )

                        session = (
                            create_requests_session(
                                context=context,
                                page=page,
                            )
                        )

                    except KeyboardInterrupt:
                        print(
                            "\nРабота остановлена пользователем."
                        )

                        print(
                            "Повторный запуск продолжит "
                            "незавершённую загрузку."
                        )

                        session.close()
                        return 130

                    except Exception as error:
                        print(
                            f"\nОшибка обработки "
                            f"{sequence}:"
                        )

                        print(error)

                    if attempt < MAX_RETRIES:
                        print(
                            f"\nПовтор через "
                            f"{RETRY_DELAY_SECONDS} секунд..."
                        )

                        time.sleep(
                            RETRY_DELAY_SECONDS
                        )

                if not success:
                    failed.append(sequence)

            session.close()

        finally:
            context.close()

    print("\n" + "=" * 70)
    print("ИТОГ")
    print("=" * 70)

    print(
        f"Загружено сейчас: {len(completed)}"
    )

    print(
        f"Уже было готово: {len(skipped)}"
    )

    print(
        f"Ошибок: {len(failed)}"
    )

    if failed:
        print(
            "\nНе удалось обработать:"
        )

        for sequence in failed:
            print(f"  {sequence}")

        print(
            "\nПовторный запуск попробует "
            "скачать только недостающие данные."
        )

        return 1

    print(
        "\nВсе последовательности подготовлены."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())