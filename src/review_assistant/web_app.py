from __future__ import annotations

import shutil
import time
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.review_assistant.config import DEFAULT_CONFIG, load_config, resolve_path
from src.review_assistant.database import ReviewDatabase
from src.review_assistant.processor import ReviewProcessor
from src.review_assistant.reporting import generate_report


st.set_page_config(
    page_title="Railway Person Review Assistant",
    page_icon="🚆",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.stApp { background: #f2f5f7; }
[data-testid="stSidebar"] { background: #112535; color: white; }
.product-banner {
  background: linear-gradient(110deg,#102838,#1c5367);
  border-left: 7px solid #efa51f; color: white; padding: 1.1rem 1.4rem;
  border-radius: 8px; margin-bottom: 1rem;
}
.safety-contract {
  background: #fff3cd; border: 1px solid #d99b14; color: #4d3a0a;
  padding: .8rem 1rem; border-radius: 7px; margin-bottom: 1rem;
}
.event-source { display:inline-block; padding:.2rem .55rem; border-radius:12px;
  font-size:.78rem; font-weight:700; background:#dbe8ff; }
.temporal-warning { background:#fff0e0; color:#8c4100; padding:.6rem;
  border-left:4px solid #e57c00; margin:.5rem 0; }
</style>
"""


def _hotkeys() -> None:
    components.html(
        """
        <script>
        const doc = window.parent.document;
        function clickLabel(label) {
          [...doc.querySelectorAll('button')].find(
            b => b.innerText.trim().startsWith(label)
          )?.click();
        }
        doc.onkeydown = (event) => {
          if (['INPUT','TEXTAREA'].includes(doc.activeElement?.tagName)) return;
          if (event.key === '1') clickLabel('ЧЕЛОВЕК');
          if (event.key === '2') clickLabel('ЛОЖНОЕ');
          if (event.key === '3') clickLabel('НЕ УВЕРЕН');
          if (event.key.toLowerCase() === 'n') clickLabel('СЛЕДУЮЩЕЕ');
          if (event.key.toLowerCase() === 'p') clickLabel('ПРЕДЫДУЩЕЕ');
          if (event.code === 'Space') {
            event.preventDefault();
            const video = doc.querySelector('video');
            if (video) video.paused ? video.play() : video.pause();
          }
        };
        </script>
        """,
        height=0,
    )


def _video_files(path: Path, config: dict, recursive: bool) -> list[Path]:
    extensions = {str(value).lower() for value in config["input"]["extensions"]}
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(item for item in iterator if item.suffix.lower() in extensions)


def _sidebar(config: dict) -> tuple[str, str]:
    st.sidebar.markdown("## Railway Review")
    st.sidebar.success("OPERATOR_ASSISTANT_MVP")
    st.sidebar.markdown(
        """
        **Autonomous alarming:** DISABLED<br>
        **Safety actuation:** DISABLED<br>
        **Human confirmation:** REQUIRED
        """
    )
    operator = st.sidebar.text_input(
        "Оператор", value=config["review"]["default_operator"]
    )
    database = st.sidebar.text_input(
        "SQLite", value=str(config["storage"]["database"])
    )
    st.sidebar.caption("Railway test остаётся SEALED · access count 0")
    return operator.strip() or "local-operator", database


def _processing_tab(config: dict, database_path: str) -> None:
    st.subheader("Обработка видео")
    st.caption(
        "Offline processing разрешён. Если pipeline медленнее исходного FPS, "
        "интерфейс показывает прогресс и ETA без заявления real-time."
    )
    source_mode = st.radio(
        "Источник", ["Локальный путь", "Загрузить файл"], horizontal=True
    )
    selected_path: Path | None = None
    if source_mode == "Локальный путь":
        raw_path = st.text_input("Видео или папка")
        if raw_path:
            selected_path = Path(raw_path).expanduser()
    else:
        upload = st.file_uploader(
            "Видео", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=False
        )
        if upload:
            inbox = resolve_path("data/review_assistant_inbox")
            inbox.mkdir(parents=True, exist_ok=True)
            safe_name = Path(upload.name).name
            selected_path = inbox / safe_name
            with selected_path.open("wb") as handle:
                shutil.copyfileobj(upload, handle)
    mode = st.selectbox(
        "Режим",
        ["combined_queue", "conservative_review", "high_recall_review"],
        format_func=lambda value: {
            "combined_queue": "Combined queue — основной",
            "conservative_review": "Conservative review",
            "high_recall_review": "High-recall review",
        }[value],
    )
    recursive = st.checkbox("Рекурсивно просматривать папку", value=False)
    camera_id = st.text_input("Camera ID", value=config["input"]["camera_id"])
    checkpoint = st.text_input(
        "Frozen detector checkpoint",
        value=os.environ.get("REVIEW_ASSISTANT_CHECKPOINT", ""),
    )
    verifier = st.text_input(
        "Frozen combined verifier",
        value=os.environ.get("REVIEW_ASSISTANT_VERIFIER", ""),
    )
    encoder = st.text_input(
        "Frozen verifier encoder",
        value=os.environ.get("REVIEW_ASSISTANT_ENCODER", ""),
    )
    if st.button("НАЧАТЬ ОБРАБОТКУ", type="primary", use_container_width=True):
        if selected_path is None or not selected_path.exists():
            st.error("Укажите существующий видеофайл или папку.")
            return
        files = _video_files(selected_path, config, recursive)
        if not files:
            st.error("Поддерживаемые видеофайлы не найдены.")
            return
        overall = st.progress(0.0)
        status = st.status("Инициализация frozen pipeline…", expanded=True)
        metrics = st.empty()
        errors: list[str] = []
        for file_index, video in enumerate(files):
            status.write(f"Обработка: {video.name}")

            def callback(payload: dict) -> None:
                if payload.get("error"):
                    return
                file_progress = float(payload.get("progress", 0.0))
                overall.progress((file_index + file_progress) / len(files))
                metrics.info(
                    f"Кадры: {payload['processed_frames']}/"
                    f"{payload['total_frames']} · "
                    f"{payload['fps']:.2f} FPS · "
                    f"ETA {payload['eta_seconds']:.0f} с · "
                    f"События: {payload['events']}"
                )

            try:
                processor = ReviewProcessor(
                    checkpoint=checkpoint or None,
                    verifier=verifier or None,
                    encoder=encoder or None,
                    database=database_path,
                )
                summary = processor.process(
                    video, mode=mode, camera_id=camera_id, progress=callback
                )
                st.session_state["selected_run_id"] = summary["run_id"]
                status.write(
                    f"{video.name}: {summary['unique_events']} уникальных событий"
                )
            except Exception as error:
                errors.append(f"{video.name}: {error}")
                status.error(errors[-1])
        overall.progress(1.0)
        if errors:
            status.update(label="Завершено с ошибками", state="error")
        else:
            status.update(label="Очередь файлов обработана", state="complete")


def _review_action(
    database_path: str,
    run_id: str,
    event_id: str,
    operator: str,
    status: str,
    comment: str,
) -> None:
    started = st.session_state.get("review_card_started", time.perf_counter())
    with ReviewDatabase(database_path) as database:
        database.review_event(
            run_id,
            event_id,
            operator=operator,
            new_status=status,
            comment=comment,
        )
        database.add_review_seconds(run_id, time.perf_counter() - started)
    st.session_state["review_card_started"] = time.perf_counter()


def _event_queue(
    config: dict,
    database_path: str,
    operator: str,
    *,
    muted: bool,
) -> None:
    with ReviewDatabase(database_path) as database:
        runs = database.list_runs()
    if not runs:
        st.info("Сначала обработайте видео.")
        return
    labels = {
        row["run_id"]: f"{row['run_id']} · {Path(row['path']).name} · {row['status']}"
        for row in runs
    }
    preferred = st.session_state.get("selected_run_id")
    index = (
        list(labels).index(preferred)
        if preferred in labels
        else 0
    )
    run_id = st.selectbox(
        "Запуск",
        list(labels),
        index=index,
        format_func=lambda key: labels[key],
        key=f"run-select-{muted}",
    )
    st.session_state["selected_run_id"] = run_id
    with ReviewDatabase(database_path) as database:
        events = database.list_events(run_id, muted=muted)
    if not events:
        st.info("В этой очереди событий нет.")
        return
    position_key = f"event-position-{run_id}-{muted}"
    position = min(int(st.session_state.get(position_key, 0)), len(events) - 1)
    event = events[position]
    st.session_state.setdefault("review_card_started", time.perf_counter())
    _hotkeys()
    st.markdown(
        f"<span class='event-source'>{event['source_label']}</span>",
        unsafe_allow_html=True,
    )
    if event["source_label"] == "TEMPORAL_ONLY":
        st.markdown(
            "<div class='temporal-warning'>Temporal-only: повышенный риск "
            "ложного срабатывания. Требуется ручная проверка.</div>",
            unsafe_allow_html=True,
        )
    left, right = st.columns([3, 2])
    with left:
        media_available = False
        if event["thumbnail_path"] and Path(event["thumbnail_path"]).is_file():
            st.image(event["thumbnail_path"], caption="Ключевой кадр события")
            media_available = True
        if event["clip_path"] and Path(event["clip_path"]).is_file():
            st.video(event["clip_path"])
            media_available = True
        if not media_available:
            st.warning("Клип события недоступен.")
    with right:
        st.metric("Событие", event["event_id"])
        st.write(
            f"Время: **{event['start_time']:.1f}–{event['end_time']:.1f} с**  \n"
            f"Длительность: **{event['duration']:.1f} с**  \n"
            f"Max confidence: **{event['maximum_confidence']:.3f}**  \n"
            f"Detector frames: **{event['real_detection_count']}**  \n"
            f"Interpolated: **{event['interpolated_count']}**  \n"
            f"Статус: **{event['review_status']}**"
        )
        comment = st.text_area(
            "Комментарий",
            value=event["operator_comment"],
            key=f"comment-{run_id}-{event['event_id']}",
        )
        row1 = st.columns(3)
        if row1[0].button("ЧЕЛОВЕК · 1", use_container_width=True):
            _review_action(
                database_path, run_id, event["event_id"], operator, "HUMAN", comment
            )
            st.rerun()
        if row1[1].button("ЛОЖНОЕ · 2", use_container_width=True):
            _review_action(
                database_path,
                run_id,
                event["event_id"],
                operator,
                "FALSE_POSITIVE",
                comment,
            )
            st.rerun()
        if row1[2].button("НЕ УВЕРЕН · 3", use_container_width=True):
            _review_action(
                database_path,
                run_id,
                event["event_id"],
                operator,
                "UNCERTAIN",
                comment,
            )
            st.rerun()
        row2 = st.columns(4)
        if row2[0].button("ПРОПУСТИТЬ", use_container_width=True):
            _review_action(
                database_path,
                run_id,
                event["event_id"],
                operator,
                "SKIPPED",
                comment,
            )
            st.rerun()
        if row2[1].button("ОТМЕНИТЬ РЕШЕНИЕ", use_container_width=True):
            with ReviewDatabase(database_path) as database:
                database.undo_last_review(run_id, event["event_id"], operator)
            st.rerun()
        if row2[2].button("ПРЕДЫДУЩЕЕ · P", use_container_width=True):
            st.session_state[position_key] = max(position - 1, 0)
            st.session_state["review_card_started"] = time.perf_counter()
            st.rerun()
        if row2[3].button("СЛЕДУЮЩЕЕ · N", use_container_width=True):
            st.session_state[position_key] = min(position + 1, len(events) - 1)
            st.session_state["review_card_started"] = time.perf_counter()
            st.rerun()
    if event["review_status"] == "FALSE_POSITIVE" and not muted:
        with ReviewDatabase(database_path) as database:
            count = database.suppression_suggestion_count(
                event["camera_id"],
                event["spatial_region"],
                float(config["suppression"]["region_iou"]),
            )
        minimum = int(config["suppression"]["minimum_false_events_for_suggestion"])
        if count >= minimum:
            st.warning(
                f"В этой области подтверждено {count} ложных событий. "
                "Можно вручную объединять будущие повторения во вкладке Muted."
            )
            confirmed = st.checkbox(
                "Я подтверждаю camera-specific suppression rule",
                key=f"suppress-confirm-{run_id}-{event['event_id']}",
            )
            if st.button("СОЗДАТЬ ПРАВИЛО ПРИГЛУШЕНИЯ", disabled=not confirmed):
                with ReviewDatabase(database_path) as database:
                    database.create_suppression_rule(
                        camera_id=event["camera_id"],
                        region=event["spatial_region"],
                        reference_confidence=event["maximum_confidence"],
                        reference_motion=event["representative_motion"],
                        operator=operator,
                        reason=f"Repeated false events: {count}",
                        manual_confirmed=confirmed,
                    )
                    run = database.get_run(run_id)
                    database.apply_suppression(
                        run_id,
                        event["event_id"],
                        config["suppression"],
                        frame_width=int(run["width"]),
                        frame_height=int(run["height"]),
                    )
                st.success("Правило создано. События остаются доступными в Muted.")
                st.rerun()
    st.caption(
        f"{position + 1}/{len(events)} · Space: play/pause · "
        "1/2/3: решение · N/P: навигация"
    )


def _reports_tab(config: dict, database_path: str) -> None:
    with ReviewDatabase(database_path) as database:
        runs = database.list_runs()
    if not runs:
        st.info("Нет завершённых запусков.")
        return
    labels = {
        row["run_id"]: f"{row['run_id']} · {Path(row['path']).name}" for row in runs
    }
    run_id = st.selectbox(
        "Запуск для отчёта",
        list(labels),
        format_func=lambda key: labels[key],
        key="report-run",
    )
    if st.button("СФОРМИРОВАТЬ HTML/PDF И CSV", type="primary"):
        metrics = generate_report(
            database_path,
            run_id,
            resolve_path(config["storage"]["output_root"]),
            resolved_config={
                key: value for key, value in config.items() if not key.startswith("_")
            },
        )
        st.success("Отчёт сформирован.")
        st.json(metrics)
    root = resolve_path(config["storage"]["output_root"]) / run_id
    if (root / "REVIEW_REPORT.html").is_file():
        st.markdown(f"Отчёт: `{root / 'REVIEW_REPORT.html'}`")


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    config = load_config(DEFAULT_CONFIG)
    st.markdown(
        """
        <div class="product-banner">
        <h1>Railway Person Review Assistant</h1>
        <p>Локальная очередь уникальных видео-событий для ручной проверки.</p>
        </div>
        <div class="safety-contract"><strong>Не автономная система безопасности.</strong>
        Автоматические тревоги и safety actuation отключены. Каждое событие
        требует решения оператора.</div>
        """,
        unsafe_allow_html=True,
    )
    if os.environ.get("REVIEW_ASSISTANT_DATABASE"):
        config["storage"]["database"] = os.environ["REVIEW_ASSISTANT_DATABASE"]
    operator, database_path = _sidebar(config)
    tabs = st.tabs(["Обработка", "Очередь", "Muted", "Отчёты"])
    with tabs[0]:
        _processing_tab(config, database_path)
    with tabs[1]:
        _event_queue(config, database_path, operator, muted=False)
    with tabs[2]:
        _event_queue(config, database_path, operator, muted=True)
    with tabs[3]:
        _reports_tab(config, database_path)


if __name__ == "__main__":
    main()
