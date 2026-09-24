from __future__ import annotations

import os
import subprocess
import threading
import traceback
import json
import shutil
import queue
from io import BytesIO
import base64
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..application import ApplicationService, SUPPORTED_EXTENSIONS
from ..config import DetectorConfig, app_data_dir, load_config, save_config
from ..geometry import order_points


class CoreBoxCropperApp:
    """Windows GUI for single-photo preview and batch processing."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Core Box Cropper")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)
        self.config = load_config()
        self.service = ApplicationService(self.config)

        self.input_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value=str(Path.home() / "CoreBoxCropper_Result"))
        self.preview_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Выберите папку или несколько фотографий.")
        self.selection_var = tk.StringVar(value="Файлов выбрано: 0")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.selected_paths: list[Path] = []
        self.current_index = -1
        self.current_image = None
        self.current_detection = None
        self.preview_photo = None
        self.manual_editor = None
        self.error_dialog = None
        self.processing = False
        self._detect_token = 0
        self.stats = {"processed": 0, "success": 0, "failed": 0}
        self._ui_queue = queue.Queue()

        # Poll background results from the Tk main thread. Tkinter widgets and
        # PhotoImage objects must only be touched from the main thread.
        self.root.report_callback_exception = self._tk_callback_exception
        self._build()
        self.root.after(40, self._drain_ui_queue)

    @property
    def log_dir(self) -> Path:
        directory = app_data_dir() / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _write_gui_error(self, text: str) -> None:
        try:
            path = self.log_dir / "gui_errors.log"
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + "=" * 90 + "\n" + text + "\n")
        except Exception:
            pass

    def _tk_callback_exception(self, exc_type, exc_value, exc_traceback) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self._write_gui_error(text)
        self._show_error_dialog("Ошибка Tkinter", text)

    def _build(self) -> None:
        self.root.configure(bg="#f3f5f7")
        header = tk.Frame(self.root, bg="#18324b", height=68)
        header.pack(fill="x")
        tk.Label(
            header, text="Core Box Cropper", fg="white", bg="#18324b",
            font=("Segoe UI", 20, "bold")
        ).pack(anchor="w", padx=24, pady=16)

        body = tk.Frame(self.root, bg="#f3f5f7")
        body.pack(fill="both", expand=True, padx=20, pady=16)
        controls = tk.Frame(body, bg="#f3f5f7", width=405)
        controls.pack(side="left", fill="y", padx=(0, 18))
        controls.pack_propagate(False)
        preview = tk.Frame(body, bg="#152432")
        preview.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(preview, bg="#152432", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        tk.Label(
            controls, text="Исходные фотографии", bg="#f3f5f7", fg="#18324b",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        tk.Label(
            controls, textvariable=self.input_var, bg="#f3f5f7", fg="#51616d",
            justify="left", wraplength=390, font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(4, 8))

        buttons = tk.Frame(controls, bg="#f3f5f7")
        buttons.pack(fill="x", pady=(0, 8))
        tk.Button(
            buttons, text="Выбрать папку", command=self._choose_folder,
            bg="#dbe3e9", fg="#243443", relief="flat", padx=8, pady=7
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(
            buttons, text="Выбрать файлы", command=self._choose_files,
            bg="#dbe3e9", fg="#243443", relief="flat", padx=8, pady=7
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        tk.Button(
            controls, text="Очистить выбор", command=self._clear_selection,
            bg="#eef1f3", fg="#51616d", relief="flat", padx=8, pady=5
        ).pack(fill="x", pady=(0, 8))
        tk.Label(
            controls, textvariable=self.selection_var, bg="#f3f5f7", fg="#18324b",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")

        list_frame = tk.Frame(controls, bg="#f3f5f7")
        list_frame.pack(fill="x", pady=(5, 5))
        self.file_list = tk.Listbox(
            list_frame, height=7, font=("Segoe UI", 9), relief="solid", bd=1,
            exportselection=False
        )
        self.file_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)
        self.file_list.bind("<Double-Button-1>", lambda _event: self._detect())

        nav = tk.Frame(controls, bg="#f3f5f7")
        nav.pack(fill="x", pady=(0, 10))
        self.prev_button = tk.Button(
            nav, text="◀ Предыдущее", command=lambda: self._navigate(-1),
            bg="#dbe3e9", fg="#243443", relief="flat", padx=7, pady=5
        )
        self.prev_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.next_button = tk.Button(
            nav, text="Следующее ▶", command=lambda: self._navigate(1),
            bg="#dbe3e9", fg="#243443", relief="flat", padx=7, pady=5
        )
        self.next_button.pack(side="left", fill="x", expand=True, padx=(3, 0))

        self._path_row(controls, "Папка результата", self.output_var, self._choose_output, "Выбрать")
        ttk.Checkbutton(
            controls, text="Включать вложенные папки", variable=self.recursive_var
        ).pack(anchor="w", pady=(0, 5))
        ttk.Checkbutton(
            controls, text="Показывать предпросмотр", variable=self.preview_var
        ).pack(anchor="w", pady=(0, 5))
        ttk.Checkbutton(
            controls, text="Перезаписывать существующие результаты", variable=self.overwrite_var
        ).pack(anchor="w", pady=(0, 12))

        self.detect_button = tk.Button(
            controls, text="Проверить выбранное фото", command=self._detect,
            bg="#1f78b4", fg="white", relief="flat", font=("Segoe UI", 10, "bold"),
            padx=10, pady=9
        )
        self.detect_button.pack(fill="x", pady=(0, 7))
        self.process_button = tk.Button(
            controls, text="Обработать все выбранные", command=self._process_all,
            state="disabled", bg="#2c9b68", fg="white", relief="flat",
            font=("Segoe UI", 10, "bold"), padx=10, pady=9
        )
        self.process_button.pack(fill="x", pady=(0, 7))
        self.manual_button = tk.Button(
            controls, text="Указать область вручную", command=self._manual,
            state="disabled", bg="#e4a13a", fg="#1d2730", relief="flat",
            padx=10, pady=8
        )
        self.manual_button.pack(fill="x", pady=(0, 7))
        self.ml_button = tk.Button(
            controls, text="Разметка ML — 4 точки", command=self._ml_annotation,
            state="disabled", bg="#8b6fb3", fg="white", relief="flat",
            font=("Segoe UI", 9, "bold"), padx=10, pady=8
        )
        self.ml_button.pack(fill="x", pady=(0, 12))

        lower = tk.Frame(controls, bg="#f3f5f7")
        lower.pack(fill="x")
        tk.Button(
            lower, text="Настройки алгоритма", command=self._settings,
            bg="#dbe3e9", fg="#243443", relief="flat", padx=10, pady=7
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        tk.Button(
            lower, text="Ошибки / журнал", command=self._show_logs,
            bg="#f0d4d4", fg="#7a2020", relief="flat", padx=10, pady=7
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))
        tk.Button(
            controls, text="Открыть папку результата", command=self._open_output_folder,
            bg="#dbe3e9", fg="#243443", relief="flat", padx=10, pady=7
        ).pack(fill="x", pady=(6, 0))

        ttk.Separator(controls).pack(fill="x", pady=14)
        tk.Label(
            controls, text="Статус", bg="#f3f5f7", fg="#18324b",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")
        tk.Label(
            controls, textvariable=self.status_var, bg="#f3f5f7", fg="#51616d",
            justify="left", wraplength=390, font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(5, 8))
        ttk.Progressbar(controls, variable=self.progress_var, maximum=100).pack(fill="x", pady=(0, 8))
        self.stats_label = tk.Label(
            controls, text=self._stats_text(), bg="#f3f5f7", fg="#51616d",
            justify="left", font=("Segoe UI", 9)
        )
        self.stats_label.pack(anchor="w")
        self._update_navigation()

    def _path_row(self, parent, label, variable, command, button_label) -> None:
        tk.Label(
            parent, text=label, bg="#f3f5f7", fg="#43525d",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(0, 4))
        row = tk.Frame(parent, bg="#f3f5f7")
        row.pack(fill="x", pady=(0, 10))
        tk.Entry(
            row, textvariable=variable, relief="solid", bd=1,
            font=("Segoe UI", 9)
        ).pack(side="left", fill="x", expand=True, ipady=4)
        tk.Button(
            row, text=button_label, command=command,
            bg="#dbe3e9", relief="flat", padx=8, pady=5
        ).pack(side="right", padx=(6, 0))

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с фотографиями")
        if not folder:
            return
        root = Path(folder)
        pattern = "**/*" if self.recursive_var.get() else "*"
        files = sorted(
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            messagebox.showwarning("Нет фотографий", "В выбранной папке не найдено JPG, PNG или WEBP изображений.")
            return
        self.selected_paths = files
        self.input_var.set(f"Папка: {root}")
        self._refresh_file_list(select_index=0)
        self._set_default_output(root)
        self._detect_current()

    def _choose_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Выберите фотографии",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.webp"), ("Все файлы", "*.*")],
        )
        if not files:
            return
        self.selected_paths = sorted({Path(p) for p in files}, key=lambda p: str(p).lower())
        self.input_var.set("Выбрано несколько фотографий")
        self._refresh_file_list(select_index=0)
        self._set_default_output(self.selected_paths[0].parent)
        self._detect_current()

    def _set_default_output(self, parent: Path) -> None:
        if not self.output_var.get() or self.output_var.get().endswith("CoreBoxCropper_Result"):
            self.output_var.set(str(parent / "Result"))

    def _refresh_file_list(self, select_index: int | None = None) -> None:
        self.file_list.delete(0, tk.END)
        for path in self.selected_paths[:500]:
            self.file_list.insert(tk.END, path.name)
        extra = len(self.selected_paths) - min(len(self.selected_paths), 500)
        self.selection_var.set(
            f"Файлов выбрано: {len(self.selected_paths)}" + (f" (+{extra})" if extra else "")
        )
        if self.selected_paths:
            if select_index is None:
                select_index = min(max(self.current_index, 0), len(self.selected_paths) - 1)
            self.current_index = select_index
            if select_index < 500:
                self.file_list.selection_clear(0, tk.END)
                self.file_list.selection_set(select_index)
                self.file_list.see(select_index)
        else:
            self.current_index = -1
        enabled = bool(self.selected_paths) and not self.processing
        self.detect_button.configure(state="normal" if enabled else "disabled")
        self.process_button.configure(state="normal" if enabled else "disabled")
        self.manual_button.configure(state="normal" if self.current_image is not None and enabled else "disabled")
        self.ml_button.configure(state="normal" if self.selected_paths and not self.processing else "disabled")
        self._update_navigation()

    def _on_file_select(self, _event=None) -> None:
        if self.processing:
            return
        selected = self.file_list.curselection()
        if not selected:
            return
        index = int(selected[0])
        if index >= len(self.selected_paths):
            return
        self.current_index = index
        self._update_navigation()
        self._detect_current()

    def _navigate(self, step: int) -> None:
        if not self.selected_paths or self.processing:
            return
        index = min(max(self.current_index + step, 0), len(self.selected_paths) - 1)
        self.current_index = index
        if index < 500:
            self.file_list.selection_clear(0, tk.END)
            self.file_list.selection_set(index)
            self.file_list.see(index)
        self._update_navigation()
        self._detect_current()

    def _update_navigation(self) -> None:
        has_previous = self.current_index > 0 and not self.processing
        has_next = 0 <= self.current_index < len(self.selected_paths) - 1 and not self.processing
        self.prev_button.configure(state="normal" if has_previous else "disabled")
        self.next_button.configure(state="normal" if has_next else "disabled")

    def _clear_selection(self) -> None:
        self._detect_token += 1
        self.selected_paths = []
        self.current_index = -1
        self.current_image = None
        self.current_detection = None
        self.input_var.set("")
        self._refresh_file_list()
        self.status_var.set("Выбор очищен.")
        self.canvas.delete("all")

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Папка для результатов")
        if path:
            self.output_var.set(path)

    def _current_source(self) -> Path | None:
        if not self.selected_paths:
            return None
        if self.current_index < 0 or self.current_index >= len(self.selected_paths):
            self.current_index = 0
        return self.selected_paths[self.current_index]

    def _detect(self) -> None:
        if not self.selected_paths:
            messagebox.showwarning("Нет фотографий", "Выберите папку или несколько фотографий.")
            return
        self._detect_current()

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                try:
                    kind, payload = self._ui_queue.get_nowait()
                except queue.Empty:
                    break

                if kind == "preview":
                    image, source = payload
                    current = self._current_source()
                    if current == source and not self.processing:
                        self.current_image = image
                        self.current_detection = None
                        if self.preview_var.get():
                            self._draw_preview_image(image)
                        self.manual_button.configure(state="normal")
                elif kind == "detection":
                    image, detection, source, token = payload
                    self._show_detection(image, detection, source, token)
                elif kind == "error":
                    text = payload
                    self._show_error(text)
        except queue.Empty:
            pass
        except Exception:
            text = traceback.format_exc()
            self._write_gui_error(text)
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(40, self._drain_ui_queue)
            except Exception:
                pass

    def _draw_preview_image(self, image) -> None:
        frame = image.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        self.canvas.update_idletasks()
        max_w = max(self.canvas.winfo_width() - 20, 100)
        max_h = max(self.canvas.winfo_height() - 20, 100)
        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.preview_photo,
        )

    def _detect_current(self) -> None:
        source = self._current_source()
        if source is None or self.processing:
            return
        self._detect_token += 1
        token = self._detect_token
        self.detect_button.configure(state="disabled")
        self.process_button.configure(state="disabled")
        self.manual_button.configure(state="disabled")
        self.status_var.set(f"Анализ: {source.name}")
        threading.Thread(target=self._detect_worker, args=(source, token), daemon=True).start()

    def _detect_worker(self, source: Path, token: int) -> None:
        try:
            image = self.service.load_image(source)
            self._ui_queue.put(("preview", (image, source)))
            detection = self.service.detector.detect(image)
            self._ui_queue.put(("detection", (image, detection, source, token)))
        except BaseException:
            error_text = traceback.format_exc()
            self._write_gui_error(error_text)
            self._ui_queue.put(("error", error_text))

    def _show_detection(self, image, detection, source: Path, token: int) -> None:
        if token != self._detect_token:
            return
        current = self._current_source()
        if current is None or current != source:
            return
        self.current_image, self.current_detection = image, detection
        enabled = bool(self.selected_paths) and not self.processing
        self.detect_button.configure(state="normal" if enabled else "disabled")
        self.process_button.configure(state="normal" if enabled else "disabled")
        self.manual_button.configure(state="normal" if enabled else "disabled")
        self.ml_button.configure(state="normal" if enabled else "disabled")
        if detection.success:
            ruler = "линейка + цветовая шкала найдены" if detection.ruler else "линейка не подтверждена"
            angle = int(detection.candidate_scores.get("orientation_degrees", 0))
            orientation = "без поворота" if angle == 0 else f"анализ с поворотом {angle}°"
            self.status_var.set(
                f"{source.name}\nЯщик: 3 ячейки; уверенность: {detection.confidence:.0%}; {ruler}.\n"
                f"{orientation}. Предпросмотр: контур объекта; итоговый crop использует расширенный ROI."
            )
        else:
            self.status_var.set(
                f"{source.name}\nНе удалось надёжно определить область: {detection.reason}"
            )
        if self.preview_var.get():
            self._draw_preview(image, detection)

    def _show_error(self, text: str) -> None:
        enabled = bool(self.selected_paths) and not self.processing
        self.detect_button.configure(state="normal" if enabled else "disabled")
        self.process_button.configure(state="normal" if enabled else "disabled")
        self._update_navigation()
        self._show_error_dialog("Ошибка анализа", text)
        self.status_var.set("Ошибка анализа. Подробности доступны в «Ошибки / журнал».")

    def _show_error_dialog(self, title: str, text: str) -> None:
        if self.error_dialog and self.error_dialog.winfo_exists():
            self.error_dialog.focus_force()
            self.error_dialog.set_text(text)
            return
        self.error_dialog = ErrorLogDialog(self.root, initial_text=text, title=title)

    def _show_logs(self) -> None:
        parts: list[str] = []
        gui_log = self.log_dir / "gui_errors.log"
        process_log = self.log_dir / "coreboxcropper.log"
        for title, path in (("ОШИБКИ GUI", gui_log), ("ЖУРНАЛ ОБРАБОТКИ", process_log)):
            try:
                content = path.read_text(encoding="utf-8") if path.exists() else "Файл пока не создан."
            except Exception as exc:
                content = f"Не удалось прочитать {path}: {exc}"
            parts.append(f"===== {title} =====\n{content}")
        self._show_error_dialog("Ошибки / журнал", "\n\n".join(parts))

    def _process_all(self) -> None:
        if not self.selected_paths or self.processing:
            return
        output_dir = Path(self.output_var.get()) if self.output_var.get() else self.selected_paths[0].parent / "Result"
        overwrite = bool(self.overwrite_var.get())
        self.processing = True
        self.stats = {"processed": 0, "success": 0, "failed": 0}
        self.progress_var.set(0)
        self.detect_button.configure(state="disabled")
        self.process_button.configure(state="disabled")
        self.manual_button.configure(state="disabled")
        self.ml_button.configure(state="disabled")
        self._update_navigation()
        paths = list(self.selected_paths)
        threading.Thread(target=self._batch_worker, args=(paths, output_dir, overwrite), daemon=True).start()

    def _batch_worker(self, paths: list[Path], output_dir: Path, overwrite: bool) -> None:
        total = len(paths)
        def on_result(result):
            self.root.after(0, lambda r=result: self._batch_result(r, total))
        try:
            results = self.service.process_paths(
                paths, output_dir, overwrite=overwrite, on_result=on_result
            )
            self.root.after(0, lambda: self._batch_finished(results))
        except BaseException:
            error_text = traceback.format_exc()
            self._write_gui_error(error_text)
            self.root.after(0, lambda text=error_text: self._batch_failed(text))

    def _batch_result(self, result, total: int) -> None:
        self.stats["processed"] += 1
        if result.status == "success":
            self.stats["success"] += 1
        else:
            self.stats["failed"] += 1
        self.progress_var.set(self.stats["processed"] / max(total, 1) * 100)
        self.stats_label.configure(text=self._stats_text())
        self.status_var.set(
            f"Обработано {self.stats['processed']} из {total}: {Path(result.input_path).name}\n{result.reason}"
        )

    def _batch_finished(self, results) -> None:
        self.processing = False
        self._refresh_file_list()
        self.progress_var.set(100 if results else 0)
        self.status_var.set(
            f"Готово. Успешно: {sum(r.status == 'success' for r in results)}; "
            f"ошибок: {sum(r.status != 'success' for r in results)}."
        )

    def _batch_failed(self, text: str) -> None:
        self.processing = False
        self._refresh_file_list()
        self.status_var.set("Массовая обработка остановлена из-за ошибки.")
        self._show_error_dialog("Ошибка массовой обработки", text)

    def _ml_annotation(self) -> None:
        if not self.selected_paths or self.processing:
            return
        if self.manual_editor and self.manual_editor.winfo_exists():
            self.manual_editor.focus_force()
            return
        self.manual_editor = MLAnnotationEditor(
            self.root, list(self.selected_paths), self.current_index, self.config,
            initial_detection=self.current_detection if self.current_image is not None else None,
            on_close=lambda: setattr(self, "manual_editor", None),
        )

    def _manual(self) -> None:
        source = self._current_source()
        if source is None:
            return
        if self.current_image is None or self.current_detection is None:
            self._detect_current()
            return
        if self.manual_editor and self.manual_editor.winfo_exists():
            self.manual_editor.focus_force()
            return
        self.manual_editor = ManualEditor(
            self.root, self.current_image,
            lambda corners: self._manual_done(source, corners)
        )

    def _manual_done(self, source: Path, corners) -> None:
        output_dir = Path(self.output_var.get()) if self.output_var.get() else source.parent / "Result"
        output = output_dir / source.name
        result = self.service.save_manual(source, output, corners)
        self.status_var.set(
            f"Сохранено вручную: {result.output_path}"
            if result.status == "success" else f"Ошибка: {result.reason}"
        )

    def _draw_preview(self, image, detection) -> None:
        frame = image.copy()
        thickness = max(3, frame.shape[1] // 500)
        if detection.corners:
            points = np.asarray(detection.corners, dtype=np.int32).reshape(-1, 2)
            if points.shape[0] == 4:
                cv2.polylines(frame, [points], True, (0, 255, 0), thickness)
        if detection.bbox:
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), max(1, thickness // 2))
        if detection.ruler:
            x1, y1, x2, y2 = map(int, detection.ruler)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), thickness)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        self.canvas.update_idletasks()
        max_w = max(self.canvas.winfo_width() - 20, 100)
        max_h = max(self.canvas.winfo_height() - 20, 100)
        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.preview_photo,
        )

    def _settings(self) -> None:
        SettingsDialog(self.root, self.config, self._apply_settings)

    def _apply_settings(self, config: DetectorConfig) -> None:
        self.config = config
        save_config(config)
        self.service = ApplicationService(config)
        self.current_image = None
        self.current_detection = None
        self.status_var.set("Настройки сохранены. Предпросмотр будет рассчитан заново.")

    def _open_output_folder(self) -> None:
        directory = Path(self.output_var.get()) if self.output_var.get() else Path.home()
        directory.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(directory)
        else:
            subprocess.run(["xdg-open", str(directory)], check=False)

    def _stats_text(self) -> str:
        return (
            f"Обработано: {self.stats['processed']}\n"
            f"Успешно: {self.stats['success']}\n"
            f"Не удалось: {self.stats['failed']}"
        )

    def run(self) -> None:
        self.root.mainloop()


class ErrorLogDialog(tk.Toplevel):
    """Copy-friendly error and processing log window."""

    def __init__(self, parent, initial_text: str = "", title: str = "Ошибки / журнал", on_clear=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x620")
        self.minsize(650, 420)
        self.transient(parent)
        self.on_clear = on_clear
        self.text = tk.Text(self, wrap="none", font=("Consolas", 9), undo=False)
        self.text.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y", padx=(0, 10), pady=10)
        xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        xscroll.pack(side="bottom", fill="x", padx=10)
        self.text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        buttons = tk.Frame(self, bg="#f3f5f7")
        buttons.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(
            buttons, text="Копировать всё", command=self._copy_all,
            bg="#1f78b4", fg="white", relief="flat", padx=14, pady=7
        ).pack(side="left")
        tk.Button(
            buttons, text="Очистить журналы", command=self._clear_logs,
            bg="#e4a13a", fg="#1d2730", relief="flat", padx=14, pady=7
        ).pack(side="left", padx=6)
        tk.Button(
            buttons, text="Закрыть", command=self.destroy,
            bg="#dbe3e9", fg="#243443", relief="flat", padx=14, pady=7
        ).pack(side="right")
        self.set_text(initial_text)

    def set_text(self, text: str) -> None:
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text or "Журнал пуст.")
        self.text.see("1.0")

    def _copy_all(self) -> None:
        text = self.text.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

    def _clear_logs(self) -> None:
        if not messagebox.askyesno(
            "Очистить журналы",
            "Удалить журнал ошибок GUI и журнал обработки?",
            parent=self,
        ):
            return
        log_dir = app_data_dir() / "logs"
        errors = log_dir / "gui_errors.log"
        process = log_dir / "coreboxcropper.log"
        for path in (errors, process):
            try:
                if path.exists():
                    path.write_text("", encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Ошибка очистки", f"Не удалось очистить {path.name}:\n{exc}", parent=self)
                return
        if self.on_clear:
            self.on_clear()
        self.set_text("Журналы очищены.")



class MLAnnotationEditor(tk.Toplevel):
    """Four-point annotation editor with fast zoom/pan for large images.

    The annotation is an ordered quadrilateral:
    1 = top-left, 2 = top-right, 3 = bottom-right, 4 = bottom-left.

    ``annotations.json`` stores the exact four points. For compatibility with
    the current YOLO detector, a tight axis-aligned bbox is also written to the
    classic ``labels/<stem>.txt`` file.
    """

    CLASS_NAME = "target_corebox"
    POINT_NAMES = ("P1", "P2", "P3", "P4")

    def __init__(self, parent, paths, start_index, config, initial_detection=None, on_close=None):
        super().__init__(parent)
        self.title("Разметка данных для ML — 4 точки")
        self.geometry("1280x900")
        self.minsize(1000, 720)
        self.transient(parent)
        self.paths = list(paths)
        self.index = min(max(int(start_index), 0), len(self.paths) - 1) if self.paths else 0
        self.config = config
        self.initial_detection = initial_detection
        self.on_close = on_close

        self.image = None
        self.pil_image = None
        self.photo = None
        self.display_photo = None
        self.scale = 1.0
        self.fit_scale = 1.0
        self.view_zoom = 1.0
        self.view_center = (0.0, 0.0)
        self.crop_left = 0.0
        self.crop_top = 0.0
        self.render_offset = (0, 0)
        self.points: list[tuple[float, float]] | None = None
        self.active_point = None
        self.last_loaded = None
        self._render_pending = False
        self._render_after_id = None
        self._load_token = 0
        self._pan_anchor = None
        self._frame_drag_anchor = None
        self._load_queue = queue.Queue()
        self._error_messages: list[str] = []
        self._image_item = None
        self._photo_cache_key = None
        self._right_pan_anchor = None
        self.status = tk.StringVar(value="")
        self.zoom_var = tk.StringVar(value="100%")
        self.dataset_var = tk.StringVar(value=str(Path.home() / "CoreBoxCropper_ML_Dataset"))
        self.show_grid_var = tk.BooleanVar(value=False)
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        # Give Tk a chance to realize the canvas before the first render.
        self.after(80, self._load_current)
        self.after(40, self._poll_load_queue)

    def _build(self):
        self.configure(bg="#f3f5f7")
        top = tk.Frame(self, bg="#18324b")
        top.pack(fill="x")
        tk.Label(top, text="Разметка данных для ML — 4 точки", fg="white", bg="#18324b",
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(top, text="target_corebox", fg="white", bg="#18324b",
                 font=("Segoe UI", 9)).pack(side="right", padx=16)

        controls = tk.Frame(self, bg="#f3f5f7")
        controls.pack(fill="x", padx=12, pady=8)
        tk.Label(controls, text="Папка датасета:", bg="#f3f5f7", fg="#43525d").pack(side="left")
        tk.Entry(controls, textvariable=self.dataset_var, width=48).pack(side="left", padx=6, ipady=3)
        tk.Button(controls, text="Выбрать", command=self._choose_dataset,
                  bg="#dbe3e9", relief="flat", padx=10, pady=5).pack(side="left")

        zoom = tk.Frame(controls, bg="#f3f5f7")
        zoom.pack(side="right")
        tk.Button(zoom, text="−", command=lambda: self._zoom_step(1 / 1.25),
                  bg="#dbe3e9", relief="flat", width=3, pady=4).pack(side="left", padx=2)
        tk.Button(zoom, textvariable=self.zoom_var, command=self._fit_view,
                  bg="#dbe3e9", relief="flat", width=7, pady=4).pack(side="left", padx=2)
        tk.Button(zoom, text="+", command=lambda: self._zoom_step(1.25),
                  bg="#dbe3e9", relief="flat", width=3, pady=4).pack(side="left", padx=2)
        tk.Button(zoom, text="1:1", command=self._one_to_one,
                  bg="#dbe3e9", relief="flat", padx=8, pady=4).pack(side="left", padx=2)
        ttk.Checkbutton(zoom, text="Сетка", variable=self.show_grid_var,
                        command=self._request_render).pack(side="left", padx=8)
        tk.Button(controls, text="Ошибки / журнал", command=self._show_errors,
                  bg="#f0c36a", fg="#243443", relief="flat", padx=10, pady=5).pack(
                      side="left", padx=(8, 0)
                  )

        self.canvas = tk.Canvas(self, bg="#152432", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<ButtonPress-2>", self._pan_press)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._pan_release)
        self.canvas.bind("<ButtonPress-3>", self._pan_press)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._pan_release)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)
        self.canvas.bind("<Button-4>", lambda e: self._wheel_zoom_linux(1, e))
        self.canvas.bind("<Button-5>", lambda e: self._wheel_zoom_linux(-1, e))
        self.canvas.bind("<Configure>", lambda _e: self._request_render())
        self.bind("<KeyPress-plus>", lambda _e: self._zoom_step(1.25))
        self.bind("<KeyPress-minus>", lambda _e: self._zoom_step(1 / 1.25))
        self.bind("<KeyPress-0>", lambda _e: self._fit_view())
        self.bind("<KeyPress-1>", lambda _e: self._one_to_one())
        self.bind("<KeyPress-space>", lambda _e: self.canvas.configure(cursor="fleur"))
        self.focus_set()

        help_text = (
            "Мышь: колесо — масштаб, правая или средняя кнопка — панорамирование. "
            "Клавиши 0 — вписать, 1 — 1:1. Маленькие маркеры не закрывают границу; "
            "точку можно перетаскивать, а за линию внутри рамки — двигать всю рамку. "
            "После очистки щёлкайте по углам по порядку: точки соединяются автоматически. "
            "Порядок: P1=ЛВ, P2=ПВ, P3=ПН, P4=ЛН."
        )
        tk.Label(self, text=help_text, bg="#f3f5f7", fg="#51616d",
                 justify="left", wraplength=1200).pack(fill="x", padx=12)

        nav = tk.Frame(self, bg="#f3f5f7")
        nav.pack(fill="x", padx=12, pady=8)
        tk.Button(nav, text="◀ Предыдущее", command=lambda: self._move(-1),
                  bg="#dbe3e9", relief="flat", padx=12, pady=7).pack(side="left")
        self.file_label = tk.Label(nav, text="", bg="#f3f5f7", fg="#18324b",
                                   font=("Segoe UI", 9, "bold"))
        self.file_label.pack(side="left", fill="x", expand=True, padx=10)
        tk.Button(nav, text="Следующее ▶", command=lambda: self._move(1),
                  bg="#dbe3e9", relief="flat", padx=12, pady=7).pack(side="right")

        bottom = tk.Frame(self, bg="#f3f5f7")
        bottom.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(bottom, text="Сохранить текущую", command=self._save_current,
                  bg="#2c9b68", fg="white", relief="flat", padx=14, pady=8).pack(side="left")
        tk.Button(bottom, text="Сохранить и следующая", command=self._save_next,
                  bg="#1f78b4", fg="white", relief="flat", padx=14, pady=8).pack(side="left", padx=6)
        tk.Button(bottom, text="Взять автоматическую рамку", command=self._reset_from_detection,
                  bg="#dbe3e9", fg="#243443", relief="flat", padx=14, pady=8).pack(side="left", padx=6)
        tk.Button(bottom, text="Очистить точки", command=self._clear_points,
                  bg="#e4a13a", fg="#1d2730", relief="flat", padx=14, pady=8).pack(side="left")
        tk.Label(bottom, textvariable=self.status, bg="#f3f5f7", fg="#51616d").pack(side="left", padx=14)
        tk.Button(bottom, text="Закрыть", command=self._close,
                  bg="#dbe3e9", relief="flat", padx=14, pady=8).pack(side="right")

    def _choose_dataset(self):
        path = filedialog.askdirectory(title="Выберите папку для ML-датасета")
        if path:
            self.dataset_var.set(path)
            self._load_current()

    def _dataset_dirs(self):
        root = Path(self.dataset_var.get()).expanduser()
        images = root / "images"
        labels = root / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (root / "classes.txt").write_text(self.CLASS_NAME + "\n", encoding="utf-8")
        return root, images, labels

    def _poll_load_queue(self):
        try:
            while True:
                try:
                    item = self._load_queue.get_nowait()
                except queue.Empty:
                    break
                token, path, image, error = item
                self._finish_load(token, path, image, error)
        except Exception:
            text = traceback.format_exc()
            self.logger.exception("Error while polling image load queue")
            self._record_error(text)
        finally:
            if self.winfo_exists():
                self.after(40, self._poll_load_queue)

    def _load_current(self):
        if not self.paths:
            return
        path = self.paths[self.index]
        self._load_token += 1
        token = self._load_token
        self.image = None
        self.pil_image = None
        self.photo = None
        self._photo_cache_key = None
        self.points = None
        self.active_point = None
        self.last_loaded = path
        self._fit_view_state()
        self.file_label.configure(text=f"Загрузка {self.index + 1}/{len(self.paths)} — {path.name}")
        self.status.set("Загрузка изображения…")
        self._request_render()

        def worker():
            try:
                # Use PIL first: on Windows it is more reliable with long/Unicode
                # paths and avoids some cv2 decoding failures for unusual JPEGs.
                with Image.open(path) as pil:
                    pil_rgb = pil.convert("RGB").copy()
                image = cv2.cvtColor(np.asarray(pil_rgb), cv2.COLOR_RGB2BGR)
                if image is None or image.size == 0:
                    raise ValueError("Не удалось открыть изображение")
                self._load_queue.put((token, path, image, None))
            except Exception as exc:
                # Fallback to OpenCV for files PIL cannot decode.
                try:
                    data = np.fromfile(str(path), dtype=np.uint8)
                    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Не удалось открыть изображение")
                    self._load_queue.put((token, path, image, None))
                except Exception as fallback_exc:
                    self._load_queue.put((token, path, None, fallback_exc))

        threading.Thread(target=worker, name="ml-image-loader", daemon=True).start()

    def _finish_load(self, token, path, image, error):
        if token != self._load_token or self.last_loaded != path:
            return
        if error is not None:
            self.image = None
            self.points = None
            self._photo_cache_key = None
            self._record_error(f"Не удалось открыть изображение: {path}\n{error}")
            self.status.set(f"Ошибка открытия: {error}")
            self._request_render()
            return
        self.image = image
        self._photo_cache_key = None
        self.pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.points = self._load_existing_points(path)
        if self.points is None:
            self.points = self._detection_points_for(path)
        if self.points is None:
            self.points = self._default_points()
            self.status.set("Установлены стартовые 4 точки. Переместите их на нужные углы и сохраните.")
        else:
            self.points = self._clamp_points(self.points)
            self.status.set("Разметка найдена. При необходимости уточните 4 точки и сохраните.")
        self.file_label.configure(text=f"{self.index + 1}/{len(self.paths)} — {path.name}")
        self._fit_view_state()
        self._request_render()
        self.after(120, self._request_render)

    def _load_existing_points(self, source):
        root = Path(self.dataset_var.get()).expanduser()
        path = root / "annotations.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                item = data.get("annotations", {}).get(source.name)
                points = item.get("corners") if isinstance(item, dict) else None
                if points and len(points) == 4:
                    ih, iw = self.image.shape[:2]
                    if item.get("coordinates") == "normalized":
                        return [(float(p[0]) * iw, float(p[1]) * ih) for p in points]
                    return [(float(p[0]), float(p[1])) for p in points]
            except Exception:
                pass

        label_path = root / "labels" / f"{source.stem}.txt"
        if label_path.exists():
            try:
                line = label_path.read_text(encoding="utf-8").strip().splitlines()[0]
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) == 0:
                    _, cx, cy, w, h = map(float, parts)
                    ih, iw = self.image.shape[:2]
                    x1, x2 = cx * iw - w * iw / 2, cx * iw + w * iw / 2
                    y1, y2 = cy * ih - h * ih / 2, cy * ih + h * ih / 2
                    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            except Exception:
                pass
        return None

    def _detection_points_for(self, path):
        try:
            detection = self.initial_detection if path == self.paths[self.index] else None
            if detection is None:
                return None
            roi = detection.final_roi or detection.bbox
            if not roi:
                return None
            x1, y1, x2, y2 = map(float, roi)
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        except Exception:
            return None

    def _default_points(self):
        h, w = self.image.shape[:2]
        return [
            (w * 0.12, h * 0.12), (w * 0.88, h * 0.12),
            (w * 0.88, h * 0.88), (w * 0.12, h * 0.88),
        ]

    def _clamp_points(self, points):
        if self.image is None:
            return points
        height, width = self.image.shape[:2]
        return [
            (
                max(0.0, min(float(width - 1), float(x))),
                max(0.0, min(float(height - 1), float(y))),
            )
            for x, y in points
        ]

    def _fit_view_state(self):
        if self.image is None:
            self.view_zoom = 1.0
            self.view_center = (0.0, 0.0)
            return
        h, w = self.image.shape[:2]
        self.view_zoom = 1.0
        self.view_center = (w / 2.0, h / 2.0)
        self.zoom_var.set("100%")

    def _canvas_size(self):
        return max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)

    def _compute_view(self):
        cw, ch = self._canvas_size()
        if self.image is None or self.pil_image is None:
            return None
        ih, iw = self.image.shape[:2]
        self.fit_scale = max(
            min(max(cw - 20, 50) / iw, max(ch - 20, 50) / ih),
            1e-6,
        )
        self.fit_scale = min(self.fit_scale, 1.0)
        self.scale = self.fit_scale * self.view_zoom
        self.scale = max(self.scale, 1e-6)

        src_w = min(iw, cw / self.scale)
        src_h = min(ih, ch / self.scale)
        cx, cy = self.view_center
        half_w, half_h = src_w / 2.0, src_h / 2.0
        left = min(max(cx - half_w, 0.0), max(iw - src_w, 0.0))
        top = min(max(cy - half_h, 0.0), max(ih - src_h, 0.0))
        right = left + src_w
        bottom = top + src_h
        self.crop_left, self.crop_top = left, top

        target_w = max(1, int(round(src_w * self.scale)))
        target_h = max(1, int(round(src_h * self.scale)))
        return (left, top, right, bottom, target_w, target_h, cw, ch)

    def _request_render(self):
        if self._render_after_id is not None:
            try:
                self.after_cancel(self._render_after_id)
            except tk.TclError:
                pass
        self._render_pending = True
        self._render_after_id = self.after_idle(self._render)

    def _request_zoom_render(self):
        if self._render_after_id is not None:
            try:
                self.after_cancel(self._render_after_id)
            except tk.TclError:
                pass
        self._render_pending = True
        self._render_after_id = self.after(40, self._render)

    def _render(self):
        self._render_pending = False
        self._render_after_id = None
        self.canvas.update_idletasks()
        self.canvas.delete("overlay")
        if self.image is None or self.pil_image is None:
            if self._image_item is not None:
                self.canvas.delete(self._image_item)
                self._image_item = None
            if self.canvas.winfo_width() >= 20:
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2,
                    text="Загрузка изображения…", fill="white",
                    font=("Segoe UI", 12), justify="center", tags="overlay"
                )
            return

        try:
            view = self._compute_view()
            if view is None:
                return
            left, top, right, bottom, target_w, target_h, cw, ch = view

            # Never hand the full-resolution source image directly to Tk.
            # Render only the current viewport and keep a bounded pixel count.
            # This is especially important for 12k+ photos on Windows.
            max_render_w = max(100, min(cw, 2200))
            max_render_h = max(100, min(ch, 1600))
            if target_w > max_render_w or target_h > max_render_h:
                factor = min(max_render_w / max(target_w, 1),
                             max_render_h / max(target_h, 1))
                target_w = max(1, int(round(target_w * factor)))
                target_h = max(1, int(round(target_h * factor)))
                # Keep annotation coordinates consistent with the actual
                # rendered pixel size after the viewport cap.
                src_w = max(right - left, 1.0)
                src_h = max(bottom - top, 1.0)
                self.scale = min(target_w / src_w, target_h / src_h)

            src_box = (
                max(0, int(round(left))),
                max(0, int(round(top))),
                min(self.pil_image.width, int(round(right))),
                min(self.pil_image.height, int(round(bottom))),
            )
            photo_cache_key = (*src_box, target_w, target_h)
            if self.photo is None or self._photo_cache_key != photo_cache_key:
                crop = self.pil_image.crop(src_box)
                if crop.size != (target_w, target_h):
                    crop = crop.resize((target_w, target_h), Image.Resampling.BILINEAR)

                try:
                    photo = ImageTk.PhotoImage(crop, master=self.canvas)
                except Exception:
                    buffer = BytesIO()
                    crop.save(buffer, format="PNG")
                    photo = tk.PhotoImage(
                        master=self.canvas,
                        data=base64.b64encode(buffer.getvalue()).decode("ascii"),
                    )
                self.photo = photo
                self.display_photo = photo
                self._photo_cache_key = photo_cache_key

            ox = max((cw - target_w) // 2, 0)
            oy = max((ch - target_h) // 2, 0)
            self.render_offset = (ox, oy)
            if self._image_item is None:
                self._image_item = self.canvas.create_image(
                    ox, oy, image=self.photo, anchor="nw", tags="image"
                )
            else:
                self.canvas.itemconfigure(self._image_item, image=self.photo)
                self.canvas.coords(self._image_item, ox, oy)
            self.canvas.tag_lower(self._image_item)
        except Exception as exc:
            self._record_error(f"Не удалось отобразить {self.last_loaded}:\n{traceback.format_exc()}")
            self.photo = None
            self.display_photo = None
            self.canvas.create_text(
                cw // 2, ch // 2,
                text=f"Не удалось отобразить изображение:\n{exc}",
                fill="white", justify="center", tags="overlay",
                font=("Segoe UI", 11)
            )
            self.status.set(f"Ошибка отображения: {exc}")
            return

        if self.show_grid_var.get():
            for fx in (0.25, 0.5, 0.75):
                x = ox + target_w * fx
                self.canvas.create_line(x, oy, x, oy + target_h,
                                        fill="#ffffff", dash=(3, 5), tags="overlay")
            for fy in (0.25, 0.5, 0.75):
                y = oy + target_h * fy
                self.canvas.create_line(ox, y, ox + target_w, y,
                                        fill="#ffffff", dash=(3, 5), tags="overlay")

        if self.points and len(self.points) == 4:
            ordered = order_points(self.points)
            screen_points = [self._to_screen(p) for p in ordered]
            self.canvas.create_polygon(screen_points, outline="#ffcc33", width=2, fill="", tags="overlay")
        elif self.points and len(self.points) > 1:
            ordered = self.points
            screen_points = [self._to_screen(p) for p in ordered]
            self.canvas.create_line(screen_points, fill="#ffcc33", width=2, tags="overlay")
        else:
            screen_points = [self._to_screen(p) for p in self.points or []]
        if self.points:
            ordered = order_points(self.points) if len(self.points) == 4 else list(self.points)
            screen_points = [self._to_screen(p) for p in ordered]
            for i, (x, y) in enumerate(screen_points):
                active = i == self.active_point
                radius = 6 if active else 4
                self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                        fill="#152432" if not active else "#ffcc33",
                                        outline="#ffcc33", width=2, tags="overlay")
                self.canvas.create_line(x - 9, y, x + 9, y, fill="#ffcc33", width=1, tags="overlay")
                self.canvas.create_line(x, y - 9, x, y + 9, fill="#ffcc33", width=1, tags="overlay")
                self.canvas.create_text(x + 11, y - 11, text=self.POINT_NAMES[i],
                                        fill="white", font=("Segoe UI", 9, "bold"), anchor="sw", tags="overlay")

        zoom_pct = int(round(self.view_zoom * 100))
        self.zoom_var.set(f"{zoom_pct}%")

    def _record_error(self, text: str) -> None:
        message = str(text).strip()
        if not message:
            return
        self._error_messages.append(message)
        self._error_messages = self._error_messages[-20:]
        try:
            path = app_data_dir() / "logs" / "gui_errors.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + "=" * 90 + "\n" + message + "\n")
        except OSError:
            pass

    def _show_errors(self) -> None:
        log_dir = app_data_dir() / "logs"
        parts = []
        if self._error_messages:
            parts.append("ОШИБКИ ТЕКУЩЕГО ОКНА\n\n" + "\n\n".join(self._error_messages))
        for title, filename in (("ОШИБКИ GUI", "gui_errors.log"), ("ЖУРНАЛ ОБРАБОТКИ", "coreboxcropper.log")):
            path = log_dir / filename
            try:
                text = path.read_text(encoding="utf-8") if path.exists() else "Журнал пуст."
            except OSError as exc:
                text = f"Не удалось прочитать журнал: {exc}"
            parts.append(f"{title}\n\n{text}")
        ErrorLogDialog(
            self, "\n\n".join(parts), title="Ошибки / журнал ML-разметки",
            on_clear=self._error_messages.clear,
        )

    def _to_screen(self, point):
        return (self.render_offset[0] + (point[0] - self.crop_left) * self.scale,
                self.render_offset[1] + (point[1] - self.crop_top) * self.scale)

    def _to_image(self, point):
        if not self.scale:
            return 0.0, 0.0
        return (self.crop_left + (point[0] - self.render_offset[0]) / self.scale,
                self.crop_top + (point[1] - self.render_offset[1]) / self.scale)

    def _inside(self, x, y):
        if self.image is None:
            return False
        h, w = self.image.shape[:2]
        return 0 <= x < w and 0 <= y < h

    def _nearest_point_index(self, ix, iy):
        if not self.points:
            return None
        points = order_points(self.points) if len(self.points) == 4 else list(self.points)
        distances = [((px - ix) ** 2 + (py - iy) ** 2, i)
                     for i, (px, py) in enumerate(points)]
        best_dist, best_idx = min(distances, key=lambda t: t[0])
        radius_px = max(12.0 / max(self.scale, 1e-6), 3.0)
        return best_idx if best_dist <= radius_px * radius_px else None

    def _press(self, event):
        if self.image is None:
            return
        ix, iy = self._to_image((event.x, event.y))
        if not self._inside(ix, iy):
            return
        if self.points:
            self.points = order_points(self.points) if len(self.points) == 4 else list(self.points)
            idx = self._nearest_point_index(ix, iy)
            if idx is not None:
                self.active_point = idx
                self.canvas.configure(cursor="tcross")
                self._request_render()
                return
            if len(self.points) == 4:
                polygon = np.asarray(self.points, dtype=np.float32)
                if cv2.pointPolygonTest(polygon, (float(ix), float(iy)), False) >= 0:
                    self._frame_drag_anchor = (ix, iy, list(self.points))
                    self.canvas.configure(cursor="fleur")
                    self.status.set("Перемещение всей рамки")
                    return
        if self.points is None:
            self.points = []
        if len(self.points) < 4:
            self.points.append((ix, iy))
            self.active_point = len(self.points) - 1
            self.canvas.configure(cursor="tcross")
            self.status.set(f"Поставлено точек: {len(self.points)}/4")
            self._request_render()
            return
        self.status.set("Тяните рамку внутри области или перетаскивайте отдельную точку.")

    def _drag(self, event):
        if self.image is None or not self.points:
            return
        ix, iy = self._to_image((event.x, event.y))
        h, w = self.image.shape[:2]
        ix, iy = max(0.0, min(float(w - 1), ix)), max(0.0, min(float(h - 1), iy))
        if self._frame_drag_anchor is not None:
            ax, ay, original = self._frame_drag_anchor
            dx, dy = ix - ax, iy - ay
            min_x = min(x for x, _ in original)
            max_x = max(x for x, _ in original)
            min_y = min(y for _, y in original)
            max_y = max(y for _, y in original)
            dx = min(max(dx, -min_x), float(w - 1) - max_x)
            dy = min(max(dy, -min_y), float(h - 1) - max_y)
            self.points = [(x + dx, y + dy) for x, y in original]
        elif self.active_point is not None:
            points = order_points(self.points) if len(self.points) == 4 else list(self.points)
            points[self.active_point] = (ix, iy)
            self.points = points
        self._request_render()

    def _release(self, _event):
        if self.points:
            if len(self.points) == 4:
                self.points = order_points(self.points)
        self.active_point = None
        self._frame_drag_anchor = None
        self.canvas.configure(cursor="crosshair")
        self._request_render()

    def _pan_press(self, event):
        if self.image is None:
            return
        self._pan_anchor = (event.x, event.y, self.view_center[0], self.view_center[1])
        self.canvas.configure(cursor="fleur")

    def _pan_drag(self, event):
        if self._pan_anchor is None or self.image is None or not self.scale:
            return
        sx, sy, cx, cy = self._pan_anchor
        dx = (event.x - sx) / self.scale
        dy = (event.y - sy) / self.scale
        h, w = self.image.shape[:2]
        self.view_center = (min(max(cx - dx, 0.0), float(w)),
                            min(max(cy - dy, 0.0), float(h)))
        self._request_render()

    def _pan_release(self, _event):
        self._pan_anchor = None
        self.canvas.configure(cursor="crosshair")
        self._request_render()

    def _zoom_to(self, factor, event=None):
        if self.image is None:
            return
        old_scale = self.scale if self.scale else 1.0
        if event is not None:
            before = self._to_image((event.x, event.y))
        else:
            before = self.view_center
        self.view_zoom = min(max(self.view_zoom * factor, 0.25), 12.0)
        # Recompute scale, then keep the image point under the cursor fixed.
        self._compute_view()
        if event is not None:
            after = self._to_image((event.x, event.y))
            self.view_center = (self.view_center[0] + (before[0] - after[0]),
                                self.view_center[1] + (before[1] - after[1]))
        self._request_zoom_render()

    def _zoom_step(self, factor):
        self._zoom_to(factor)

    def _wheel_zoom(self, event):
        if event.delta == 0:
            return
        self._zoom_to(1.25 if event.delta > 0 else 1 / 1.25, event)

    def _wheel_zoom_linux(self, direction, event):
        self._zoom_to(1.25 if direction > 0 else 1 / 1.25, event)

    def _fit_view(self):
        self._fit_view_state()
        self._request_render()

    def _one_to_one(self):
        if self.image is None:
            return
        self.view_zoom = 1.0 / max(self.fit_scale, 1e-6)
        self.view_zoom = min(max(self.view_zoom, 0.25), 12.0)
        self.view_center = (self.image.shape[1] / 2.0, self.image.shape[0] / 2.0)
        self._request_render()

    def _clear_points(self):
        self.points = []
        self.active_point = None
        self._frame_drag_anchor = None
        self.status.set("Точки очищены. Щёлкайте по углам по порядку: P1, P2, P3, P4.")
        self._request_render()

    def _reset_from_detection(self):
        points = self._detection_points_for(self.paths[self.index])
        if points is None:
            self.status.set("Автоматическая рамка для этой фотографии недоступна.")
            return
        self.points = order_points(points)
        self.active_point = None
        self.status.set("Точки восстановлены из автоматического результата. Теперь подправьте их.")
        self._request_render()

    def _save_current(self):
        if self.image is None:
            return False
        if not self.points or len(self.points) != 4:
            messagebox.showwarning("Нет 4 точек", "Сначала задайте все четыре угла нужной области.", parent=self)
            return False
        try:
            self.points = order_points(self.points)
            root, images_dir, labels_dir = self._dataset_dirs()
            source = self.paths[self.index]
            destination = images_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

            h, w = self.image.shape[:2]
            points = []
            for x, y in self.points:
                x = max(0.0, min(float(w - 1), float(x)))
                y = max(0.0, min(float(h - 1), float(y)))
                points.append((x, y))

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            label_path = labels_dir / f"{source.stem}.txt"
            label_path.write_text(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")

            self._update_manifest(root, source, destination, points, (x1, y1, x2, y2), w, h)
            self.status.set(f"Сохранено: {source.name} — 4 точки")
            return True
        except Exception as exc:
            text = traceback.format_exc()
            self.status.set(f"Ошибка сохранения: {exc}")
            messagebox.showerror("Ошибка разметки", text, parent=self)
            return False

    def _update_manifest(self, root, source, destination, points, bbox, width, height):
        path = root / "annotations.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "version": 2, "classes": [self.CLASS_NAME], "annotation_type": "quadrilateral", "annotations": {}
            }
        except Exception:
            data = {"version": 2, "classes": [self.CLASS_NAME], "annotation_type": "quadrilateral", "annotations": {}}
        x1, y1, x2, y2 = bbox
        normalized = [[round(float(x / width), 8), round(float(y / height), 8)] for x, y in points]
        data["version"] = 2
        data["classes"] = [self.CLASS_NAME]
        data["annotation_type"] = "quadrilateral"
        data.setdefault("annotations", {})
        data["annotations"][source.name] = {
            "image": str(destination.relative_to(root)).replace("\\", "/"),
            "source": str(source),
            "width": int(width),
            "height": int(height),
            "corners": normalized,
            "coordinates": "normalized",
            "bbox_xyxy": [round(float(v), 2) for v in (x1, y1, x2, y2)],
            "class": self.CLASS_NAME,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        yaml = root / "dataset.yaml"
        yaml.write_text(
            f"path: {json.dumps(root.as_posix(), ensure_ascii=False)}\n"
            f"train: images\nval: images\nnames:\n  0: {self.CLASS_NAME}\n",
            encoding="utf-8"
        )

    def _save_next(self):
        if not self._save_current():
            return
        if self.index < len(self.paths) - 1:
            self.index += 1
            self._load_current()
        else:
            self.status.set("Разметка сохранена для последней фотографии.")

    def _move(self, step):
        if not self.paths:
            return
        self.index = min(max(self.index + step, 0), len(self.paths) - 1)
        self._load_current()

    def _close(self):
        if self.on_close:
            try:
                self.on_close()
            except Exception:
                pass
        self.destroy()


class ManualEditor(tk.Toplevel):
    def __init__(self, parent, image, on_done):
        super().__init__(parent)
        self.title("Указать область вручную")
        self.geometry("900x700")
        self.image = image
        self.on_done = on_done
        self.canvas = tk.Canvas(self, bg="#172838", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        tk.Label(
            self, text="Перетаскивайте четыре точки по углам нужного ящика, затем нажмите «Подтвердить».",
            fg="#43525d"
        ).pack(pady=(0, 4))
        tk.Button(
            self, text="Подтвердить", command=self._confirm,
            bg="#2c9b68", fg="white", relief="flat", padx=16, pady=8
        ).pack(pady=(0, 10))
        self.photo = None
        self.display_photo = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.points = []
        self.active = None
        self.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self._render()

    def _render(self):
        if self.canvas.winfo_width() < 20:
            return
        pil = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        max_w, max_h = self.canvas.winfo_width() - 20, self.canvas.winfo_height() - 20
        self.scale = min(max_w / pil.width, max_h / pil.height, 1.0)
        shown = pil.resize((max(1, int(pil.width * self.scale)), max(1, int(pil.height * self.scale))))
        self.offset = ((self.canvas.winfo_width() - shown.width) // 2, (self.canvas.winfo_height() - shown.height) // 2)
        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, image=self.photo, anchor="nw")
        if not self.points:
            pad_x, pad_y = pil.width * 0.1, pil.height * 0.1
            self.points = [(pad_x, pad_y), (pil.width * 0.7, pad_y), (pil.width * 0.7, pil.height * 0.9), (pad_x, pil.height * 0.9)]
        ordered = order_points(self.points)
        screen_points = [self._to_screen(point) for point in ordered]
        self.canvas.create_polygon(screen_points, outline="#ffcc33", width=3, fill="")
        for i, (x, y) in enumerate(screen_points):
            self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill="#ffcc33", outline="#172838", width=2)
            self.canvas.create_text(x + 14, y - 14, text=str(i + 1), fill="white", font=("Segoe UI", 10, "bold"))

    def _to_screen(self, point):
        return self.offset[0] + point[0] * self.scale, self.offset[1] + point[1] * self.scale

    def _to_image(self, point):
        return ((point[0] - self.offset[0]) / self.scale, (point[1] - self.offset[1]) / self.scale)

    def _press(self, event):
        screen_points = [self._to_screen(point) for point in order_points(self.points)]
        distances = [
            ((event.x - x) ** 2 + (event.y - y) ** 2, index)
            for index, (x, y) in enumerate(screen_points)
        ]
        if not distances:
            self.active = None
            return
        best = min(distances, key=lambda item: float(item[0]))
        self.active = int(best[1]) if float(best[0]) < 24 ** 2 else None

    def _drag(self, event):
        if self.active is not None:
            ordered = order_points(self.points)
            ordered[self.active] = self._to_image((event.x, event.y))
            self.points = ordered
            self._render()

    def _confirm(self):
        self.on_done(order_points(self.points))
        self.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, config, on_save):
        super().__init__(parent)
        self.title("Настройки алгоритма")
        self.transient(parent)
        self.resizable(False, False)
        self.on_save = on_save
        self.entries = {}
        fields = [
            ("max_analysis_dimension", "Максимальный размер анализа"),
            ("confidence_threshold", "Порог уверенности"),
            ("crop_margin_percent", "Запас итогового crop"),
            ("ruler_color_min_saturation", "Минимальная насыщенность шкалы"),
            ("ruler_tick_step_cm", "Шаг делений линейки (см)"),
            ("ruler_length_cm", "Длина линейки (см)"),
            ("expected_core_cells", "Количество ячеек"),
        ]
        for row, (name, label) in enumerate(fields):
            tk.Label(self, text=label, anchor="w").grid(row=row, column=0, padx=12, pady=6, sticky="w")
            entry = tk.Entry(self, width=16)
            entry.insert(0, str(getattr(config, name)))
            entry.grid(row=row, column=1, padx=12, pady=6)
            self.entries[name] = entry
        tk.Button(
            self, text="Сохранить", command=lambda: self._save(config),
            bg="#2c9b68", fg="white", relief="flat", padx=14, pady=7
        ).grid(row=len(fields), column=0, columnspan=2, pady=12)

    def _save(self, original):
        try:
            values = {}
            for name, entry in self.entries.items():
                old = getattr(original, name)
                values[name] = int(entry.get()) if isinstance(old, int) else float(entry.get())
            updated = DetectorConfig(**{**original.__dict__, **values})
            updated.validate()
            self.on_save(updated)
            self.destroy()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Некорректные настройки", str(exc), parent=self)
