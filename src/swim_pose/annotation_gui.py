from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

from .constants import FRAME_STATUSES, KEYPOINT_NAMES, SKELETON_EDGES
from .io import read_json, write_json


class AnnotationGui:
    def __init__(self, annotation_root: str | Path, frame_root: str | Path) -> None:
        self.annotation_root = Path(annotation_root)
        self.frame_root = Path(frame_root)
        self.annotation_paths = sorted(self.annotation_root.rglob("*.json"))
        if not self.annotation_paths:
            raise ValueError(f"No annotation JSON files found under {self.annotation_root}")

        self.index = 0
        self.selected_keypoint = 0
        self.scale = 1.0
        self.current_annotation: dict | None = None
        self.current_photo = None
        self.current_image = None
        self.undo_stack: list[dict] = []

    def run(self) -> None:
        import tkinter as tk

        self.tk = tk.Tk()
        self.tk.title("Swim Pose Annotation")
        self.tk.geometry("1560x940")
        self.tk.protocol("WM_DELETE_WINDOW", self._on_close)

        root = self.tk
        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(root, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(left, bg="black", width=1200, height=900)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.header_var = tk.StringVar()
        self.info_var = tk.StringVar()
        tk.Label(right, textvariable=self.header_var, justify=tk.LEFT, wraplength=330).pack(anchor="w", padx=10, pady=(10, 4))
        tk.Label(right, textvariable=self.info_var, justify=tk.LEFT, wraplength=330).pack(anchor="w", padx=10, pady=(0, 8))

        self.listbox = tk.Listbox(right, exportselection=False, width=42, height=24)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_keypoint)

        nav_row = tk.Frame(right)
        nav_row.pack(fill=tk.X, padx=10, pady=8)
        tk.Button(nav_row, text="Prev", command=self.prev_annotation).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(nav_row, text="Save", command=self.save_current).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        tk.Button(nav_row, text="Next", command=self.next_annotation).pack(side=tk.LEFT, expand=True, fill=tk.X)

        edit_row = tk.Frame(right)
        edit_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Button(edit_row, text="Undo", command=self.undo_last_change).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(edit_row, text="Clear Point", command=self.clear_selected_point).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        tk.Button(edit_row, text="Clear Frame", command=self.clear_frame_points).pack(side=tk.LEFT, expand=True, fill=tk.X)

        status_row_top = tk.Frame(right)
        status_row_top.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Button(status_row_top, text="Labeled", command=lambda: self.set_frame_status("labeled")).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(status_row_top, text="No Swimmer", command=self.mark_no_swimmer).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        status_row_bottom = tk.Frame(right)
        status_row_bottom.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Button(status_row_bottom, text="Review", command=lambda: self.set_frame_status("review")).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(status_row_bottom, text="Pending", command=lambda: self.set_frame_status("pending")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        help_text = (
            "Controls:\n"
            "Left click: set selected keypoint\n"
            "Right click / Delete / Backspace / c: clear selected keypoint\n"
            "0 / 1 / 2: set visibility\n"
            "j / k: next or previous keypoint\n"
            "a / d: previous or next file\n"
            "l: mark labeled\n"
            "n: mark no swimmer\n"
            "r: mark review\n"
            "p: mark pending\n"
            "Ctrl+Z: undo\n"
            "Ctrl+S: save\n"
        )
        tk.Label(right, text=help_text, justify=tk.LEFT, anchor="w").pack(fill=tk.X, padx=10, pady=(0, 10))

        self.tk.bind("<KeyPress-0>", lambda _event: self.set_visibility(0))
        self.tk.bind("<KeyPress-1>", lambda _event: self.set_visibility(1))
        self.tk.bind("<KeyPress-2>", lambda _event: self.set_visibility(2))
        self.tk.bind("<KeyPress-j>", lambda _event: self.change_keypoint(1))
        self.tk.bind("<KeyPress-k>", lambda _event: self.change_keypoint(-1))
        self.tk.bind("<KeyPress-d>", lambda _event: self.next_annotation())
        self.tk.bind("<KeyPress-a>", lambda _event: self.prev_annotation())
        self.tk.bind("<KeyPress-l>", lambda _event: self.set_frame_status("labeled"))
        self.tk.bind("<KeyPress-n>", lambda _event: self.mark_no_swimmer())
        self.tk.bind("<KeyPress-r>", lambda _event: self.set_frame_status("review"))
        self.tk.bind("<KeyPress-p>", lambda _event: self.set_frame_status("pending"))
        self.tk.bind("<KeyPress-c>", lambda _event: self.clear_selected_point())
        self.tk.bind("<Delete>", lambda _event: self.clear_selected_point())
        self.tk.bind("<BackSpace>", lambda _event: self.clear_selected_point())
        self.tk.bind("<Control-z>", lambda _event: self.undo_last_change())
        self.tk.bind("<Control-s>", lambda _event: self.save_current())

        self.load_annotation(0)
        self.tk.mainloop()

    def load_annotation(self, index: int) -> None:
        self.index = max(0, min(index, len(self.annotation_paths) - 1))
        path = self.annotation_paths[self.index]
        self.current_annotation = read_json(path)
        self._ensure_metadata()
        self.undo_stack = []
        image_path = self._resolve_image_path(self.current_annotation["image_path"])
        original = Image.open(image_path).convert("RGB")

        max_width, max_height = 1180, 880
        width, height = original.size
        self.scale = min(max_width / width, max_height / height, 1.0)
        display_size = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
        self.current_image = original.resize(display_size, Image.Resampling.BILINEAR)
        self._refresh_canvas()
        self.info_var.set(f"image: {self.current_annotation.get('image_path', '')}")

    def refresh_keypoint_list(self) -> None:
        assert self.current_annotation is not None
        self.listbox.delete(0, "end")
        for index, keypoint in enumerate(KEYPOINT_NAMES):
            point = self.current_annotation["points"][keypoint]
            x = point.get("x")
            y = point.get("y")
            visibility = point.get("visibility", 0)
            coord = "--" if x is None or y is None else f"({int(x)}, {int(y)})"
            marker = ">" if index == self.selected_keypoint else " "
            self.listbox.insert("end", f"{marker} {keypoint:<15} v={visibility} {coord}")
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.selected_keypoint)
        self.listbox.activate(self.selected_keypoint)

    def save_current(self) -> None:
        assert self.current_annotation is not None
        self._ensure_metadata()
        write_json(self.annotation_paths[self.index], self.current_annotation)
        self._set_info(f"Saved {self.annotation_paths[self.index].name}")

    def next_annotation(self) -> None:
        self.save_current()
        if self.index < len(self.annotation_paths) - 1:
            self.load_annotation(self.index + 1)

    def prev_annotation(self) -> None:
        self.save_current()
        if self.index > 0:
            self.load_annotation(self.index - 1)

    def set_visibility(self, visibility: int) -> None:
        assert self.current_annotation is not None
        self._push_undo()
        keypoint = KEYPOINT_NAMES[self.selected_keypoint]
        point = self.current_annotation["points"][keypoint]
        point["visibility"] = visibility
        if visibility == 0:
            point["x"] = None
            point["y"] = None
        elif self.frame_status == "no_swimmer":
            self.current_annotation["metadata"]["frame_status"] = "labeled"
        self._set_info(f"Set {keypoint} visibility to {visibility}")
        self._refresh_canvas()

    def change_keypoint(self, delta: int) -> None:
        self.selected_keypoint = (self.selected_keypoint + delta) % len(KEYPOINT_NAMES)
        self.refresh_keypoint_list()

    def clear_selected_point(self) -> None:
        assert self.current_annotation is not None
        self._push_undo()
        keypoint = KEYPOINT_NAMES[self.selected_keypoint]
        self.current_annotation["points"][keypoint] = {"x": None, "y": None, "visibility": 0}
        self._set_info(f"Cleared {keypoint}")
        self._refresh_canvas()

    def clear_frame_points(self) -> None:
        assert self.current_annotation is not None
        self._push_undo()
        for keypoint in KEYPOINT_NAMES:
            self.current_annotation["points"][keypoint] = {"x": None, "y": None, "visibility": 0}
        self.current_annotation["metadata"]["frame_status"] = "pending"
        self._set_info("Cleared all keypoints and marked frame as pending")
        self._refresh_canvas()

    def mark_no_swimmer(self) -> None:
        assert self.current_annotation is not None
        self._push_undo()
        for keypoint in KEYPOINT_NAMES:
            self.current_annotation["points"][keypoint] = {"x": None, "y": None, "visibility": 0}
        self.current_annotation["metadata"]["frame_status"] = "no_swimmer"
        self._set_info("Marked frame as no swimmer")
        self._refresh_canvas()

    def set_frame_status(self, status: str) -> None:
        assert self.current_annotation is not None
        if status not in FRAME_STATUSES:
            return
        self._push_undo()
        self.current_annotation["metadata"]["frame_status"] = status
        self._set_info(f"Marked frame as {status}")
        self._refresh_canvas()

    def undo_last_change(self) -> None:
        if not self.undo_stack:
            return
        self.current_annotation = self.undo_stack.pop()
        self._ensure_metadata()
        self._set_info("Reverted the last edit")
        self._refresh_canvas()

    def _on_left_click(self, event) -> None:
        assert self.current_annotation is not None
        self._push_undo()
        keypoint = KEYPOINT_NAMES[self.selected_keypoint]
        point = self.current_annotation["points"][keypoint]
        point["x"] = round(float(event.x / self.scale), 1)
        point["y"] = round(float(event.y / self.scale), 1)
        if point.get("visibility", 0) == 0:
            point["visibility"] = 2
        if self.frame_status in {"pending", "no_swimmer"}:
            self.current_annotation["metadata"]["frame_status"] = "labeled"
        self._set_info(f"Placed {keypoint} at ({point['x']}, {point['y']})")
        self._refresh_canvas()

    def _on_right_click(self, _event) -> None:
        self.clear_selected_point()

    def _on_select_keypoint(self, _event) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.selected_keypoint = int(selection[0])
            self.refresh_keypoint_list()

    def _refresh_canvas(self) -> None:
        assert self.current_annotation is not None
        assert self.current_image is not None
        self.current_photo = ImageTk.PhotoImage(self._draw_overlay(self.current_image.copy()))
        self.canvas.delete("all")
        self.canvas.config(width=self.current_image.width, height=self.current_image.height)
        self.canvas.create_image(0, 0, image=self.current_photo, anchor="nw")
        self.header_var.set(
            f"File {self.index + 1}/{len(self.annotation_paths)}\n"
            f"{self.annotation_paths[self.index].name}\n"
            f"clip={self.current_annotation.get('clip_id', '')}, frame={self.current_annotation.get('frame_index', 0)}\n"
            f"status={self.frame_status}"
        )
        self.refresh_keypoint_list()

    def _draw_overlay(self, display_image: Image.Image) -> Image.Image:
        assert self.current_annotation is not None
        draw = ImageDraw.Draw(display_image)
        points = self.current_annotation["points"]
        scaled_points: dict[str, tuple[float, float]] = {}
        for keypoint, point in points.items():
            if point.get("x") is None or point.get("y") is None:
                continue
            scaled_points[keypoint] = (float(point["x"]) * self.scale, float(point["y"]) * self.scale)

        for start, end in SKELETON_EDGES:
            if start in scaled_points and end in scaled_points:
                draw.line([scaled_points[start], scaled_points[end]], fill=(0, 255, 180), width=2)

        for index, keypoint in enumerate(KEYPOINT_NAMES):
            point = points[keypoint]
            if point.get("x") is None or point.get("y") is None:
                continue
            x, y = scaled_points[keypoint]
            radius = 5 if index == self.selected_keypoint else 4
            color = _point_color(point.get("visibility", 0), selected=index == self.selected_keypoint)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
            draw.text((x + 6, y - 6), str(index + 1), fill=(255, 255, 255))
        return display_image

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        candidate = self.frame_root / path
        if candidate.exists():
            return candidate
        if path.exists():
            return path
        raise FileNotFoundError(f"Image path not found: {image_path}")

    def _ensure_metadata(self) -> None:
        assert self.current_annotation is not None
        self.current_annotation.setdefault("metadata", {})
        metadata = self.current_annotation["metadata"]
        metadata.setdefault("frame_status", "pending")
        metadata.setdefault("stroke_phase", "")
        metadata.setdefault("difficulties", [])

    def _push_undo(self) -> None:
        assert self.current_annotation is not None
        self.undo_stack.append(deepcopy(self.current_annotation))
        if len(self.undo_stack) > 50:
            self.undo_stack = self.undo_stack[-50:]

    def _set_info(self, message: str) -> None:
        image_path = ""
        if self.current_annotation is not None:
            image_path = self.current_annotation.get("image_path", "")
        if image_path:
            self.info_var.set(f"{message}\nimage: {image_path}")
            return
        self.info_var.set(message)

    @property
    def frame_status(self) -> str:
        assert self.current_annotation is not None
        return self.current_annotation.get("metadata", {}).get("frame_status", "pending")

    def _on_close(self) -> None:
        self.save_current()
        self.tk.destroy()


def _point_color(visibility: int, selected: bool) -> tuple[int, int, int]:
    if selected:
        return (255, 128, 0)
    if visibility == 2:
        return (40, 220, 120)
    if visibility == 1:
        return (255, 220, 0)
    return (200, 60, 60)
