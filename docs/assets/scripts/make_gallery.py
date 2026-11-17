"""Create the notebook thumbnails in docs/assets/gallery/ for the README.

Takes one figure from each executed example notebook, trims the white margin
and pads it to a uniform 4:3 thumbnail.

Usage (from the repository root):  python docs/assets/scripts/make_gallery.py
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageChops

#: notebook name -> index of the figure to use
SELECTION = {
    "operators-and-solvers": 2,
    "nonlinear-problems": 1,
    "quantum-dynamics": 1,
    "hydrogen-atom": 0,
    "heat-conduction": 2,
    "heat-equation-rod": 0,
    "poisson-equation-2d": 0,
    "quantum-harmonic-oscillator": 0,
    "wave-equation-drum": 0,
    "spherical-harmonics": 0,
    "convergence-study": 1,
    "adaptive-refinement-demo": 0,
}
SIZE = (480, 360)
MARGIN = 12


def figures(notebook: Path) -> list[Image.Image]:
    cells = json.loads(notebook.read_text())["cells"]
    return [
        Image.open(io.BytesIO(base64.b64decode(out["data"]["image/png"]))).convert("RGB")
        for cell in cells
        for out in cell.get("outputs", [])
        if "image/png" in out.get("data", {})
    ]


def thumbnail(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, "white")
    box = ImageChops.difference(image, background).getbbox()
    if box:
        image = image.crop(box)
    w, h = SIZE[0] - 2 * MARGIN, SIZE[1] - 2 * MARGIN
    scale = min(w / image.width, h / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    thumb = Image.new("RGB", SIZE, "white")
    thumb.paste(image, ((SIZE[0] - image.width) // 2, (SIZE[1] - image.height) // 2))
    return thumb


def main() -> None:
    out_dir = Path("docs/assets/gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, index in SELECTION.items():
        thumb = thumbnail(figures(Path("examples") / f"{name}.ipynb")[index])
        thumb.save(out_dir / f"{name}.png", optimize=True)
        print("wrote", out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
