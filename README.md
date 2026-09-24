# 3d-prints

Parametric 3D print designs and the **printkit** tooling that generates them.

## Layout

```
designs/
  lab/          Lab equipment holders and accessories
  makerspace/   Demos, tools, and community designs for the makerspace
  personal/     Personal projects
printkit/       Reusable Python package — build any design from a YAML spec
tests/          pytest suite for printkit building blocks
```

## Quick start

Each design folder has its own `README.md` and a self-contained Python script.
Dependencies are `trimesh`, `manifold3d`, `numpy`, and `matplotlib`.

```bash
pip install trimesh manifold3d numpy matplotlib pillow pyembree
python designs/personal/chess_shakers/chess_shakers.py
```

## printkit (coming soon)

Describe a design in YAML → get STLs, preview renders, and print notes.
