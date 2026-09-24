"""
Qubit 1X dsDNA HS reagent holder: working-solution bottle (RGT) + Standard 1 + Standard 2.
Lectern style: 45° bevel carrying the pocket labels, "Qubit 1X dsDNA HS" on the front,
the 1X/HS helix emblem on top, finger lifts under both ends. Units: mm.

Needs tapestation_holders.py (for the emblem) in the same folder.
    pip install trimesh manifold3d shapely matplotlib numpy
    python qubit_holder.py
"""
import numpy as np
import trimesh
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from manifold3d import Manifold, CrossSection, FillRule, OpType, JoinType
import tapestation_holders as TS

# ---------------- measured (ruler, across the bottom) ----------------
BOTTLE_D = 63.5        # 2.5 in
STD_D = 25.4           # 1.0 in
BOTTLE_CLEAR = 1.5     # added to the diameter; confirm with the fit-test rings
STD_CLEAR = 1.2

# ---------------- holder ----------------
BOTTLE_DEPTH, STD_DEPTH, FLOOR = 25.0, 20.0, 3.0
H = BOTTLE_DEPTH + FLOOR
END_WALL, WEB, BACK_WALL, FRONT_ZONE, CORNER = 13.0, 8.0, 6.0, 18.0, 5.0
BEVEL = 10.0                       # 45° bevel, 10 mm back and 10 mm down
LABEL_CAP, ENGRAVE = 4.6, 0.8
NAME = ("Qubit", 9.0, "1X dsDNA HS", 6.0)   # front face: big word, then the assay
FRONT_DEPTH = 0.6
EMBLEM = dict(symbol="dsDNA", turns=2, length=22.0, badge="1X", hs=True)
EMBLEM_SCALE = 1.6

# ---------------- fit-test rings ----------------
RING_TESTS = [("RGT", BOTTLE_D, 1.0), ("RGT", BOTTLE_D, 2.0), ("STD", STD_D, 0.8), ("STD", STD_D, 1.6)]

FONT = FontProperties(family="DejaVu Sans", weight="bold")


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add)


def text_cs(s, cap):
    ref = TextPath((0, 0), "H", size=10, prop=FONT).get_extents()
    k = cap / ref.height
    rings = [np.array(p) * k for p in TextPath((0, 0), s, size=10, prop=FONT).to_polygons() if len(p) >= 3]
    cs = CrossSection(rings, FillRule.EvenOdd)
    b = cs.bounds()
    return cs.translate([-(b[0] + b[2]) / 2, -(b[1] + b[3]) / 2])


def text_width(s, cap):
    b = text_cs(s, cap).bounds()
    return b[2] - b[0]


def rounded_block(W, D, h, corner=CORNER):
    base = CrossSection.square([W - 2 * corner, D - 2 * corner]).translate([corner, corner]).offset(
        corner, JoinType.Round, circular_segments=48)
    bot = Manifold.batch_hull([Manifold.extrude(base.offset(-0.4, JoinType.Round), 0.01),
                               Manifold.extrude(base, 0.01).translate([0, 0, 0.4])])
    mid = Manifold.extrude(base, h - 1.4).translate([0, 0, 0.4])
    top = Manifold.batch_hull([Manifold.extrude(base, 0.01).translate([0, 0, h - 1.01]),
                               Manifold.extrude(base.offset(-1.0, JoinType.Round), 0.01).translate([0, 0, h - 0.01])])
    return union([bot, mid, top])


def pocket(x, y, d, depth):
    r = d / 2
    return union([Manifold.cylinder(depth + 1, r, r, 160).translate([x, y, H - depth]),
                  Manifold.cylinder(1.01, r, r + 1.0, 160).translate([x, y, H - 1.0])])


def front_text(s, x, z, cap):
    cut = Manifold.extrude(text_cs(s, cap), FRONT_DEPTH + 0.5).translate([0, 0, -FRONT_DEPTH])
    return cut.rotate([90, 0, 0]).translate([x, 0, z])


def finger_lifts(W, D, length=52.0):
    prof = CrossSection([np.array([(-1, -1), (-1, 16), (0, 16), (9, 7), (9, -1)])], FillRule.EvenOdd)
    left = Manifold.extrude(prof, length).rotate([90, 0, 0]).translate([0, D / 2 + length / 2, 0])
    return union([left, left.mirror([1, 0, 0]).translate([W, 0, 0])])


def layout():
    bd, sd = BOTTLE_D + BOTTLE_CLEAR, STD_D + STD_CLEAR
    bx = END_WALL + bd / 2
    s1 = END_WALL + bd + WEB + sd / 2
    s2 = s1 + sd + WEB
    W = s2 + sd / 2 + END_WALL
    D = FRONT_ZONE + bd + BACK_WALL
    return dict(W=W, D=D, bd=bd, sd=sd,
                pockets=[("RGT", bx, FRONT_ZONE + bd / 2, bd, BOTTLE_DEPTH),
                         ("STD 1", s1, FRONT_ZONE + sd / 2, sd, STD_DEPTH),
                         ("STD 2", s2, FRONT_ZONE + sd / 2, sd, STD_DEPTH)])


def make_holder():
    L = layout()
    W, D = L["W"], L["D"]
    cuts = []
    wedge = CrossSection([np.array([(-1, H - BEVEL), (BEVEL, H + 0.01), (-1, H + 5)])], FillRule.EvenOdd)
    cuts.append(Manifold.extrude(wedge, W + 2).rotate([90, 0, 0]).rotate([0, 0, 90]).translate([-1, 0, 0]))
    for lab, x, y, d, depth in L["pockets"]:
        cuts.append(pocket(x, y, d, depth))
        t = Manifold.extrude(text_cs(lab, LABEL_CAP), ENGRAVE + 1).translate([0, 0, -ENGRAVE]).rotate([45, 0, 0])
        cuts.append(t.translate([x, BEVEL / 2, H - BEVEL / 2]))
    std = [p for p in L["pockets"] if p[0].startswith("STD")]
    ex = sum(p[1] for p in std) / len(std)
    ey = (FRONT_ZONE + L["sd"] + D) / 2 + 1
    emb = TS.to_cs(TS.lockup(EMBLEM)).scale([EMBLEM_SCALE, EMBLEM_SCALE])
    cuts.append(Manifold.extrude(emb, ENGRAVE + 1).translate([ex, ey, H - ENGRAVE]))
    big, bcap, small, scap = NAME
    w1, w2, gap = text_width(big, bcap), text_width(small, scap), 6.0
    x1 = W / 2 - (w1 + gap + w2) / 2 + w1 / 2
    zc = (H - BEVEL) / 2 + 0.2
    cuts += [front_text(big, x1, zc, bcap), front_text(small, x1 + w1 / 2 + gap + w2 / 2, zc, scap),
             finger_lifts(W, D)]
    return rounded_block(W, D, H) - union(cuts)


def make_fit_rings(height=5.0, wall=2.4):
    """One ring per test clearance, each with a tab engraved with its clearance.
    Big rings in the back row, small rings in front, so the set fits the A1 mini bed."""
    def ring(d, clr):
        r_in = (d + clr) / 2
        body = Manifold.cylinder(height, r_in + wall, r_in + wall, 160) - \
            Manifold.cylinder(height + 2, r_in, r_in, 160).translate([0, 0, -1])
        tab = Manifold.cube([16, 9, height]).translate([-8, -(r_in + wall + 8), 0])
        label = Manifold.extrude(text_cs(f"+{clr:.1f}", 4.0), ENGRAVE + 1).translate(
            [0, -(r_in + wall + 3.5), height - ENGRAVE])
        return union([body, tab]) - label, r_in + wall
    parts, x_big, x_small = [], 0.0, 0.0
    big = [t for t in RING_TESTS if t[1] > 40]
    small = [t for t in RING_TESTS if t[1] <= 40]
    row_gap, y_small = 6.0, None
    for name, d, clr in big:
        piece, R = ring(d, clr)
        parts.append(piece.translate([x_big + R, 0, 0]))
        x_big += 2 * R + 6
        y_small = -(R + 8 + row_gap)
    for name, d, clr in small:
        piece, R = ring(d, clr)
        parts.append(piece.translate([x_small + R + 20, y_small - R, 0]))
        x_small += 2 * R + 30
    return union(parts)


def export(m, path):
    mesh = m.to_mesh()
    t = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts)
    t.export(path)
    return t


if __name__ == "__main__":
    L = layout()
    t = export(make_holder(), "qubit_1x_hs_holder.stl")
    print(f"holder: {L['W']:.1f} x {L['D']:.1f} x {H:.1f} mm, watertight={t.is_watertight}, "
          f"pockets {L['bd']:.1f} / {L['sd']:.1f} mm")
    r = export(make_fit_rings(), "qubit_fit_test_rings.stl")
    print(f"fit rings: watertight={r.is_watertight}, footprint {r.extents[0]:.0f} x {r.extents[1]:.0f} mm")
