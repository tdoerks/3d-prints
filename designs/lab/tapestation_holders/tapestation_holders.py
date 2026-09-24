"""
TapeStation reagent holders: one module per assay, holding the reagent (RGT)
and ladder (LDR) tubes. The assay emblem and tube labels are cut into the top,
and the assay name is cut into all four sides, so it stays
visible whichever faces are clipped to neighbours. Modules sit on the same 10 mm
grid and use the same bowtie keys as the essential-oil holders.

    pip install trimesh manifold3d shapely matplotlib
    python tapestation_holders.py

Add an assay by adding a line to ASSAYS. Units: mm.
"""
import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import unary_union
from shapely import affinity
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from manifold3d import Manifold, CrossSection, FillRule, JoinType, OpType

# ---------------- assays ----------------
# symbol "dsDNA" = double helix, "RNA" = single strand; helix length encodes size range
ASSAYS = {
    "D5000": dict(symbol="dsDNA", turns=2, length=22.0, badge="5K", hs=False),
    "D1000": dict(symbol="dsDNA", turns=1, length=11.0, badge="1K", hs=False),
    "RNA":   dict(symbol="RNA",   turns=2, length=22.0, badge="RNA", hs=False),
}
TUBE_LABELS = ("RGT", "LDR")      # left tube, right tube

# ---------------- module ----------------
WIDTH, DEPTH   = 40.0, 50.0       # footprint, on the 10 mm grid
TUBE_DIAM      = 10.8             # 2 mL screw-cap tube body
TUBE_CLEAR     = 0.6              # added to the diameter
TUBE_DEPTH     = 20.0             # how deep the tubes sit
FLOOR          = 2.0
HEIGHT         = TUBE_DEPTH + FLOOR
TUBE_X         = (10.0, 30.0)     # tube centres across the width
TUBE_Y         = 38.0             # tube centres, from the front edge
EMBLEM_Y       = 12.5             # emblem centre, from the front edge
LABEL_Y        = 27.5             # RGT / LDR centre, from the front edge
LABEL_CAP      = 3.5
NAME_CAP       = 6.0              # side-face name height
NAME_Z         = 16.0             # side-face name centre height (clear of the key slots)
NAME_DEPTH     = 0.6
CORNER_R       = 2.0
TOP_CHAMFER    = 0.8
BOTTOM_CHAMFER = 0.4
HOLE_CHAMFER   = 0.8

# ---------------- emblem style ----------------
STRAND_W   = 1.0    # strand groove width
RUNG_W     = 0.8    # rung groove width
CROSS_GAP  = 0.7    # plastic left where one strand passes under the other
OUTLINE_W  = 0.8    # badge outline groove width
CAP_H      = 4.0    # badge text height
BADGE_H    = 6.6
BADGE_PAD  = 1.8
GAP        = 1.4    # between badge and symbol
AMP        = 3.0    # helix half-height
ENGRAVE    = 0.8    # depth of everything cut into the top

# ---------------- keys / sockets (identical to the oil holders) ----------------
GRID, KEY_NECK, KEY_FLARE, KEY_HALF = 10.0, 5.0, 8.0, 3.5
KEY_HEIGHT, SOCKET_EXTRA_H, KEY_FIT = 10.0, 0.5, 0.15
MIN_GAP = 2.0       # plastic kept between a socket and a tube hole
N_KEYS = 12

FONT = FontProperties(family="DejaVu Sans", weight="bold")
SEG = 96


# ======================= 2D emblem geometry (shapely) =======================
def text_shape(s, cap_h):
    """Text outline with the given cap height, centred on (0, 0)."""
    ref = TextPath((0, 0), "H", size=10, prop=FONT).get_extents()
    k = cap_h / ref.height
    g = Polygon()
    for ring in TextPath((0, 0), s, size=10, prop=FONT).to_polygons():
        if len(ring) >= 3:
            g = g.symmetric_difference(Polygon(ring))
    g = affinity.scale(g, k, k, origin=(0, 0))
    x0, y0, x1, y1 = g.bounds
    return affinity.translate(g, -(x0 + x1) / 2, -(y0 + y1) / 2)


def rung_positions(turns, length):
    half = length / (2 * turns)
    for k in range(int(round(2 * turns))):
        for t in (0.3, 0.7):
            yield -length / 2 + (k + t) * half, np.pi * (k + t)


def double_helix(turns, length, amp=AMP):
    x = np.linspace(-length / 2, length / 2, 800)
    phase = 2 * np.pi * turns * (x + length / 2) / length
    lines = [np.c_[x, s * amp * np.sin(phase)] for s in (1, -1)]
    strands = [LineString(p).buffer(STRAND_W / 2) for p in lines]
    half = length / (2 * turns)
    for i in range(1, int(round(2 * turns))):          # over/under at each crossing
        xk = -length / 2 + i * half
        over, under = (0, 1) if i % 2 else (1, 0)
        near = lines[over][np.abs(lines[over][:, 0] - xk) < 1.6]
        strands[under] = strands[under].difference(LineString(near).buffer(STRAND_W / 2 + CROSS_GAP))
    rungs = [LineString([(xr, -amp * abs(np.sin(ph)) + 0.3), (xr, amp * abs(np.sin(ph)) - 0.3)])
             .buffer(RUNG_W / 2, cap_style="flat") for xr, ph in rung_positions(turns, length)]
    return unary_union(strands + rungs)


def single_strand(turns, length, amp=AMP):
    """One strand with half-rungs pointing toward the missing partner."""
    x = np.linspace(-length / 2, length / 2, 800)
    phase = 2 * np.pi * turns * (x + length / 2) / length
    strand = LineString(np.c_[x, amp * np.sin(phase)]).buffer(STRAND_W / 2)
    half_rungs = [LineString([(xr, amp * np.sin(ph)), (xr, 0.1 * amp * np.sin(ph))])
                  .buffer(RUNG_W / 2, cap_style="flat") for xr, ph in rung_positions(turns, length)]
    return unary_union([strand] + half_rungs)


def badge(code, hs):
    """Outlined pill with the code; hs adds a solid tab with knocked-out HS."""
    parts = [("code", text_shape(code, CAP_H))]
    if hs:
        parts.append(("hs", text_shape("HS", CAP_H)))
    widths = [t.bounds[2] - t.bounds[0] + 2 * BADGE_PAD for _, t in parts]
    total = sum(widths)
    r = BADGE_H / 2
    outer = LineString([(-total / 2 + r, 0), (total / 2 - r, 0)]).buffer(r)
    cut = outer.difference(outer.buffer(-OUTLINE_W))
    x = -total / 2
    for (kind, t), w in zip(parts, widths):
        cx = x + w / 2
        if kind == "code":
            cut = cut.union(affinity.translate(t, cx + (0.3 if len(parts) == 1 else 0.6), 0))
        else:
            seg = outer.intersection(box(x, -BADGE_H, x + w + 1, BADGE_H))
            cut = cut.union(seg.difference(affinity.translate(t, cx - 0.6, 0)))
        x += w
    return cut


def lockup(a):
    sym = double_helix(a["turns"], a["length"]) if a["symbol"] == "dsDNA" else single_strand(a["turns"], a["length"])
    b = badge(a["badge"], a["hs"])
    sym_h = 2 * AMP + STRAND_W
    total = BADGE_H + GAP + sym_h
    return unary_union([affinity.translate(b, 0, total / 2 - BADGE_H / 2),
                        affinity.translate(sym, 0, -total / 2 + sym_h / 2)])


def top_artwork(a):
    parts = [affinity.translate(lockup(a), WIDTH / 2, EMBLEM_Y)]
    for x, label in zip(TUBE_X, TUBE_LABELS):
        parts.append(affinity.translate(text_shape(label, LABEL_CAP), x, LABEL_Y))
    return unary_union(parts)


def to_cs(geom):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    contours = []
    for p in polys:
        contours.append(np.array(p.exterior.coords)[:-1])
        contours += [np.array(i.coords)[:-1] for i in p.interiors]
    return CrossSection(contours, FillRule.EvenOdd)


# ======================= key sockets =======================
def socket_outline(side, p, grow=0.0):
    """Half-dovetail socket polygon in module XY for a side and a grid position."""
    e, s = 1.0, (KEY_FLARE - KEY_NECK) / 2 / KEY_HALF
    uv = [(p - KEY_NECK / 2 + s * e, -e), (p + KEY_NECK / 2 - s * e, -e),
          (p + KEY_FLARE / 2, KEY_HALF), (p - KEY_FLARE / 2, KEY_HALF)]
    m = {"front": lambda u, v: (u, v), "back": lambda u, v: (u, DEPTH - v),
         "left": lambda u, v: (v, u), "right": lambda u, v: (WIDTH - v, u)}[side]
    poly = Polygon([m(u, v) for u, v in uv])
    return poly.buffer(grow, join_style="mitre") if grow else poly


def socket_layout():
    holes = [Point(x, TUBE_Y) for x in TUBE_X]
    r = (TUBE_DIAM + TUBE_CLEAR) / 2
    out = []
    for side, length in (("front", WIDTH), ("back", WIDTH), ("left", DEPTH), ("right", DEPTH)):
        for k in range(1, int(round(length / GRID))):
            poly = socket_outline(side, k * GRID)
            if min(poly.distance(h) for h in holes) - r >= MIN_GAP:
                out.append((side, k * GRID))
    return out


def sockets():
    parts = []
    for side, p in socket_layout():
        body = Manifold.extrude(to_cs(socket_outline(side, p)), KEY_HEIGHT + SOCKET_EXTRA_H + 1)
        mouth = Manifold.extrude(to_cs(socket_outline(side, p, 0.3)), 1.4)  # flared entry
        parts += [body.translate([0, 0, -1]), mouth.translate([0, 0, -1])]
    return Manifold.batch_boolean(parts, OpType.Add)


# ======================= module =======================
def slab(cs, z):
    return Manifold.extrude(cs, 0.01).translate([0, 0, z])


def body():
    base = CrossSection.square([WIDTH - 2 * CORNER_R, DEPTH - 2 * CORNER_R]).translate([CORNER_R, CORNER_R])
    base = base.offset(CORNER_R, JoinType.Round, circular_segments=48)
    bottom = Manifold.batch_hull([slab(base.offset(-BOTTOM_CHAMFER, JoinType.Round), 0), slab(base, BOTTOM_CHAMFER)])
    mid = Manifold.extrude(base, HEIGHT - BOTTOM_CHAMFER - TOP_CHAMFER).translate([0, 0, BOTTOM_CHAMFER])
    top = Manifold.batch_hull([slab(base, HEIGHT - TOP_CHAMFER - 0.01),
                               slab(base.offset(-TOP_CHAMFER, JoinType.Round), HEIGHT - 0.01)])
    return bottom + mid + top


def side_names(name):
    """Assay name cut into all four side faces, upright and readable from outside each face."""
    cs = to_cs(text_shape(name, NAME_CAP))
    cut = Manifold.extrude(cs, NAME_DEPTH + 0.5).translate([0, 0, -NAME_DEPTH])
    cut = cut.rotate([90, 0, 0]).translate([0, 0, NAME_Z])  # faces -Y, reads along +X, cuts toward +Y
    faces = [(0, WIDTH / 2, 0), (180, WIDTH / 2, DEPTH), (-90, 0, DEPTH / 2), (90, WIDTH, DEPTH / 2)]
    return Manifold.batch_boolean([cut.rotate([0, 0, ang]).translate([x, y, 0]) for ang, x, y in faces], OpType.Add)


def make_module(name, a):
    m = body()
    r = (TUBE_DIAM + TUBE_CLEAR) / 2
    for x in TUBE_X:
        m -= Manifold.cylinder(TUBE_DEPTH + 1, r, r, SEG).translate([x, TUBE_Y, FLOOR])
        m -= Manifold.cylinder(HOLE_CHAMFER + 0.01, r, r + HOLE_CHAMFER + 0.01, SEG).translate(
            [x, TUBE_Y, HEIGHT - HOLE_CHAMFER])
    art = Manifold.extrude(to_cs(top_artwork(a)), ENGRAVE + 1).translate([0, 0, HEIGHT - ENGRAVE])
    return m - art - side_names(name) - sockets()


def make_key():
    d, n, w = KEY_HALF, KEY_NECK / 2, KEY_FLARE / 2
    cs = CrossSection([[(-d, -w), (0, -n), (d, -w), (d, w), (0, n), (-d, w)]]).offset(-KEY_FIT, JoinType.Miter)
    ch = 0.5
    s = 1 - 2 * ch / KEY_FLARE
    bot = Manifold.extrude(cs.scale([s, s]), ch, scale_top=(1 / s, 1 / s))
    mid = Manifold.extrude(cs, KEY_HEIGHT - 2 * ch).translate([0, 0, ch])
    top = Manifold.extrude(cs, ch, scale_top=(s, s)).translate([0, 0, KEY_HEIGHT - ch])
    return bot + mid + top


def export(m, path):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts)
    tm.export(path)
    return tm.is_watertight


if __name__ == "__main__":
    for name, a in ASSAYS.items():
        ok = export(make_module(name, a), f"tapestation_{name}.stl")
        print(f"{name}: {WIDTH:.0f} x {DEPTH:.0f} x {HEIGHT:.0f} mm, watertight={ok}")
    key = make_key()
    plate = Manifold.batch_boolean([key.translate([(i % 4) * 12.0, (i // 4) * 12.0, 0]) for i in range(N_KEYS)], OpType.Add)
    print("keys:", export(plate, "tapestation_keys.stl"))
    print("sockets:", socket_layout())


def preview(path="tapestation_preview.png"):
    """Top, front and right-side views of each module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle, Circle
    from matplotlib.path import Path

    def draw(ax, contours, outline):
        ax.add_patch(Rectangle((0, 0), *outline, fc="#2c2c2a", ec="none"))
        verts, codes = [], []
        for c in contours:
            verts += c.tolist() + [c[0].tolist()]
            codes += [Path.MOVETO] + [Path.LINETO] * (len(c) - 1) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), fc="#e9e6dd", ec="#9a978f", lw=0.4))

    n = len(ASSAYS)
    fig, axs = plt.subplots(3, n, figsize=(3.4 * n, 8.6), dpi=150,
                            gridspec_kw=dict(height_ratios=[DEPTH, HEIGHT + 4, HEIGHT + 4]))
    for i, (name, a) in enumerate(ASSAYS.items()):
        m = make_module(name, a)
        draw(axs[0][i], m.slice(HEIGHT - 0.4).to_polygons(), (WIDTH, DEPTH))
        for x in TUBE_X:
            axs[0][i].add_patch(Circle((x, TUBE_Y), (TUBE_DIAM + TUBE_CLEAR) / 2, fc="white", ec="#9a978f", lw=0.4))
        draw(axs[1][i], m.rotate([-90, 0, 0]).slice(-0.3).to_polygons(), (WIDTH, HEIGHT))
        side = [c[:, ::-1] for c in m.rotate([0, 90, 0]).slice(-(WIDTH - 0.3)).to_polygons()]
        draw(axs[2][i], side, (DEPTH, HEIGHT))
        axs[0][i].set_title(name, fontsize=10)
        for ax, (w, h) in zip(axs[:, i], ((WIDTH, DEPTH), (WIDTH, HEIGHT), (DEPTH, HEIGHT))):
            ax.set_xlim(-1, DEPTH + 1); ax.set_ylim(-1, h + 1); ax.set_aspect("equal"); ax.axis("off")
    for row, label, h in ((0, "top", DEPTH), (1, "front", HEIGHT), (2, "side", HEIGHT)):
        axs[row][0].text(-3, h / 2, label, rotation=90, va="center", ha="right", fontsize=8, color="#6b6963")
    plt.tight_layout()
    plt.savefig(path, facecolor="white")


if __name__ == "__main__":
    preview()
