"""
Modular essential-oil bottle holders that clip together with bowtie keys.

Each module holds one bottle. Modules sit on a 10 mm grid, and every side has
dovetail sockets on the underside at grid positions, so any module can join
any other along any side. Drop a bowtie key into the facing sockets from below.

Edit BOTTLES (measure: wrap a paper strip around the bottle, measure its
length, divide by 3.14), then run:
    pip install trimesh manifold3d
    python essential_oil_modules.py
All units are mm.
"""
import math
import trimesh
from manifold3d import Manifold, CrossSection, JoinType, OpType

# ---------------- BOTTLES ----------------
# diam  = bottle outer diameter (typical values; replace with your measurements)
# depth = how deep the bottle sits (about 1/3 of bottle height)
BOTTLES = {
    "10ml":  dict(diam=26.0, depth=18),
    "15ml":  dict(diam=29.0, depth=20),
    "30ml":  dict(diam=33.0, depth=24),
    "100ml": dict(diam=45.0, depth=32),
    "118ml": dict(diam=49.0, depth=35),
}
CLEARANCE      = 1.0    # added to bottle diameter (bottle slides in easily)

# ---------------- MODULE ----------------
GRID           = 10.0   # module sizes and socket positions snap to this
MIN_WALL       = 3.0    # thinnest wall beside the pocket
FLOOR          = 2.0    # solid floor under the bottle
CORNER_R       = 2.0    # outer corner rounding
POCKET_CHAMFER = 1.0    # lead-in at pocket top
TOP_CHAMFER    = 0.8    # outer top edge
BOTTOM_CHAMFER = 0.4    # outer bottom edge (counters elephant's foot)

# ---------------- KEYS / SOCKETS ----------------
KEY_NECK       = 5.0    # width where the two modules meet
KEY_FLARE      = 8.0    # width at each end of the bowtie
KEY_HALF       = 3.5    # how far the key reaches into each module
KEY_HEIGHT     = 10.0
SOCKET_EXTRA_H = 0.5    # socket is this much taller so keys sit recessed
KEY_FIT        = 0.15   # gap per side between key and socket (raise if tight)
MIN_GAP        = 2.0    # min plastic between a socket and the pocket
N_KEYS         = 12     # keys on the key plate

SEG = 160
EPS = 0.01


def rounded_square(size, r):
    cs = CrossSection.square([size - 2 * r, size - 2 * r]).translate([r, r])
    return cs.offset(r, JoinType.Round, circular_segments=64)


def slab(cs, z):
    return Manifold.extrude(cs, EPS).translate([0, 0, z])


def module_body(size, height):
    base = rounded_square(size, CORNER_R)
    bottom = Manifold.batch_hull([slab(base.offset(-BOTTOM_CHAMFER, JoinType.Round), 0),
                                  slab(base, BOTTOM_CHAMFER)])
    mid = Manifold.extrude(base, height - BOTTOM_CHAMFER - TOP_CHAMFER).translate([0, 0, BOTTOM_CHAMFER])
    top = Manifold.batch_hull([slab(base, height - TOP_CHAMFER - EPS),
                               slab(base.offset(-TOP_CHAMFER, JoinType.Round), height - EPS)])
    return bottom + mid + top


def socket_positions(size, pocket_r):
    """Grid positions along one side where a socket fits without breaking into the pocket."""
    c = size / 2
    ok = []
    for k in range(1, int(round(size / GRID))):
        p = k * GRID
        nearest_x = min(max(c, p - KEY_FLARE / 2), p + KEY_FLARE / 2)
        gap = math.hypot(c - nearest_x, c - KEY_HALF) - pocket_r
        if gap >= MIN_GAP:
            ok.append(p)
    return ok


def socket(p):
    """Half-dovetail socket on the y=0 side, centred at x=p, open at the bottom."""
    e = 1.0
    slope = (KEY_FLARE - KEY_NECK) / 2 / KEY_HALF
    poly = [(p - KEY_NECK / 2 + slope * e, -e), (p + KEY_NECK / 2 - slope * e, -e),
            (p + KEY_FLARE / 2, KEY_HALF), (p - KEY_FLARE / 2, KEY_HALF)]
    cs = CrossSection([poly])
    s = Manifold.extrude(cs, KEY_HEIGHT + SOCKET_EXTRA_H + 1).translate([0, 0, -1])
    mouth = Manifold.extrude(cs.offset(0.3, JoinType.Miter), 1.4).translate([0, 0, -1])  # flared entry
    return s + mouth


def make_module(diam, depth):
    pocket_d = diam + CLEARANCE
    r = pocket_d / 2
    size = math.ceil((pocket_d + 2 * MIN_WALL) / GRID) * GRID
    while len(socket_positions(size, r)) < 2:  # grow until each side has room for 2 sockets
        size += GRID
    height = depth + FLOOR
    c = size / 2

    body = module_body(size, height)
    pocket = Manifold.cylinder(depth + 1, r, r, SEG).translate([c, c, FLOOR])
    cham = Manifold.cylinder(POCKET_CHAMFER + EPS, r, r + POCKET_CHAMFER + EPS, SEG).translate(
        [c, c, height - POCKET_CHAMFER])
    body = body - pocket - cham

    positions = socket_positions(size, r)
    side = Manifold.batch_boolean([socket(p) for p in positions], OpType.Add)
    for ang in (0, 90, 180, 270):
        body = body - side.translate([-c, -c, 0]).rotate([0, 0, ang]).translate([c, c, 0])
    return body, size, height, positions


def key_shape():
    d, n, w = KEY_HALF, KEY_NECK / 2, KEY_FLARE / 2
    cs = CrossSection([[(-d, -w), (0, -n), (d, -w), (d, w), (0, n), (-d, w)]])
    return cs.offset(-KEY_FIT, JoinType.Miter)


def make_key():
    cs = key_shape()
    ch = 0.5
    s = 1 - 2 * ch / KEY_FLARE
    bot = Manifold.extrude(cs.scale([s, s]), ch, scale_top=(1 / s, 1 / s))
    mid = Manifold.extrude(cs, KEY_HEIGHT - 2 * ch).translate([0, 0, ch])
    top = Manifold.extrude(cs, ch, scale_top=(s, s)).translate([0, 0, KEY_HEIGHT - ch])
    return bot + mid + top


def export(m, path):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts)
    tm.export(path)
    return tm.is_watertight


if __name__ == "__main__":
    for name, b in BOTTLES.items():
        mod, size, height, pos = make_module(b["diam"], b["depth"])
        wt = export(mod, f"eo_holder_{name}.stl")
        print(f"{name:>6}: {size:.0f} x {size:.0f} x {height:.0f} mm, pocket {b['diam'] + CLEARANCE:.1f} mm, "
              f"{len(pos)} sockets/side, watertight={wt}")

    key = make_key()
    cols = 4
    plate = Manifold.batch_boolean(
        [key.translate([(i % cols) * 12.0, (i // cols) * 12.0, 0]) for i in range(N_KEYS)], OpType.Add)
    print(f"  keys: {N_KEYS} keys, watertight={export(plate, 'eo_keys.stl')}")

    rings, x = [], 0.0
    for b in BOTTLES.values():
        d = b["diam"] + CLEARANCE
        x += d / 2 + 2
        ring = Manifold.cylinder(4, d / 2 + 2, d / 2 + 2, SEG) - Manifold.cylinder(6, d / 2, d / 2, SEG).translate([0, 0, -1])
        rings.append(ring.translate([x, 0, 0]))
        x += d / 2 + 2 + 4
    print(f" rings: 5 fit-test rings, watertight={export(Manifold.batch_boolean(rings, OpType.Add), 'eo_fit_test_rings.stl')}")
