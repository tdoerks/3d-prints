"""
Chess-style fantasy salt & pepper shakers: the KING (salt) and the QUEEN (pepper),
plus a two-square chessboard stand (queen on d1, king on e1). Both shakers are
hollow, fully detailed figures that fill through the same threaded plug in the base.

    pip install trimesh manifold3d numpy
    python chess_shakers.py

Prints upright with no supports. Units: mm. Front of the figure faces -Y.
"""
import math
import numpy as np
import trimesh
from manifold3d import Manifold, CrossSection, FillRule, OpType, JoinType

# ---------------- general ----------------
SEG = 160            # smoothness of turned surfaces
BEAD_SEG = 20        # smoothness of small beads / pearls

# ---------------- salt holes ----------------
HOLE_COUNT = 8
HOLE_D     = 1.4     # salt; pepper will use larger holes
HOLE_R_POS = 5.2     # radius of the hole ring on the crown dome

# ---------------- fill plug (thread) ----------------
THREAD_RC    = 11.2  # plug thread: eccentric-circle radius
THREAD_E     = 0.7   # eccentricity -> thread depth 1.4 mm
THREAD_PITCH = 3.2
THREAD_CLEAR = 0.45  # radial clearance, hole vs plug
BASE_SOLID   = 8.0   # base thickness under the salt chamber
CB_R, CB_D   = 15.0, 1.6    # counterbore for the plug flange (radius, depth)
FLANGE_CLEAR = 0.3

# ---------------- outer profile (radius, height) ----------------
PLINTH = [(0, 0), (22.8, 0), (23.6, 0.8), (23.6, 5.2), (22.4, 6.4), (22.4, 7.6), (21.2, 8.4), (20.6, 9.4)]
SKIRT = [(20.2, 10.0), (18.6, 16.0), (17.0, 22.0), (15.6, 28.0), (14.3, 34.0), (13.2, 39.5), (12.4, 44.0)]
CHEST = [(12.0, 45.5), (12.0, 47.0), (12.5, 51.0), (12.9, 55.0), (12.9, 58.0)]
SHOULDERS = [(15.6, 62.6), (14.0, 65.2), (10.4, 66.6), (9.0, 67.4), (8.6, 69.0)]
CROWN_Z0, CROWN_Z1, CROWN_R = 71.2, 78.0, 10.8
DOME_R, DOME_H = 9.6, 4.2

# ---------------- salt chamber profile (radius, height) ----------------
CHAMBER = [(0, BASE_SOLID), (17.6, BASE_SOLID), (17.6, 10.0), (16.0, 16), (14.4, 22), (13.0, 28), (11.7, 34),
           (10.6, 39.5), (9.8, 44), (9.4, 45.5), (9.4, 50), (10.0, 55), (10.0, 63.6), (6.2, 68.0), (6.2, 70.0),
           (8.2, 72.0), (8.2, 74.6), (3.2, 79.6), (0, 79.6)]

FRONT = -90.0  # degrees; the figure faces -Y


# ======================= primitives =======================
def revolve(points):
    return Manifold.revolve(CrossSection([np.array(points, float)], FillRule.EvenOdd), SEG)


def sphere(r, seg=BEAD_SEG):
    return Manifold.sphere(r, seg)


def torus(R, r, seg_major=SEG, seg_minor=24):
    return Manifold.revolve(CrossSection.circle(r, seg_minor).translate([R, 0]), seg_major)


def capsule(p0, r0, p1, r1):
    return Manifold.batch_hull([sphere(r0).translate(p0), sphere(r1).translate(p1)])


def union(parts):
    parts = [p for p in parts if p is not None]
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


def polar(r, ang_deg, z):
    a = math.radians(ang_deg)
    return [r * math.cos(a), r * math.sin(a), z]


def on_surface(obj, r, ang_deg, z):
    """Place obj (built with its outward direction along +X) on the surface at (r, angle, z)."""
    return obj.rotate([0, 0, ang_deg]).translate(polar(r, ang_deg, z))


def radial_relief(poly2d, depth, taper=1.0):
    """2D shape (x = tangential, y = vertical) extruded outward along +X."""
    cs = CrossSection([np.array(poly2d, float)], FillRule.EvenOdd)
    m = Manifold.extrude(cs, depth, scale_top=(taper, taper))
    return m.rotate([90, 0, 0]).rotate([0, 0, 90])


def outer_r(z):
    """Outer radius of the smooth body at height z (skirt and chest)."""
    pts = SKIRT + CHEST
    zs = [p[1] for p in pts]
    rs = [p[0] for p in pts]
    return float(np.interp(z, zs, rs))


def ring(obj_fn, n, R, z, phase=0.0):
    return union([on_surface(obj_fn(k), R, phase + 360.0 * k / n, z) for k in range(n)])


# ======================= relief helper check =======================
def _radial_axis_check():
    # a thin plate 2 wide (tangential) x 4 tall, extruded 1 outward, placed at angle 0 must span x in [0,1]
    m = radial_relief([(-1, -2), (1, -2), (1, 2), (-1, 2)], 1.0)
    return [round(v, 3) for v in m.bounding_box()]


# ======================= king parts =======================
def body_profile():
    dome = [(DOME_R * math.cos(t), CROWN_Z1 + DOME_H * math.sin(t)) for t in np.linspace(0, math.pi / 2, 24)]
    crown = [(8.6, 69.0), (CROWN_R, CROWN_Z0), (CROWN_R, CROWN_Z1), (DOME_R, CROWN_Z1)] + dome[1:]
    return revolve(PLINTH + SKIRT + CHEST + SHOULDERS + crown[1:] + [(0, CROWN_Z1 + DOME_H)])


def plinth_details():
    # engraved lozenges around the lower drum are cut later; here the raised bead ring and hem
    beads = ring(lambda k: sphere(0.8), 84, 22.4, 7.0)
    hem = torus(20.2, 1.1).translate([0, 0, 10.2])
    return union([beads, hem])


def plinth_engraving():
    lozenge = radial_relief([(0, -1.6), (1.6, 0), (0, 1.6), (-1.6, 0)], 1.4)
    return ring(lambda k: lozenge, 36, 23.6 - 0.6, 3.0)


def robe_folds():
    folds = []
    for side in (-1, 1):
        for k in range(4):
            ang = FRONT + side * (20 + 12.5 * k)
            big = k % 2 == 0
            zb, zt = 11.6, 43.0
            rb, rt = (1.5, 0.55) if big else (1.1, 0.45)
            p0 = polar(outer_r(zb) - 0.25, ang, zb)
            p1 = polar(outer_r(zt) - 0.15, ang, zt)
            folds.append(capsule(p0, rb, p1, rt))
    return union(folds)


def front_trim():
    """Raised band down the front of the robe with round studs."""
    z0, z1, lift, half_w = 10.4, 44.2, 0.8, 2.75
    prof = [(0, z0)] + [(outer_r(z) + lift, z) for z in np.linspace(z0, z1, 24)] + [(0, z1)]
    band = revolve(prof) ^ Manifold.cube([2 * half_w, 40, z1 - z0]).translate([-half_w, -40, z0])
    studs = [sphere(0.9).translate(polar(outer_r(z) + lift, FRONT, z)) for z in np.arange(14.0, 42.0, 5.2)]
    return union([band] + studs)


CAPE_EDGE = (8.8, 4.0, 58.0, -7.0)   # (z_low, y_low, z_high, y_high): the cape's front edge line


def cape_edge_y(z):
    z0, y0, z1, y1 = CAPE_EDGE
    return y0 + (y1 - y0) * (z - z0) / (z1 - z0)


def mantle():
    """Cape over the back, its front edges slanting back toward the hem, with folds and rolled hems."""
    lift = 1.4
    z0, z1 = 8.8, 58.5
    prof = [(0, z0)] + [(outer_r(z) + lift, z) for z in np.linspace(z0, z1, 40)] + [(0, z1)]
    b = (CAPE_EDGE[3] - CAPE_EDGE[1]) / (CAPE_EDGE[2] - CAPE_EDGE[0])
    a = CAPE_EDGE[1] - b * CAPE_EDGE[0]
    cape = revolve(prof).trim_by_plane([0, 1, -b], a / math.sqrt(1 + b * b))   # keeps y >= a + b*z
    parts = [cape]
    for k in range(7):
        ang = 30 + 20 * k
        zb, zt = 11.5, 55.0
        parts.append(capsule(polar(outer_r(zb) + lift - 0.3, ang, zb), 1.4,
                             polar(outer_r(zt) + lift - 0.3, ang, zt), 0.5))
    for sx in (-1, 1):
        pts = []
        for z in np.linspace(z0 + 0.9, z1 - 1.0, 16):
            rc = outer_r(z) + 0.75
            y = cape_edge_y(z) + 0.2
            pts.append([sx * math.sqrt(rc ** 2 - y ** 2), y, z])
        parts += [capsule(pts[i], 0.9, pts[i + 1], 0.9) for i in range(len(pts) - 1)]
    return union(parts)


def belt():
    band = torus(12.3, 1.3).translate([0, 0, 45.6]) ^ Manifold.cube([40, 16, 10]).translate([-20, -20, 40])
    frame = Manifold.cube([5.2, 1.2, 4.4], center=True) - Manifold.cube([3.0, 3.0, 2.2], center=True)
    prong = Manifold.cube([0.7, 1.0, 2.6], center=True)
    buckle = union([frame, prong]).translate(polar(13.2, FRONT, 45.6))
    return union([band, buckle])


def necklace():
    beads = []
    pts = []
    for phi in np.linspace(-60, 60, 400):
        z = 54.3 + 4.5 * (phi / 60.0) ** 2
        pts.append(np.array(polar(outer_r(z) + 0.2, FRONT + phi, z)))
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    for s in np.arange(0.8, dist[-1], 1.45):
        beads.append(sphere(0.65, 16).translate(pts[int(np.searchsorted(dist, s))].tolist()))
    # medallion with an eight-point star
    z_m = 51.0
    disc = Manifold.cylinder(1.4, 2.8, 2.8, 48).rotate([90, 0, 0]).translate([0, -(outer_r(z_m) - 0.3), z_m])
    rim = torus(2.8, 0.35, 48, 12).rotate([90, 0, 0]).translate([0, -(outer_r(z_m) + 1.1), z_m])
    star_pts = []
    for i in range(16):
        rr = 1.9 if i % 2 == 0 else 0.85
        a = math.pi / 2 + i * math.pi / 8
        star_pts.append((rr * math.cos(a), rr * math.sin(a)))
    star = Manifold.extrude(CrossSection([np.array(star_pts)], FillRule.EvenOdd), 0.5)
    star = star.rotate([90, 0, 0]).translate([0, -(outer_r(z_m) + 1.1), z_m])
    return union(beads + [disc, rim, star])


def ermine_collar():
    roll = torus(14.3, 2.1).translate([0, 0, 63.6])
    return roll


def ermine_spots():
    spot = capsule([0, 0, -0.6], 0.45, [0, 0, 0.6], 0.32)
    row1 = ring(lambda k: spot, 16, 16.0, 64.8, phase=FRONT + 11.25)
    row2 = ring(lambda k: sphere(0.42), 16, 15.02, 65.57, phase=FRONT)
    return union([row1, row2])


def shoulder_chain():
    return ring(lambda k: sphere(0.55, 16), 62, 11.8, 66.1)


def crown_details():
    parts = []
    # beaded rims
    parts.append(ring(lambda k: sphere(0.7), 46, CROWN_R + 0.05, CROWN_Z0 + 0.7))
    parts.append(ring(lambda k: sphere(0.7), 46, CROWN_R + 0.05, CROWN_Z1 - 0.6))
    zc = (CROWN_Z0 + CROWN_Z1) / 2 + 0.05
    for k in range(8):
        ang = FRONT + 45 * k
        tall = k % 2 == 0
        # gem on the band
        if tall:
            gem = sphere(1.5, 28).scale([0.55, 1, 1])
        else:
            rh = [(0, -1.9), (1.5, 0), (0, 1.9), (-1.5, 0)]
            base = Manifold.extrude(CrossSection([np.array(rh)], FillRule.EvenOdd), 0.01)
            top = Manifold.extrude(CrossSection([np.array(rh) * 0.45], FillRule.EvenOdd), 0.01).translate([0, 0, 0.9])
            gem = Manifold.batch_hull([base, top]).rotate([90, 0, 0]).rotate([0, 0, 90])
        parts.append(on_surface(gem, CROWN_R - 0.1, ang, zc))
        # crown point with a pearl
        h = 6.6 if tall else 4.4
        w = 4.2 if tall else 3.4
        pt = [(-w / 2, 0), (w / 2, 0), (w / 2 * 0.8, h * 0.55), (0.35, h), (-0.35, h), (-w / 2 * 0.8, h * 0.55)]
        spike = Manifold.extrude(CrossSection([np.array(pt)], FillRule.EvenOdd), 1.8)
        spike = spike.rotate([90, 0, 0]).rotate([0, 0, 90])  # thickness along X (radial), shape in Y-Z
        spike = spike.translate([-1.8, 0, 0])
        spike = union([spike, sphere(1.0).translate([-0.9, 0, h + 0.55])])
        if tall:
            lobe = sphere(0.75)
            spike = union([spike, lobe.translate([-0.9, w / 2 * 0.8 + 0.35, h * 0.52]),
                           lobe.translate([-0.9, -(w / 2 * 0.8 + 0.35), h * 0.52])])
        parts.append(on_surface(spike, CROWN_R, ang, CROWN_Z1 - 0.2))
    # arches over the cap, springing from the points and meeting under the orb
    cap = sphere(1.0, 96).scale([DOME_R + 0.9, DOME_R + 0.9, DOME_H + 0.9]).translate([0, 0, CROWN_Z1])
    slab = Manifold.cube([1.6, 40, 20]).translate([-0.8, -20, CROWN_Z1])
    for k in range(4):
        parts.append(cap ^ slab.rotate([0, 0, 45 * k + FRONT + 90]))
    return union(parts)


def finial():
    top = CROWN_Z1 + DOME_H
    stem = Manifold.cylinder(2.6, 2.4, 1.5, 48).translate([0, 0, top - 0.8])
    orb_z = top + 3.8
    orb = sphere(2.8, 48).translate([0, 0, orb_z])
    band = torus(2.8, 0.4, 48, 12).translate([0, 0, orb_z])
    meridian = (torus(2.8, 0.4, 48, 12).rotate([90, 0, 0]) ^ Manifold.cube([10, 10, 5]).translate([-5, -5, 0])
                ).rotate([0, 0, FRONT + 90]).translate([0, 0, orb_z])
    cz = orb_z + 2.0
    cross2d = [
        [(-1.0, 0.0), (1.0, 0.0), (1.0, 5.4), (0.0, 6.8), (-1.0, 5.4)],                 # upright
        [(0.8, 1.4), (3.0, 3.6), (3.0, 4.6), (0.8, 4.0)],                               # right arm
        [(-0.8, 1.4), (-0.8, 4.0), (-3.0, 4.6), (-3.0, 3.6)],                           # left arm
    ]
    cross = union([Manifold.extrude(CrossSection([np.array(p)], FillRule.EvenOdd), 2.2) for p in cross2d])
    cross = cross.translate([0, 0, -1.1]).rotate([90, 0, 0]).rotate([0, 0, FRONT + 90]).translate([0, 0, cz])
    return union([stem, orb, band, meridian, cross])


def salt_holes():
    holes = []
    for k in range(HOLE_COUNT):
        ang = FRONT + 22.5 + 360.0 * k / HOLE_COUNT
        holes.append(Manifold.cylinder(10, HOLE_D / 2, HOLE_D / 2, 24).translate(polar(HOLE_R_POS, ang, 75.0)))
    return union(holes)


def thread(rc, length):
    turns = length / THREAD_PITCH
    cs = CrossSection.circle(rc, 96).translate([THREAD_E, 0])
    return Manifold.extrude(cs, length, n_divisions=int(turns * 72), twist_degrees=360.0 * turns)


def fill_opening():
    counterbore = Manifold.cylinder(CB_D + 0.01, CB_R, CB_R, 128).translate([0, 0, -0.01])
    hole = thread(THREAD_RC + THREAD_CLEAR, BASE_SOLID + 1.0).translate([0, 0, CB_D - 0.2])
    lead = Manifold.cylinder(0.8, 12.9, 12.1, 96).translate([0, 0, CB_D - 0.01])
    return union([counterbore, hole, lead])


def make_king():
    solid = union([body_profile(), plinth_details(), robe_folds(), front_trim(), mantle(), belt(),
                   necklace(), ermine_collar(), shoulder_chain(), crown_details(), finial()])
    cuts = union([revolve(CHAMBER), plinth_engraving(), ermine_spots(), salt_holes(), fill_opening()])
    return solid - cuts


def make_plug():
    flange = Manifold.cylinder(CB_D, CB_R - FLANGE_CLEAR, CB_R - FLANGE_CLEAR, 128)
    boss_len = BASE_SOLID - CB_D - 0.4
    boss = thread(THREAD_RC, boss_len).translate([0, 0, CB_D - 0.01])
    top = CB_D + boss_len
    edge = Manifold.cylinder(1.0, 13, 13, 96).translate([0, 0, top - 1.0]) - \
        Manifold.cylinder(1.0, 12.0, 10.9, 96).translate([0, 0, top - 1.0])      # chamfer the leading edge
    slot = Manifold.cube([16, 2.4, 1.2]).translate([-8, -1.2, -0.01])             # coin slot
    return union([flange, boss - edge]) - slot


# ======================= QUEEN (pepper) =======================
Q_HOLE_COUNT, Q_HOLE_D, Q_HOLE_R_POS = 7, 2.2, 4.9     # pepper needs bigger holes than salt

Q_SKIRT = [(20.2, 10.0), (18.0, 14.5), (16.1, 19.0), (14.6, 23.5), (13.3, 28.0), (12.3, 32.5), (11.5, 37.0),
           (10.9, 41.5), (10.5, 46.0)]
Q_BODICE = [(10.3, 47.5), (10.3, 49.0), (10.8, 52.0), (11.4, 55.0), (11.6, 57.0), (11.4, 59.0)]
Q_SHOULDERS = [(11.8, 61.0), (11.2, 62.8), (9.5, 64.2), (7.5, 65.0), (6.9, 66.4), (6.7, 67.4)]
Q_CROWN_Z0, Q_CROWN_Z1, Q_CROWN_R = 69.3, 74.3, 8.6
Q_DOME_R, Q_DOME_H = 7.6, 3.4
Q_CHAMBER = [(0, BASE_SOLID), (17.6, BASE_SOLID), (17.6, 10.0), (15.5, 14.5), (13.6, 19.0), (12.1, 23.5),
             (10.8, 28.0), (9.8, 32.5), (9.0, 37.0), (8.4, 41.5), (8.0, 46.0), (7.8, 47.5), (7.8, 49.0),
             (8.3, 52.0), (8.9, 55.0), (9.0, 57.0), (9.0, 60.2), (4.4, 64.8), (4.4, 67.6), (6.2, 69.4),
             (6.2, 71.0), (2.6, 74.6), (0, 74.6)]
Q_V_TOP, Q_V_BOT = (47.5, 1.6), (8.5, 13.0)            # front opening of the overskirt: (z, half-width)
Q_TIERS = np.linspace(10.6, 45.0, 6)                   # flounce boundaries on the underskirt


def outer_r_q(z):
    pts = Q_SKIRT + Q_BODICE + Q_SHOULDERS[:2]
    return float(np.interp(z, [p[1] for p in pts], [p[0] for p in pts]))


def q_body_profile():
    dome = [(Q_DOME_R * math.cos(t), Q_CROWN_Z1 + Q_DOME_H * math.sin(t)) for t in np.linspace(0, math.pi / 2, 24)]
    crown = [(Q_CROWN_R, Q_CROWN_Z0), (Q_CROWN_R, Q_CROWN_Z1), (Q_DOME_R, Q_CROWN_Z1)] + dome[1:]
    return revolve(PLINTH + Q_SKIRT + Q_BODICE + Q_SHOULDERS + crown + [(0, Q_CROWN_Z1 + Q_DOME_H)])


def q_v_halfwidth(z):
    (z1, w1), (z0, w0) = Q_V_TOP, Q_V_BOT
    return w0 + (w1 - w0) * (z - z0) / (z1 - z0)


def q_v_wedge(grow=0.0):
    (z1, w1), (z0, w0) = Q_V_TOP, Q_V_BOT
    return Manifold.hull_points([[sx * (w + grow), y, z] for z, w in ((z1, w1), (z0, w0))
                                 for sx in (-1, 1) for y in (-40, -2)])


def q_flounce_r(z):
    for zb, zt in zip(Q_TIERS[:-1], Q_TIERS[1:]):
        if zb <= z <= zt:
            if z < zb + 0.9:
                return outer_r_q(z) + (z - zb)
            return outer_r_q(z) + 0.9 - 0.75 * (z - zb - 0.9) / (zt - zb - 0.9)
    return outer_r_q(z)


def q_overskirt():
    """Outer gown layer, open at the front in a V, with pleats and pearl-studded rolled edges."""
    lift = 1.2
    z0, z1 = 9.0, 46.2
    prof = [(0, z0)] + [(outer_r_q(z) + lift, z) for z in np.linspace(z0, z1, 40)] + [(0, z1)]
    parts = [revolve(prof) - q_v_wedge()]
    for d in [s * (48 + 22 * k) for s in (-1, 1) for k in range(6)] + [180]:
        parts.append(capsule(polar(outer_r_q(11.5) + lift - 0.25, FRONT + d, 11.5), 1.3,
                             polar(outer_r_q(44.5) + lift - 0.2, FRONT + d, 44.5), 0.4))
    for sx in (-1, 1):
        pts = []
        for z in np.linspace(9.6, 46.0, 14):
            x = sx * (q_v_halfwidth(z) + 0.35)
            rc = outer_r_q(z) + 0.6
            pts.append([x, -math.sqrt(rc ** 2 - x ** 2), z])
        parts += [capsule(pts[i], 0.8, pts[i + 1], 0.8) for i in range(len(pts) - 1)]
        for z in np.arange(12.0, 45.0, 4.2):
            x = sx * (q_v_halfwidth(z) + 0.35)
            rc = outer_r_q(z) + 1.35
            parts.append(sphere(0.7).translate([x, -math.sqrt(rc ** 2 - x ** 2), z]))
    return union(parts)


def q_flounces():
    """Tiered ruffles on the underskirt, showing through the V opening (one continuous profile)."""
    pts = [(0, Q_TIERS[0]), (outer_r_q(Q_TIERS[0]) - 0.3, Q_TIERS[0])]
    for zb, zt in zip(Q_TIERS[:-1], Q_TIERS[1:]):
        pts += [(q_flounce_r(z), z) for z in np.linspace(zb + 0.9, zt - 0.02, 8)]
        pts.append((outer_r_q(zt) - 0.3, zt))
    pts.append((0, Q_TIERS[-1]))
    return revolve(pts) ^ q_v_wedge(grow=0.6)


def q_waist():
    band = torus(10.9, 1.0).translate([0, 0, 46.6])
    pts = []
    for d in np.linspace(-180, 180, 1200):
        c = max(0.0, math.cos(math.radians(d)))
        z = 45.2 - 1.9 * c * c
        pts.append(polar(outer_r_q(z) + 1.45, FRONT + d, z))
    pts = np.array(pts)
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    beads = []
    for s in np.arange(0, dist[-1] - 0.7, 1.3):
        p = pts[int(np.searchsorted(dist, s))]
        if p[1] < 0 and abs(p[0]) < q_v_halfwidth(p[2]) + 1.3:
            continue                                     # stop at the edges of the opening
        beads.append(sphere(0.6, 16).translate(p.tolist()))
    chain = [sphere(0.6, 16).translate([0, -(q_flounce_r(z) + 0.45), z]) for z in np.arange(44.4, 37.4, -1.25)]
    drop = Manifold.batch_hull([sphere(1.2, 24).translate([0, 0, 35.6]), sphere(0.5, 16).translate([0, 0, 37.2])])
    drop = drop.translate([0, -(q_flounce_r(36.3) + 0.8), 0])
    return union([band, drop] + beads + chain)


def q_stomacher():
    lift, z0, z1 = 0.6, 45.4, 57.6
    prof = [(0, z0)] + [(outer_r_q(z) + lift, z) for z in np.linspace(z0, z1, 20)] + [(0, z1)]
    shape = [(-4.4, 57.6), (4.4, 57.6), (3.0, 52.0), (0.0, 45.4), (-3.0, 52.0)]
    prism = Manifold.extrude(CrossSection([np.array(shape)], FillRule.EvenOdd), 30).rotate([90, 0, 0])
    panel = revolve(prof) ^ prism
    pearls = [sphere(r, 20).translate([0, -(outer_r_q(z) + lift + 0.35), z])
              for z, r in ((53.8, 0.9), (51.9, 0.8), (50.1, 0.75), (48.4, 0.7))]
    return union([panel] + pearls)


def q_necklace():
    parts = []
    for z_of, dmax, br, sp in ((lambda c: 61.3 - 1.0 * c * c, 105, 0.6, 1.25),
                               (lambda c: 60.0 - 2.1 * c * c, 100, 0.65, 1.35)):
        pts = []
        for d in np.linspace(-dmax, dmax, 800):
            z = z_of(max(0.0, math.cos(math.radians(d))))
            pts.append(polar(outer_r_q(z) + 0.3, FRONT + d, z))
        pts = np.array(pts)
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        parts += [sphere(br, 16).translate(pts[int(np.searchsorted(dist, s))].tolist())
                  for s in np.arange(0.3, dist[-1], sp)]
    drop = Manifold.batch_hull([sphere(1.1, 24).translate([0, 0, 55.9]), sphere(0.45, 16).translate([0, 0, 57.4])])
    parts.append(drop.translate([0, -(outer_r_q(56.6) + 0.6 + 0.75), 0]))
    return union(parts)


def q_collar():
    """Standing lace collar behind the neck: scalloped and pearl-tipped, ribbed on both faces, pierced like lace."""
    zb, zt, rin0, rin1, th = 60.5, 73.5, 10.4, 14.0, 1.2
    a0, a1, n = 30.0, 150.0, 9
    shell = revolve([(rin0, zb), (rin0 + th, zb), (rin1 + th, zt), (rin1, zt)])
    fan = [(0, 0)] + [(30 * math.cos(math.radians(a)), 30 * math.sin(math.radians(a))) for a in np.linspace(a0, a1, 40)]
    collar = shell ^ Manifold.extrude(CrossSection([np.array(fan)], FillRule.EvenOdd), 30).translate([0, 0, 55])
    step = (a1 - a0) / n
    rm = rin1 + th / 2
    r_in = lambda z: rin0 + (rin1 - rin0) * (z - zb) / (zt - zb)
    cuts = [sphere(2.45, 32).translate(polar(rm, a0 + step * (i + 0.5), zt + 1.6)) for i in range(n)]
    hole = Manifold.cylinder(6, 0.65, 0.65, 20).translate([0, 0, -3]).rotate([0, 90, 0])
    for i in range(n):
        a = a0 + step * (i + 0.5)
        for z in (65.0, 68.8):
            cuts.append(on_surface(hole, r_in(z) + th / 2, a, z))
    collar = collar - union(cuts)
    parts = [collar]
    for i in range(n + 1):
        a = min(max(a0 + step * i, a0 + 1.2), a1 - 1.2)
        parts.append(sphere(0.65).translate(polar(rm, a, zt - 0.1)))
        parts.append(capsule(polar(r_in(62.0) + 0.05, a, 62.0), 0.4, polar(r_in(72.8) + 0.05, a, 72.8), 0.4))
        parts.append(capsule(polar(r_in(62.0) + th - 0.05, a, 62.0), 0.4, polar(r_in(72.8) + th - 0.05, a, 72.8), 0.4))
    return union(parts)


def q_crown_details():
    parts = [ring(lambda k: sphere(0.6), 43, Q_CROWN_R + 0.05, Q_CROWN_Z0 + 0.6),
             ring(lambda k: sphere(0.6), 43, Q_CROWN_R + 0.05, Q_CROWN_Z1 - 0.55)]
    zc = (Q_CROWN_Z0 + Q_CROWN_Z1) / 2
    for k in range(12):
        ang = FRONT + 30 * k
        tall = k % 2 == 0
        if tall:
            gem = sphere(1.2, 28).scale([0.55, 1, 1])
        else:
            gem = Manifold.batch_hull([sphere(0.95, 20), sphere(0.45, 16).translate([0, 0, 1.3])])
            gem = gem.scale([0.55, 1, 1]).translate([0, 0, -0.4])
        parts.append(on_surface(gem, Q_CROWN_R - 0.1, ang, zc))
        h, w, pr = (5.2, 2.8, 0.8) if tall else (3.4, 2.2, 0.65)
        pt = [(-w / 2, 0), (w / 2, 0), (w / 2 * 0.75, h * 0.6), (0.3, h), (-0.3, h), (-w / 2 * 0.75, h * 0.6)]
        spike = Manifold.extrude(CrossSection([np.array(pt)], FillRule.EvenOdd), 1.4)
        spike = spike.rotate([90, 0, 0]).rotate([0, 0, 90]).translate([-1.4, 0, 0])
        spike = union([spike, sphere(pr).translate([-0.7, 0, h + pr * 0.55])])
        parts.append(on_surface(spike, Q_CROWN_R, ang, Q_CROWN_Z1 - 0.2))
    rr = 2.8
    zz = Q_CROWN_Z1 + Q_DOME_H * math.sqrt(1 - (rr / Q_DOME_R) ** 2)
    parts.append(ring(lambda k: sphere(0.55, 16), 10, rr, zz))
    return union(parts)


def q_finial():
    top = Q_CROWN_Z1 + Q_DOME_H
    stem = Manifold.cylinder(2.4, 2.2, 1.4, 48).translate([0, 0, top - 0.6])
    calyx = ring(lambda k: sphere(0.6, 16), 6, 1.6, top + 1.4)
    pearl = sphere(2.5, 48).translate([0, 0, top + 3.8])
    return union([stem, calyx, pearl])


def q_bow():
    knot = sphere(1.0, 24).scale([0.75, 1.0, 1.0])
    loop = torus(1.6, 0.5, 48, 12).rotate([0, 90, 0]).scale([0.9, 1.0, 0.8])
    tails = [capsule([0, s * 0.6, -0.5], 0.5, [0, s * 2.0, -5.0], 0.42) for s in (-1, 1)]
    bow = union([knot, loop.translate([0, 1.9, 0.2]), loop.translate([0, -1.9, 0.2])] + tails)
    z = 44.6
    return on_surface(bow, outer_r_q(z) + 1.45, FRONT + 180, z)


def q_back_lacing():
    """Corset lacing down the back of the bodice: two rows of eyelets with criss-cross laces."""
    back = FRONT + 180
    zs = np.linspace(48.6, 57.4, 6)
    def pt(z, off, lift):
        r = outer_r_q(z) + lift
        a = back + math.degrees(off / r)
        return polar(r, a, z)
    parts = []
    for z in zs:
        parts += [sphere(0.45, 16).translate(pt(z, s * 1.7, 0.05)) for s in (-1, 1)]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        parts.append(capsule(pt(z0, -1.7, 0.1), 0.32, pt(z1, 1.7, 0.1), 0.32))
        parts.append(capsule(pt(z0, 1.7, 0.1), 0.32, pt(z1, -1.7, 0.1), 0.32))
    return union(parts)


def pepper_holes():
    return union([Manifold.cylinder(10, Q_HOLE_D / 2, Q_HOLE_D / 2, 32).translate(
        polar(Q_HOLE_R_POS, FRONT + 180 + 360.0 * k / Q_HOLE_COUNT, 69.5)) for k in range(Q_HOLE_COUNT)])


def make_queen():
    solid = union([q_body_profile(), plinth_details(), q_overskirt(), q_flounces(), q_waist(), q_stomacher(),
                   q_necklace(), q_collar(), q_crown_details(), q_finial(), q_bow(), q_back_lacing()])
    cuts = union([revolve(Q_CHAMBER), plinth_engraving(), pepper_holes(), fill_opening()])
    return solid - cuts


# ======================= BOARD (stand for the pair) =======================
# Two squares of a chessboard: the queen on d1 (light), the king on e1 (dark), as in a real game.
SQ, FRAME, BOARD_R = 54.0, 10.0, 6.0
BOARD_W, BOARD_D = 2 * SQ + 2 * FRAME, SQ + 2 * FRAME
FRAME_TOP, LIGHT_TOP, DARK_TOP = 6.5, 5.0, 4.4       # everything at or below DARK_TOP shows the first colour
POCKET_D, POCKET_DEPTH = 48.4, 1.8                  # shaker bases are 47.2 mm across
BUMPER_D, BUMPER_DEPTH = 10.0, 1.0                  # recesses underneath for felt or rubber feet


def text_cs(s, x_height, family="DejaVu Serif"):
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    prop = FontProperties(family=family, weight="bold")
    ref = TextPath((0, 0), "x", size=10, prop=prop).get_extents()
    k = x_height / ref.height
    rings = [np.array(p) * k for p in TextPath((0, 0), s, size=10, prop=prop).to_polygons() if len(p) >= 3]
    cs = CrossSection(rings, FillRule.EvenOdd)
    b = cs.bounds()
    return cs.translate([-(b[0] + b[2]) / 2, -(b[1] + b[3]) / 2])


def rounded_rect(w, d, r):
    return CrossSection.square([w - 2 * r, d - 2 * r]).translate([-(w / 2 - r), -(d / 2 - r)]).offset(
        r, JoinType.Round, circular_segments=64)


def board_path(inset, step):
    """Points every `step` mm along the rounded outline, `inset` in from the edge."""
    w, d, r = BOARD_W / 2 - inset, BOARD_D / 2 - inset, BOARD_R - inset
    seg = []
    for cx, cy, a0 in ((w - r, d - r, 0), (-(w - r), d - r, 90), (-(w - r), -(d - r), 180), (w - r, -(d - r), 270)):
        for a in np.linspace(a0, a0 + 90, 30):
            seg.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    seg.append(seg[0])
    seg = np.array(seg)
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    n = int(dist[-1] // step)
    ss = np.linspace(0, dist[-1], n, endpoint=False)
    return np.c_[np.interp(ss, dist, seg[:, 0]), np.interp(ss, dist, seg[:, 1])]


def make_board():
    base = rounded_rect(BOARD_W, BOARD_D, BOARD_R)
    bottom = Manifold.batch_hull([Manifold.extrude(base.offset(-0.4, JoinType.Round), 0.01),
                                  Manifold.extrude(base, 0.01).translate([0, 0, 0.4])])
    mid = Manifold.extrude(base, FRAME_TOP - 1.2).translate([0, 0, 0.4])
    top = Manifold.batch_hull([Manifold.extrude(base, 0.01).translate([0, 0, FRAME_TOP - 0.81]),
                               Manifold.extrude(base.offset(-0.8, JoinType.Round), 0.01).translate([0, 0, FRAME_TOP - 0.01])])
    board = union([bottom, mid, top])

    play = CrossSection.square([2 * SQ, SQ]).translate([-SQ, -SQ / 2])
    lip = Manifold.batch_hull([Manifold.extrude(play, 0.01).translate([0, 0, FRAME_TOP - 0.5]),
                               Manifold.extrude(play.offset(0.55, JoinType.Miter), 0.01).translate([0, 0, FRAME_TOP])])
    cuts = [Manifold.extrude(play, 10).translate([0, 0, LIGHT_TOP]), lip,
            Manifold.cube([SQ, SQ, 10]).translate([0, -SQ / 2, DARK_TOP])]          # e1 is the dark square
    for x, top_z in ((-SQ / 2, LIGHT_TOP), (SQ / 2, DARK_TOP)):
        r = POCKET_D / 2
        cuts.append(Manifold.cylinder(POCKET_DEPTH + 1, r, r, 192).translate([x, 0, top_z - POCKET_DEPTH]))
        cuts.append(Manifold.cylinder(0.51, r, r + 0.5, 192).translate([x, 0, top_z - 0.5]))
    # file letters on the front rail, rank numbers on the side rails; cut down to the first colour
    y_front = -(SQ / 2 + FRAME / 2) + 0.4
    for s_, x, y in (("d", -SQ / 2, y_front), ("e", SQ / 2, y_front),
                     ("1", -(SQ + FRAME / 2), 0), ("1", SQ + FRAME / 2, 0)):
        cuts.append(Manifold.extrude(text_cs(s_, 3.4), 5).translate([x, y, DARK_TOP]))
    # carved diamonds around the outer face, matching the shaker bases
    loz = radial_relief([(0, -1.6), (1.6, 0), (0, 1.6), (-1.6, 0)], 1.4)
    for ang, fixed, span in ((-90, -BOARD_D / 2, BOARD_W / 2 - BOARD_R), (90, BOARD_D / 2, BOARD_W / 2 - BOARD_R),
                             (180, -BOARD_W / 2, BOARD_D / 2 - BOARD_R), (0, BOARD_W / 2, BOARD_D / 2 - BOARD_R)):
        n = int(2 * span // 4.12)
        for t in np.linspace(-span + 2.06, span - 2.06, n):
            if ang in (-90, 90):
                p = [t, fixed - math.copysign(0.6, fixed), 2.5]
            else:
                p = [fixed - math.copysign(0.6, fixed), t, 2.5]
            cuts.append(loz.rotate([0, 0, ang]).translate(p))
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(Manifold.cylinder(BUMPER_DEPTH + 0.01, BUMPER_D / 2, BUMPER_D / 2, 64).translate(
                [sx * (BOARD_W / 2 - 14), sy * (BOARD_D / 2 - 12), -0.01]))
    board = board - union(cuts)
    beads = [sphere(0.75, 16).translate([x, y, FRAME_TOP]) for x, y in board_path(2.2, 1.75)]
    return union([board] + beads)


def export(m, path):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts)
    tm.export(path)
    return tm


if __name__ == "__main__":
    for name, maker, chamber in (("king_salt_shaker", make_king, CHAMBER), ("queen_pepper_shaker", make_queen, Q_CHAMBER)):
        piece = maker()
        t = export(piece, name + ".stl")
        bb = piece.bounding_box()
        print(f"{name}: {bb[3]-bb[0]:.1f} x {bb[4]-bb[1]:.1f} x {bb[5]-bb[2]:.1f} mm, "
              f"watertight={t.is_watertight}, chamber ~{revolve(chamber).volume() / 1000:.0f} mL")
    tb = export(make_board(), "chessboard_stand.stl")
    print(f"board: {BOARD_W:.0f} x {BOARD_D:.0f} x {FRAME_TOP:.1f} mm, watertight={tb.is_watertight}")
    tp = export(make_plug(), "shaker_fill_plug.stl")
    print(f"plug (fits both): watertight={tp.is_watertight}")
