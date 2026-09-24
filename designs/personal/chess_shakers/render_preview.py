import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from PIL import Image


def corner_normals(mesh, angle_deg=35.0, chunk=15000):
    """Per-corner normals, smoothed across edges flatter than angle_deg."""
    fn = mesh.face_normals.astype(np.float32)
    fa = mesh.area_faces.astype(np.float32)
    vf = mesh.vertex_faces
    cos_t = np.cos(np.radians(angle_deg))
    out = np.zeros((len(mesh.faces), 3, 3), np.float32)
    for s in range(0, len(mesh.faces), chunk):
        f = mesh.faces[s:s + chunk]
        nb = vf[f]                                   # (c,3,D)
        valid = nb >= 0
        nbi = np.where(valid, nb, 0)
        nn = fn[nbi]                                 # (c,3,D,3)
        cos = (nn * fn[s:s + chunk, None, None, :]).sum(-1)
        w = np.where(valid & (cos > cos_t), fa[nbi], 0)[..., None]
        cn = (nn * w).sum(2)
        cn /= np.linalg.norm(cn, axis=-1, keepdims=True) + 1e-12
        out[s:s + chunk] = cn
    return out


def render(parts, az=25, el=14, width=760, height=1180, margin=0.06, ao_rays=10, ao_dist=4.0,
           bg=(244, 242, 236), cut_plane=None, cut_color=None):
    """parts: list of (trimesh, rgb). Orthographic camera looking at the parts from (az, el)."""
    meshes = [m for m, _ in parts]
    mesh = trimesh.util.concatenate(meshes)
    cols = np.concatenate([np.tile(np.array(c, np.float32) / 255, (len(m.faces), 1)) for m, c in parts])
    cn = corner_normals(mesh)
    ray = RayMeshIntersector(mesh)

    a, e = np.radians(az), np.radians(el)
    to_cam = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    w = -to_cam
    right = np.cross(w, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, w)

    v = mesh.vertices
    pu, pv = v @ right, v @ up
    cu, cv = (pu.min() + pu.max()) / 2, (pv.min() + pv.max()) / 2
    span = max((pu.max() - pu.min()) / width, (pv.max() - pv.min()) / height) * (1 + 2 * margin)
    center = mesh.bounds.mean(axis=0)
    c0 = center - right * ((center @ right) - cu) - up * ((center @ up) - cv)

    ys, xs = np.mgrid[0:height, 0:width]
    du = (xs - width / 2 + 0.5) * span
    dv = -(ys - height / 2 + 0.5) * span
    origins = c0 + np.outer(du.ravel(), right) + np.outer(dv.ravel(), up) + to_cam * 500
    dirs = np.tile(w, (len(origins), 1))

    tri, ridx, loc = ray.intersects_id(origins, dirs, multiple_hits=False, return_locations=True)
    img = np.tile(np.array(bg, np.float32) / 255, (width * height, 1))
    if len(tri) == 0:
        return Image.fromarray((img.reshape(height, width, 3) * 255).astype(np.uint8))

    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tri], loc)
    n = (cn[tri] * bary[..., None]).sum(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    flip = (n @ to_cam) < 0
    n[flip] *= -1
    base = cols[tri].copy()
    if cut_plane is not None:
        pn, off = np.array(cut_plane[0], float), cut_plane[1]
        on_cut = (np.abs(mesh.face_normals[tri] @ pn) > 0.999) & (np.abs(loc @ pn - off) < 0.05)
        base[on_cut] = np.array(cut_color, np.float32) / 255

    key = np.array([-0.55, -0.75, 0.9]); key /= np.linalg.norm(key)
    fill = np.array([0.8, -0.4, 0.3]); fill /= np.linalg.norm(fill)
    rim = np.array([0.2, 0.9, 0.6]); rim /= np.linalg.norm(rim)

    eps = n * 0.03
    shadow_hit = ray.intersects_any(loc + eps, np.tile(key, (len(loc), 1)))
    lam_k = np.clip(n @ key, 0, 1) * np.where(shadow_hit, 0.25, 1.0)
    lam_f = np.clip(n @ fill, 0, 1)
    lam_r = np.clip(n @ rim, 0, 1)

    # ambient occlusion
    rng = np.random.default_rng(1)
    occl = np.zeros(len(loc), np.float32)
    for _ in range(ao_rays):
        d = rng.normal(size=(len(loc), 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d[(d * n).sum(1) < 0] *= -1
        d = d * 0.7 + n * 0.3
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t2, r2, l2 = ray.intersects_id(loc + eps, d, multiple_hits=False, return_locations=True)
        hit = np.zeros(len(loc), bool)
        dist = np.linalg.norm(l2 - (loc + eps)[r2], axis=1)
        hit[r2[dist < ao_dist]] = True
        occl += hit
    ao = 1 - 0.6 * occl / ao_rays

    h = key + to_cam; h /= np.linalg.norm(h)
    spec = np.clip(n @ h, 0, 1) ** 40 * 0.18 * np.where(shadow_hit, 0.0, 1.0)
    light = 0.30 * ao + 0.62 * lam_k + 0.20 * lam_f * ao + 0.18 * lam_r
    shade = base * light[:, None] + spec[:, None]
    img[ridx] = np.clip(shade, 0, 1)
    return Image.fromarray((img.reshape(height, width, 3) * 255).astype(np.uint8))


def panel(images, gap=24, bg=(244, 242, 236)):
    h = max(i.height for i in images)
    W = sum(i.width for i in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (W, h), bg)
    x = 0
    for i in images:
        out.paste(i, (x, (h - i.height) // 2))
        x += i.width + gap
    return out
