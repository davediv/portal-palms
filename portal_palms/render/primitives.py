"""One vertex buffer of reusable primitives, and the painter that instances them.

Every metaphor scene draws from this same pool. Geometry is built once at init
and never touched again; a scene "changes" by writing different per-instance
transforms, so switching words costs a buffer write and nothing else.

There is one instanced draw call per primitive *kind* that a scene actually
uses, which in practice is three to five for the whole interior.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Final

import moderngl
import numpy as np
from numpy.typing import NDArray

F32 = np.float32
Arr = NDArray[np.float32]

# Per-instance floats: pos(3) scale(3) quat(4) rgba(4) params(4).
INSTANCE_FLOATS: Final = 18
MAX_PER_PRIM: Final = 128

AX_X: Final = 0
AX_Y: Final = 1
AX_Z: Final = 2


class Prim(IntEnum):
    CUBE = 0
    SPHERE = 1
    CYLINDER = 2
    CONE = 3
    PLANE = 4
    TORUS = 5
    TETRA = 6


PRIM_COUNT: Final = len(Prim)


# --------------------------------------------------------------------------- builders


def _cube() -> tuple[Arr, NDArray[np.uint32]]:
    """Unit cube centred on the origin, hard normals so the faces read separately."""
    verts: list[float] = []
    idx: list[int] = []
    faces = (
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    for n, u, v in faces:
        base = len(verts) // 6
        for su, sv in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            verts.extend(0.5 * (n[k] + u[k] * su + v[k] * sv) for k in range(3))
            verts.extend(n)
        idx.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return np.array(verts, dtype=F32).reshape(-1, 6), np.array(idx, dtype=np.uint32)


def _icosphere(subdiv: int = 2) -> tuple[Arr, NDArray[np.uint32]]:
    """Unit-radius sphere. Normals are just the normalised positions."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    tris = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    pts = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in base]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        got = cache.get(key)
        if got is not None:
            return got
        m = pts[a] + pts[b]
        m /= np.linalg.norm(m)
        pts.append(m)
        cache[key] = len(pts) - 1
        return len(pts) - 1

    for _ in range(subdiv):
        nxt: list[tuple[int, int, int]] = []
        for a, b, c in tris:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nxt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        tris = nxt

    pos = np.asarray(pts, dtype=F32)
    verts = np.concatenate((pos, pos), axis=1)
    idx = np.asarray(tris, dtype=np.uint32).reshape(-1)
    return verts, idx


def _cylinder(segments: int = 20) -> tuple[Arr, NDArray[np.uint32]]:
    """Unit-diameter cylinder along Y, spanning y in [-0.5, 0.5], with caps."""
    verts: list[float] = []
    idx: list[int] = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        cx, cz = math.cos(a), math.sin(a)
        for y in (-0.5, 0.5):
            verts.extend((cx * 0.5, y, cz * 0.5, cx, 0.0, cz))
    for i in range(segments):
        a0 = i * 2
        a1 = ((i + 1) % segments) * 2
        idx.extend((a0, a1, a0 + 1, a1, a1 + 1, a0 + 1))
    for y, ny in ((-0.5, -1.0), (0.5, 1.0)):
        centre = len(verts) // 6
        verts.extend((0.0, y, 0.0, 0.0, ny, 0.0))
        ring = len(verts) // 6
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            verts.extend((math.cos(a) * 0.5, y, math.sin(a) * 0.5, 0.0, ny, 0.0))
        for i in range(segments):
            n0 = ring + i
            n1 = ring + (i + 1) % segments
            if ny > 0.0:
                idx.extend((centre, n0, n1))
            else:
                idx.extend((centre, n1, n0))
    return np.array(verts, dtype=F32).reshape(-1, 6), np.array(idx, dtype=np.uint32)


def _cone(segments: int = 20) -> tuple[Arr, NDArray[np.uint32]]:
    """Unit-diameter cone along Y, base at y=-0.5, apex at y=+0.5."""
    verts: list[float] = []
    idx: list[int] = []
    slope = math.atan2(0.5, 1.0)
    ny = math.sin(slope)
    nr = math.cos(slope)
    for i in range(segments):
        a = 2.0 * math.pi * (i + 0.5) / segments
        verts.extend((0.0, 0.5, 0.0, math.cos(a) * nr, ny, math.sin(a) * nr))
        a0 = 2.0 * math.pi * i / segments
        verts.extend(
            (math.cos(a0) * 0.5, -0.5, math.sin(a0) * 0.5, math.cos(a0) * nr, ny, math.sin(a0) * nr)
        )
    for i in range(segments):
        apex = i * 2
        b0 = i * 2 + 1
        b1 = ((i + 1) % segments) * 2 + 1
        idx.extend((apex, b0, b1))
    centre = len(verts) // 6
    verts.extend((0.0, -0.5, 0.0, 0.0, -1.0, 0.0))
    ring = len(verts) // 6
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        verts.extend((math.cos(a) * 0.5, -0.5, math.sin(a) * 0.5, 0.0, -1.0, 0.0))
    for i in range(segments):
        idx.extend((centre, ring + (i + 1) % segments, ring + i))
    return np.array(verts, dtype=F32).reshape(-1, 6), np.array(idx, dtype=np.uint32)


def _plane() -> tuple[Arr, NDArray[np.uint32]]:
    """Unit quad in the XZ plane, facing +Y."""
    verts = np.array(
        [
            [-0.5, 0.0, -0.5, 0.0, 1.0, 0.0],
            [0.5, 0.0, -0.5, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5, 0.0, 1.0, 0.0],
            [-0.5, 0.0, 0.5, 0.0, 1.0, 0.0],
        ],
        dtype=F32,
    )
    return verts, np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


def _torus(
    major_seg: int = 28, minor_seg: int = 10, minor_r: float = 0.12
) -> tuple[Arr, NDArray[np.uint32]]:
    """Ring of outer radius 0.5 lying in XZ."""
    verts: list[float] = []
    idx: list[int] = []
    major_r = 0.5 - minor_r
    for i in range(major_seg):
        u = 2.0 * math.pi * i / major_seg
        cu, su = math.cos(u), math.sin(u)
        for j in range(minor_seg):
            v = 2.0 * math.pi * j / minor_seg
            cv, sv = math.cos(v), math.sin(v)
            nx, ny, nz = cu * cv, sv, su * cv
            verts.extend(
                (
                    cu * major_r + nx * minor_r,
                    ny * minor_r,
                    su * major_r + nz * minor_r,
                    nx,
                    ny,
                    nz,
                )
            )
    for i in range(major_seg):
        for j in range(minor_seg):
            a = i * minor_seg + j
            b = i * minor_seg + (j + 1) % minor_seg
            c = ((i + 1) % major_seg) * minor_seg + j
            d = ((i + 1) % major_seg) * minor_seg + (j + 1) % minor_seg
            idx.extend((a, c, b, b, c, d))
    return np.array(verts, dtype=F32).reshape(-1, 6), np.array(idx, dtype=np.uint32)


def _tetra() -> tuple[Arr, NDArray[np.uint32]]:
    """Shard. Flat-shaded so it catches the key light in hard planes."""
    p = (
        (0.0, 0.5, 0.0),
        (-0.45, -0.3, 0.35),
        (0.45, -0.3, 0.35),
        (0.0, -0.3, -0.5),
    )
    tris = ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2))
    verts: list[float] = []
    idx: list[int] = []
    for a, b, c in tris:
        va, vb, vc = p[a], p[b], p[c]
        ux, uy, uz = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
        wx, wy, wz = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
        nx, ny, nz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        base = len(verts) // 6
        for vtx in (va, vb, vc):
            verts.extend((vtx[0], vtx[1], vtx[2], nx / ln, ny / ln, nz / ln))
        idx.extend((base, base + 1, base + 2))
    return np.array(verts, dtype=F32).reshape(-1, 6), np.array(idx, dtype=np.uint32)


# --------------------------------------------------------------------------- pool


class PrimitivePool:
    """One VBO, one IBO, one instance buffer and one VAO per primitive kind."""

    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program) -> None:
        self.ctx = ctx
        builders = {
            Prim.CUBE: _cube(),
            Prim.SPHERE: _icosphere(2),
            Prim.CYLINDER: _cylinder(),
            Prim.CONE: _cone(),
            Prim.PLANE: _plane(),
            Prim.TORUS: _torus(),
            Prim.TETRA: _tetra(),
        }

        verts: list[Arr] = []
        indices: list[NDArray[np.uint32]] = []
        self.ranges: list[tuple[int, int]] = []
        vbase = 0
        ibase = 0
        for prim in Prim:
            v, i = builders[prim]
            verts.append(v)
            indices.append(i + vbase)
            self.ranges.append((ibase, int(i.size)))
            vbase += int(v.shape[0])
            ibase += int(i.size)

        vbo_data = np.concatenate(verts, axis=0)
        ibo_data = np.concatenate(indices)
        self.vbo = ctx.buffer(vbo_data.tobytes())
        self.ibo = ctx.buffer(ibo_data.tobytes())
        self.vertex_count = vbase
        self.index_count = ibase

        # One instance buffer per kind. OpenGL 4.1 has no baseInstance, so a
        # single shared buffer could not be drawn in per-kind ranges.
        self.instances: list[Arr] = []
        self.buffers: list[moderngl.Buffer] = []
        self.vaos: list[moderngl.VertexArray] = []
        for prim in Prim:
            data = np.zeros((MAX_PER_PRIM, INSTANCE_FLOATS), dtype=F32)
            buf = ctx.buffer(reserve=data.nbytes, dynamic=True)
            count = self.ranges[int(prim)][1]
            vao = ctx.vertex_array(
                prog,
                [
                    (self.vbo, "3f 3f", "in_vert", "in_norm"),
                    (buf, "3f 3f 4f 4f 4f/i", "i_pos", "i_scale", "i_quat", "i_color", "i_params"),
                ],
                index_buffer=self.ibo,
                index_element_size=4,
            )
            vao.vertices = count
            self.instances.append(data)
            self.buffers.append(buf)
            self.vaos.append(vao)

    def draw(self, counts: NDArray[np.int32]) -> int:
        """Upload and draw every kind with at least one instance. Returns draw calls."""
        calls = 0
        ranges = self.ranges
        bufs = self.buffers
        vaos = self.vaos
        data = self.instances
        for k in range(PRIM_COUNT):
            n = int(counts[k])
            if n <= 0:
                continue
            # The whole fixed-size array goes up every time. It is 9KB and a
            # slice to the used prefix would allocate a view object per draw.
            bufs[k].write(data[k])
            first, count = ranges[k]
            vaos[k].render(moderngl.TRIANGLES, vertices=count, first=first, instances=n)
            calls += 1
        return calls

    def release(self) -> None:
        for v in self.vaos:
            v.release()
        for b in self.buffers:
            b.release()
        self.vbo.release()
        self.ibo.release()


# --------------------------------------------------------------------------- painter


class Painter:
    """The authoring surface for metaphor scenes.

    Every method writes one instance into a preallocated array. Nothing here
    allocates: colours arrive as plain float tuples (whose members are cached
    Python floats), quaternions are built from two trig calls, and the arrays
    were sized at init.
    """

    __slots__ = ("counts", "inst", "overflow")

    def __init__(self, pool: PrimitivePool) -> None:
        self.inst = pool.instances
        self.counts = np.zeros(PRIM_COUNT, dtype=np.int32)
        self.overflow = 0

    def reset(self) -> None:
        self.counts[:] = 0
        self.overflow = 0

    def _slot(self, kind: int) -> int:
        n = int(self.counts[kind])
        if n >= MAX_PER_PRIM:
            self.overflow += 1
            return -1
        self.counts[kind] = n + 1
        return n

    # ------------------------------------------------------------------ core

    def put(
        self,
        kind: int,
        x: float,
        y: float,
        z: float,
        sx: float,
        sy: float,
        sz: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        r: float,
        g: float,
        b: float,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
        mat: float = 0.0,
    ) -> None:
        i = self._slot(kind)
        if i < 0:
            return
        row = self.inst[kind][i]
        row[0] = x
        row[1] = y
        row[2] = z
        row[3] = sx
        row[4] = sy
        row[5] = sz
        row[6] = qx
        row[7] = qy
        row[8] = qz
        row[9] = qw
        row[10] = r
        row[11] = g
        row[12] = b
        row[13] = a
        row[14] = emis
        row[15] = 1.0
        row[16] = rim
        row[17] = mat

    def axis(
        self,
        kind: int,
        x: float,
        y: float,
        z: float,
        sx: float,
        sy: float,
        sz: float,
        angle: float,
        ax: int,
        col: tuple[float, float, float],
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        """Place a primitive rotated about one cardinal axis -- the common case."""
        h = angle * 0.5
        s = math.sin(h)
        self.put(
            kind,
            x,
            y,
            z,
            sx,
            sy,
            sz,
            s if ax == AX_X else 0.0,
            s if ax == AX_Y else 0.0,
            s if ax == AX_Z else 0.0,
            math.cos(h),
            col[0],
            col[1],
            col[2],
            a,
            emis,
            rim,
        )

    def box(
        self,
        x: float,
        y: float,
        z: float,
        sx: float,
        sy: float,
        sz: float,
        col: tuple[float, float, float],
        angle: float = 0.0,
        ax: int = AX_Z,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        self.axis(Prim.CUBE, x, y, z, sx, sy, sz, angle, ax, col, a, emis, rim)

    def ball(
        self,
        x: float,
        y: float,
        z: float,
        r: float,
        col: tuple[float, float, float],
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        self.put(
            Prim.SPHERE, x, y, z, r, r, r, 0.0, 0.0, 0.0, 1.0, col[0], col[1], col[2], a, emis, rim
        )

    def column(
        self,
        x: float,
        y: float,
        z: float,
        d: float,
        h: float,
        col: tuple[float, float, float],
        angle: float = 0.0,
        ax: int = AX_Z,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        self.axis(Prim.CYLINDER, x, y, z, d, h, d, angle, ax, col, a, emis, rim)

    def cone(
        self,
        x: float,
        y: float,
        z: float,
        d: float,
        h: float,
        col: tuple[float, float, float],
        angle: float = 0.0,
        ax: int = AX_Z,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        self.axis(Prim.CONE, x, y, z, d, h, d, angle, ax, col, a, emis, rim)

    def slab(
        self,
        x: float,
        y: float,
        z: float,
        sx: float,
        sz: float,
        col: tuple[float, float, float],
        angle: float = 0.0,
        ax: int = AX_Y,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        self.axis(Prim.PLANE, x, y, z, sx, 1.0, sz, angle, ax, col, a, emis, rim)

    def ring(
        self,
        x: float,
        y: float,
        z: float,
        d: float,
        col: tuple[float, float, float],
        angle: float = 0.0,
        ax: int = AX_X,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        self.axis(Prim.TORUS, x, y, z, d, d, d, angle, ax, col, a, emis, rim)

    def floor(
        self,
        y: float = 0.0,
        size: float = 46.0,
        col: tuple[float, float, float] = (0.247, 0.839, 0.549),
        bright: float = 0.0,
    ) -> None:
        """The shared grid ground plane. Drawn additively, so it never occludes."""
        self.put(
            Prim.PLANE, 0.0, y, 0.0, size, 1.0, size,
            0.0, 0.0, 0.0, 1.0, col[0], col[1], col[2], 1.0, bright, 0.0, 1.0,
        )  # fmt: skip

    def shard(
        self,
        x: float,
        y: float,
        z: float,
        s: float,
        col: tuple[float, float, float],
        angle: float = 0.0,
        ax: int = AX_Y,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        self.axis(Prim.TETRA, x, y, z, s, s, s, angle, ax, col, a, emis, rim)

    # ------------------------------------------------------------------ helpers

    def rod(
        self,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        thick: float,
        col: tuple[float, float, float],
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        """A cylinder spanning two points. Limbs, tethers and connectors."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-5:
            return
        ux, uy, uz = dx / length, dy / length, dz / length
        # Shortest-arc quaternion taking +Y onto the direction.
        w = 1.0 + uy
        if w < 1e-5:
            qx, qy, qz, qw = 0.0, 0.0, 1.0, 0.0  # antiparallel: half turn about Z
        else:
            qx, qy, qz, qw = uz, 0.0, -ux, w
            inv = 1.0 / math.sqrt(qx * qx + qz * qz + qw * qw)
            qx *= inv
            qz *= inv
            qw *= inv
        self.put(
            Prim.CYLINDER,
            (x0 + x1) * 0.5,
            (y0 + y1) * 0.5,
            (z0 + z1) * 0.5,
            thick,
            length,
            thick,
            qx,
            qy,
            qz,
            qw,
            col[0],
            col[1],
            col[2],
            a,
            emis,
            rim,
        )

    def figure(
        self,
        x: float,
        y: float,
        z: float,
        scale: float,
        col: tuple[float, float, float],
        *,
        lean: float = 0.0,
        stride: float = 0.0,
        arm: float = 0.0,
        crouch: float = 0.0,
        facing: float = 1.0,
        a: float = 1.0,
        emis: float = 0.0,
        rim: float = 0.0,
    ) -> None:
        """A small articulated person. Eight primitives, no skeleton, no rig.

        `lean` tilts the torso, `stride` swings the legs (a walk cycle is just a
        sine into this), `arm` swings the arms, `crouch` drops the hips. Every
        metaphor that needs a human uses this, so a person always looks the same
        across the whole set.
        """
        s = scale
        hip_y = y + s * (0.95 - crouch * 0.35)
        head_y = hip_y + s * 1.05
        lx = math.sin(lean) * s * 0.5
        # torso
        self.rod(x, hip_y, z, x + lx, head_y - s * 0.18, z, s * 0.20, col, a, emis, rim)
        # head
        self.ball(x + lx * 1.25, head_y, z, s * 0.20, col, a, emis, rim)
        # legs
        sw = math.sin(stride) * s * 0.42
        for side in (-1.0, 1.0):
            foot_x = x + sw * side * facing
            self.rod(
                x + side * s * 0.10,
                hip_y,
                z + side * s * 0.10,
                foot_x,
                y,
                z + side * s * 0.10,
                s * 0.11,
                col,
                a,
                emis,
                rim,
            )
        # arms
        aw = math.sin(arm) * s * 0.40
        sh_y = head_y - s * 0.30
        for side in (-1.0, 1.0):
            self.rod(
                x + lx + side * s * 0.16,
                sh_y,
                z + side * s * 0.12,
                x + lx + aw * side * facing + side * s * 0.10,
                sh_y - s * 0.62,
                z + side * s * 0.12,
                s * 0.09,
                col,
                a,
                emis,
                rim,
            )
