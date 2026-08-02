"""All GLSL for the app, as source strings composed at init.

Everything targets `#version 330 core`: macOS caps OpenGL at 4.1, so there are
no compute shaders and no SSBOs anywhere in this file. Particles move by
transform feedback, which is core in 3.3 and works everywhere.

Every fragment shader that draws into the main target writes **premultiplied**
colour, and the main pass runs with blend_func = (ONE, ONE_MINUS_SRC_ALPHA).
That is what lets the portal shader emit an opaque interior and an additive
outer glow from the same draw call, instead of needing a second composite pass.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- shared

VERSION: Final = "#version 330 core\n"

# Ashima Arts / Stefan Gustavson simplex noise, MIT licensed. Chosen over a
# sampled flipbook because the rim must cost bandwidth, not memory, and because
# a looping texture reads as a loop the moment anyone films it.
NOISE: Final = """
vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v){
  const vec2 Cc = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, Cc.yyy));
  vec3 x0 = v - i + dot(i, Cc.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + Cc.xxx;
  vec3 x2 = x0 - i2 + Cc.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

float hash11(float p){ return fract(sin(p * 127.1) * 43758.5453123); }
float hash12(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
"""

PALETTE: Final = """
const vec3 BASE     = vec3(0.043, 0.059, 0.102);
const vec3 GREEN    = vec3(0.247, 0.839, 0.549);
const vec3 YELLOW   = vec3(1.000, 0.773, 0.192);
const vec3 MAGENTA  = vec3(1.000, 0.239, 0.498);
const vec3 OFFWHITE = vec3(0.949, 0.957, 0.941);
"""

# --------------------------------------------------------------------------- fullscreen

# No vertex buffer: gl_VertexID drives a 3-vertex oversized triangle, which
# beats a quad because it avoids the diagonal seam's duplicated fragment work.
FULLSCREEN_VS: Final = (
    VERSION
    + """
out vec2 v_uv;
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
)

# --------------------------------------------------------------------------- camera

CAMERA_FS: Final = (
    VERSION
    + PALETTE
    + """
in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_cam;
uniform vec2  u_fit;        // camera plate half-extent in uv, per axis
uniform float u_mirror;     // 1 to flip horizontally
uniform float u_dim;        // 1 normal, lower to sink the plate toward base

void main(){
  // The plate is letterboxed on whichever axis is constrained and faded into
  // the base colour along that axis. Never cover-cropped: a 4:3 sensor
  // cover-cropped into 9:16 throws away the horizontal room the two-hand
  // gesture needs, and into 16:9 it throws away the vertical room raised hands
  // need. Bars are the honest answer.
  vec2 uv = v_uv;
  vec2 plate = (uv - 0.5) / (u_fit * 2.0) + 0.5;
  // The camera buffer is a top-down numpy array uploaded as-is, so row 0 lands
  // at v=0, which GL puts at the bottom. Flip v here rather than paying for a
  // cv2.flip on the CPU every frame.
  vec2 cam_uv = vec2(mix(plate.x, 1.0 - plate.x, u_mirror), 1.0 - plate.y);

  vec3 col = BASE;
  float inside = step(0.0, plate.x) * step(plate.x, 1.0)
               * step(0.0, plate.y) * step(plate.y, 1.0);
  if (inside > 0.5) {
    col = texture(u_cam, cam_uv).rgb;
    // Cool the plate down and drop its contrast so the portal reads as the
    // only light source in the room.
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, 0.85);
    // u_dim sinks the plate further toward base while the word card is up, so
    // the example sentence stays readable over a moving room. Held fairly high
    // the rest of the time: the person has to stay clearly visible, that is the
    // whole point of a mirror.
    col = mix(BASE, col, 0.86 * u_dim);

    // Only fade the axis that actually has a bar against it.
    float fade = 1.0;
    if (u_fit.y < 0.499) fade *= smoothstep(0.0, 0.13, plate.y) * smoothstep(1.0, 0.87, plate.y);
    if (u_fit.x < 0.499) fade *= smoothstep(0.0, 0.10, plate.x) * smoothstep(1.0, 0.90, plate.x);
    col = mix(BASE, col, fade);
  }

  // Scrims. Text is anchored to the top and bottom edges, and behind it is
  // whatever the room happens to be -- a white wall and a bright shirt will
  // swallow off-white type. Darkening those bands is what guarantees the copy
  // is legible no matter what the camera sees.
  col *= 1.0 - 0.62 * smoothstep(0.40, 0.0, v_uv.y);
  col *= 1.0 - 0.40 * smoothstep(0.90, 1.0, v_uv.y);

  // Vignette, and a faint paddy-green cast in the corners so the dark areas are
  // not dead flat on a phone camera filming the screen.
  vec2 d = v_uv - 0.5;
  float vig = 1.0 - dot(d, d) * 0.70;
  col *= vig;
  col += GREEN * 0.012 * (1.0 - vig);

  f_color = vec4(col, 1.0);
}
"""
)

# --------------------------------------------------------------------------- portal

# One draw call produces the whole portal: the masked interior, the torn rim,
# the outward sparks and the glow. Premultiplied output means the interior can
# be opaque (alpha 1) while the glow is purely additive (alpha 0).
PORTAL_VS: Final = (
    VERSION
    + """
out vec2 v_uv;
uniform vec2 u_center;      // portal centre in render-frame uv
uniform vec2 u_extent;      // half-size of the quad we draw, in uv
void main(){
  // Exactly four vertices for a triangle strip. An oversized triangle would be
  // cheaper to set up but would cover nine times the area, and this quad is
  // sized to the rim's real footprint precisely to keep fragment cost down.
  vec2 p = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  vec2 local = p * 2.0 - 1.0;
  vec2 uv = u_center + local * u_extent;
  v_uv = uv;
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
"""
)

PORTAL_FS: Final = (
    VERSION
    + NOISE
    + PALETTE
    + """
in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_scene;

uniform vec2  u_center;
uniform vec2  u_radius;     // (major, minor) in uv-x units
uniform float u_rot;        // ring roll, radians
uniform float u_aspect;     // render height / width, to keep the ellipse round
uniform float u_time;
uniform float u_open;       // 0 seam .. 1 fully open
uniform float u_intensity;  // user effect-intensity setting
uniform float u_charge;     // 0..1 hold-to-lock progress
uniform float u_flash;      // 0..1 lock impact flash
uniform vec2  u_parallax;
uniform float u_pxscale;    // render width in pixels
uniform float u_scale01;    // portal size relative to a nominal open portal
uniform float u_scene_aspect;
uniform float u_zoom;

float fbm_ring(vec2 dir, float t, float base_freq, int octaves){
  float sum = 0.0;
  float amp = 1.0;
  float freq = base_freq;
  float norm = 0.0;
  for (int i = 0; i < 4; ++i){
    if (i >= octaves) break;
    sum  += snoise(vec3(dir * freq, t)) * amp;
    norm += amp;
    amp  *= 0.5;
    freq *= 2.31;
    t    *= 1.9;
  }
  return sum / max(norm, 1e-4);
}

void main(){
  // ---- portal space: rotate, then divide by the two radii so r == 1 is the rim
  vec2 d = v_uv - u_center;
  d.y *= u_aspect;
  float cs = cos(-u_rot), sn = sin(-u_rot);
  vec2 p = vec2(d.x * cs - d.y * sn, d.x * sn + d.y * cs);
  vec2 q = p / max(u_radius, vec2(1e-4));

  float r = length(q);
  if (r > 1.85) discard;                // past where the sparks and halo die

  vec2 dir = q / max(r, 1e-5);          // wraps seamlessly, unlike raw atan2

  // ---- the tear: displace the boundary with layered animated noise
  float t = u_time;
  // A domain warp before the main octaves is what turns a wobbling circle into
  // something that looks ripped rather than drawn.
  vec2 warp = vec2(snoise(vec3(dir * 1.7, t * 0.31)), snoise(vec3(dir * 1.7 + 5.2, t * 0.31)));
  vec2 wdir = dir + warp * 0.35;

  float big   = fbm_ring(wdir, t * 0.33, 1.9, 3);
  float mid   = snoise(vec3(wdir * 6.1, t * 0.95));
  float fine  = snoise(vec3(dir  * 15.0, t * 2.4));
  float crack = snoise(vec3(dir  * 34.0, t * 5.1));

  // Weighted toward the low octaves: big irregular bites out of the silhouette
  // read as torn, while piling on high frequency only reads as electrical buzz.
  float tear = big * 0.205 + mid * 0.070 + fine * 0.024 + crack * 0.008;
  // A closed seam is a tight bright line; the tear only ragged-ises as it opens.
  tear *= mix(0.30, 1.0, u_open) * u_intensity;
  // Charging pulls the edge taut and makes it breathe faster the closer it gets.
  tear *= 1.0 - 0.35 * u_charge;

  float edge = 1.0 + tear;
  float sd = r - edge;                  // < 0 inside the tear

  // ---- everything below is measured in PIXELS, not in r.
  // r is normalised by both radii, so one r-unit is a very different distance
  // along the major and minor axes. Without this conversion the filament would
  // be a fat band across a wide portal and sub-pixel thin along a closed seam.
  float gr = length(vec2(dir.x / u_radius.x, dir.y / u_radius.y));
  float sd_px = sd * u_pxscale / max(gr, 1e-4);
  float sizef = clamp(u_scale01, 0.45, 1.5);

  // ---- the filament: bright, thin, and unevenly lit along its length
  float amp = 0.22 + 1.05 * (0.5 + 0.5 * snoise(vec3(dir * 3.7, t * 1.15)));
  // Occasional burn-through gaps, so the edge is broken rather than continuous.
  float gap = smoothstep(0.62, 0.30, snoise(vec3(dir * 7.3 + 31.0, t * 0.85)));
  amp *= mix(0.10, 1.0, gap);

  float w = (2.6 + 2.4 * u_charge) * sizef;
  float k = sd_px / w;
  float core = exp(-k * k) * amp;

  // A second filament at a slightly different radius reads as another layer of
  // ripped material behind the first, which is most of what sells "torn".
  float tear2 = snoise(vec3(wdir * 3.3 + 11.0, t * 0.51)) * 0.11
              + snoise(vec3(dir * 9.7 + 4.0,  t * 1.7))  * 0.035;
  float sd2_px = (r - 1.0 - tear2 * mix(0.30, 1.0, u_open)) * u_pxscale / max(gr, 1e-4);
  float k2 = sd2_px / (w * 1.7);
  float core2 = exp(-k2 * k2) * amp * 0.45
              * smoothstep(0.15, 0.75, snoise(vec3(dir * 5.1 + 60.0, t * 0.7)) * 0.5 + 0.5);

  float halo = exp(-abs(sd_px) / (22.0 * sizef)) * (0.35 + 0.65 * amp);

  // ---- outward sparks: thin licks of flame that only exist beyond the edge
  float slick = smoothstep(0.10, 0.95, snoise(vec3(dir * 8.0, t * 1.5)));
  float flick = 0.45 + 0.55 * (0.5 + 0.5 * snoise(vec3(dir * 24.0, t * 6.5)));
  float spark_px = (8.0 + 46.0 * slick) * sizef * u_intensity;
  float outside = smoothstep(-1.0, 3.0, sd_px);
  float sparks = exp(-max(sd_px, 0.0) / max(spark_px, 1e-3)) * slick * flick * outside;
  // Break the streaks up along their length so they read as licks of flame
  // rather than as clean rays from a light source.
  sparks *= 0.35 + 0.85 * (0.5 + 0.5 * snoise(vec3(dir * 5.0, r * 7.0 - t * 3.2)));
  // Guarantee the glow has died before the quad ends, whatever the settings do.
  sparks *= smoothstep(1.82, 1.25, r);

  // ---- inner bleed: a narrow lip of light just inside the tear. Kept tight on
  // purpose -- spread over the whole interior it becomes a green wash that
  // buries the metaphor, which is the thing the user is actually here to read.
  float inner_glow = exp(min(sd_px, 0.0) / (26.0 * sizef)) * smoothstep(2.0, -2.0, sd_px);

  // ---- interior. Sampled from the *unrotated* offset, so rolling the ring
  // spins the frame while the world behind it stays upright. Combined with the
  // camera parallax in scene.py this reads as a hole with depth behind it
  // rather than a texture stuck to the hands.
  vec2 suv = 0.5 + vec2(d.x, d.y / u_scene_aspect) / (u_radius.x * u_zoom) + u_parallax;
  vec3 interior = texture(u_scene, suv).rgb;
  float inside = smoothstep(1.0, -1.0, sd_px);

  // Interior darkens toward the rim and picks up its colour, so the edge is not
  // a hard cut-out and the eye reads depth rather than a pasted-on oval.
  interior *= 0.34 + 0.66 * smoothstep(0.0, -70.0 * sizef, sd_px);
  interior += GREEN * inner_glow * 0.16 * u_open;

  // ---- rim colour: white-hot centre, green bleeding in, yellow to magenta out
  vec3 rim = vec3(0.0);
  rim += OFFWHITE * (core * 3.0 + core2 * 1.6);
  rim += mix(GREEN, YELLOW, smoothstep(0.0, 0.6, u_charge)) * halo * 0.55;
  rim += mix(YELLOW, MAGENTA, smoothstep(0.0, 0.5, sd_px / max(spark_px, 1e-3)))
         * sparks * 1.35;
  rim += GREEN * inner_glow * 0.10;

  // Lock impact: one bright ring that runs outward and dies.
  float ring_px = u_flash * 260.0 * sizef;
  float ring = exp(-abs(sd_px - ring_px) / 14.0) * u_flash * (1.0 - u_flash * 0.5);
  rim += mix(OFFWHITE, YELLOW, 0.4) * ring * 2.2;

  rim *= u_intensity;

  vec3 col = interior * inside + rim;
  float alpha = inside;                 // premultiplied: glow adds, interior replaces
  f_color = vec4(col, alpha);
}
"""
)

# --------------------------------------------------------------------------- particles

# Transform feedback: one VBO of state in, one out, swapped every frame. No
# readback, no CPU touch, and no SSBO — which we could not use at 4.1 anyway.
PARTICLE_UPDATE_VS: Final = (
    VERSION
    + NOISE
    + """
in vec2  in_pos;
in vec2  in_vel;
in float in_life;
in float in_seed;

out vec2  out_pos;
out vec2  out_vel;
out float out_life;
out float out_seed;

uniform float u_dt;
uniform float u_time;
uniform vec2  u_center;
uniform vec2  u_radius;
uniform float u_rot;
uniform float u_aspect;
uniform float u_emit;       // 0..1 how hard the rim is shedding right now
uniform float u_implode;    // 1 while the portal is snapping shut

void main(){
  vec2  pos  = in_pos;
  vec2  vel  = in_vel;
  float life = in_life - u_dt * mix(0.55, 1.5, in_seed);
  float seed = in_seed;

  if (life <= 0.0) {
    // Respawn somewhere on the current rim. Staggering by seed keeps the
    // emission continuous instead of pulsing once per particle lifetime.
    float a = hash11(seed * 91.7 + floor(u_time * 3.1) + gl_VertexID * 0.017) * 6.2831853;
    vec2 adir = vec2(cos(a), sin(a));
    // Emit from the same hot spots the rim shader lights up, so the sparks look
    // thrown off the bright parts of the tear instead of raining evenly off a
    // circle. Below-threshold angles simply respawn dead and cost nothing.
    float slick = smoothstep(0.10, 0.95, snoise(vec3(adir * 8.0, u_time * 1.5)));
    if (hash11(seed * 57.3 + u_time * 2.0) > slick * 0.9 + 0.08) {
      out_pos = pos; out_vel = vec2(0.0); out_life = 0.0; out_seed = seed;
      return;
    }
    float wob = 1.0 + snoise(vec3(adir * 2.0, u_time * 0.4)) * 0.16;
    vec2 e = adir * u_radius * wob;
    float cs = cos(u_rot), sn = sin(u_rot);
    vec2 world = vec2(e.x * cs - e.y * sn, e.x * sn + e.y * cs);
    world.y /= u_aspect;
    pos = u_center + world;

    vec2 n = normalize(world + vec2(1e-5));
    float speed = (0.06 + 0.34 * hash11(seed * 13.3 + u_time)) * mix(0.35, 1.0, u_emit);
    vel = n * speed;
    vel += vec2(-n.y, n.x) * (hash11(seed * 7.7 + u_time * 1.3) - 0.5) * 0.22;
    life = mix(0.25, 1.0, hash11(seed * 3.1 + u_time * 0.7));
    // Off-duty particles park at zero life and zero size instead of being
    // culled, which keeps the draw a single fixed-count instanced call.
    if (u_emit <= 0.001) { life = 0.0; }
  } else {
    // Curl-ish drift so the sparks swirl instead of flying straight out.
    float n1 = snoise(vec3(pos * 9.0, u_time * 0.8));
    float n2 = snoise(vec3(pos * 9.0 + 17.0, u_time * 0.8));
    vel += vec2(n2, -n1) * u_dt * 0.55;
    vel *= (1.0 - u_dt * 1.6);
    vel.y -= u_dt * 0.05;              // a touch of rise, embers not rain
    if (u_implode > 0.0) {
      vec2 toward = u_center - pos;
      vel += toward * u_dt * 14.0 * u_implode;
      life -= u_dt * 1.8 * u_implode;
    }
    pos += vel * u_dt;
  }

  out_pos = pos;
  out_vel = vel;
  out_life = max(life, 0.0);
  out_seed = seed;
}
"""
)

PARTICLE_DRAW_VS: Final = (
    VERSION
    + """
in vec2  quad;
in vec2  in_pos;
in vec2  in_vel;
in float in_life;
in float in_seed;

out vec2  v_quad;
out float v_life;
out float v_seed;

uniform float u_aspect;
uniform float u_size;

void main(){
  float life = in_life;
  // Stretch each spark along its own velocity: a round dot reads as dust, a
  // streak reads as something thrown off an energy edge.
  vec2 dirv = in_vel;
  float sp = length(dirv);
  vec2 axis = sp > 1e-4 ? dirv / sp : vec2(1.0, 0.0);
  float stretch = 1.0 + min(sp * 9.0, 5.5);

  float s = u_size * (0.30 + 0.70 * in_seed) * smoothstep(0.0, 0.25, life);
  vec2 local = vec2(quad.x * stretch, quad.y) * s;
  vec2 world = vec2(local.x * axis.x - local.y * axis.y,
                    local.x * axis.y + local.y * axis.x);
  world.y /= u_aspect;

  vec2 uv = in_pos + world;
  v_quad = quad;
  v_life = life;
  v_seed = in_seed;
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
"""
)

PARTICLE_DRAW_FS: Final = (
    VERSION
    + PALETTE
    + """
in vec2  v_quad;
in float v_life;
in float v_seed;
out vec4 f_color;
uniform float u_intensity;
void main(){
  float d = dot(v_quad, v_quad);
  if (d > 1.0 || v_life <= 0.0) discard;
  float a = (1.0 - d);
  a *= a;
  a *= smoothstep(0.0, 0.35, v_life);
  // Young sparks are white-hot, then cool through yellow into magenta as they die.
  vec3 col = mix(MAGENTA, YELLOW, smoothstep(0.0, 0.55, v_life));
  col = mix(col, OFFWHITE, smoothstep(0.6, 1.0, v_life) * 0.85);
  f_color = vec4(col * a * 1.7 * u_intensity, 0.0);   // additive: alpha stays 0
}
"""
)

# --------------------------------------------------------------------------- scene

# The 25 metaphors share one vertex buffer of primitives and one instance
# buffer. Per-instance transform is position + scale + quaternion, composed on
# the GPU, so the CPU never builds a matrix.
SCENE_VS: Final = (
    VERSION
    + """
in vec3 in_vert;
in vec3 in_norm;

in vec3 i_pos;
in vec3 i_scale;
in vec4 i_quat;
in vec4 i_color;
in vec4 i_params;     // x emissive, y alpha, z rim boost, w material id

out vec3 v_norm;
out vec3 v_world;
out vec4 v_color;
out vec4 v_params;

uniform mat4 u_viewproj;

vec3 qrot(vec4 q, vec3 v){
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main(){
  vec3 world = qrot(i_quat, in_vert * i_scale) + i_pos;
  v_world = world;
  v_norm  = normalize(qrot(i_quat, in_norm / max(i_scale, vec3(1e-4))));
  v_color = i_color;
  v_params = i_params;
  gl_Position = u_viewproj * vec4(world, 1.0);
}
"""
)

SCENE_FS: Final = (
    VERSION
    + PALETTE
    + """
in vec3 v_norm;
in vec3 v_world;
in vec4 v_color;
in vec4 v_params;
out vec4 f_color;

uniform vec3 u_eye;
uniform vec3 u_key;      // key light direction
uniform vec3 u_keycol;
uniform vec3 u_fillcol;

void main(){
  vec3 n = normalize(v_norm);
  vec3 v = normalize(u_eye - v_world);

  if (v_params.w > 0.5) {
    // Grid floor. A ground plane for almost nothing: it gives every metaphor
    // depth cues and a horizon, and fades out before the far plane so there is
    // no hard edge to give the illusion away.
    vec2 dw = fwidth(v_world.xz * 0.5);
    vec2 g = abs(fract(v_world.xz * 0.5) - 0.5) / max(dw, 1e-5);
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    // Seen edge-on a cell shrinks below a pixel and the grid aliases to a solid
    // fill. Fade it out before that happens or the floor becomes a flat wash.
    line *= smoothstep(0.50, 0.16, max(dw.x, dw.y));
    float fade = exp(-length(v_world - u_eye) * 0.16);
    vec3 gcol = v_color.rgb * line * fade * (0.55 + v_params.x);
    f_color = vec4(gcol, 0.0);            // additive: the floor never occludes
    return;
  }

  float ndl = max(dot(n, normalize(u_key)), 0.0);
  float wrap = ndl * 0.72 + 0.28;                 // soft wrap, nothing goes black
  vec3 lit = v_color.rgb * (u_keycol * wrap + u_fillcol * (0.35 + 0.35 * n.y));

  // Rim light along the silhouette keeps small primitives readable against the
  // dark interior without needing a shadow pass.
  float fres = pow(1.0 - max(dot(n, v), 0.0), 2.6);
  lit += mix(GREEN, YELLOW, v_params.z) * fres * (0.30 + 0.9 * v_params.z);

  lit += v_color.rgb * v_params.x * 2.4;          // emissive term feeds the bloom

  float a = v_color.a * v_params.y;
  f_color = vec4(lit * a, a);
}
"""
)

SCENE_BG_FS: Final = (
    VERSION
    + PALETTE
    + """
in vec2 v_uv;
out vec4 f_color;
uniform float u_time;
uniform vec3  u_tint;
void main(){
  // A shallow gradient plus a slow drifting glow, so the interior never reads
  // as a flat fill behind the primitives.
  vec2 d = v_uv - vec2(0.5, 0.45);
  float g = exp(-dot(d, d) * 4.2);
  // Deliberately dark: the metaphors are lit primitives read against negative
  // space, and a bright backdrop flattens them into silhouettes.
  vec3 col = BASE * 0.85 + u_tint * g * 0.045;
  col += u_tint * 0.012 * (0.5 + 0.5 * sin(u_time * 0.6 + v_uv.y * 3.0));
  f_color = vec4(col, 1.0);
}
"""
)

# --------------------------------------------------------------------------- bloom

BLOOM_BRIGHT_FS: Final = (
    VERSION
    + """
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform float u_threshold;
void main(){
  vec3 c = texture(u_src, v_uv).rgb;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float k = max(l - u_threshold, 0.0) / max(l, 1e-4);
  f_color = vec4(c * k, 1.0);
}
"""
)

BLOOM_BLUR_FS: Final = (
    VERSION
    + """
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_dir;        // texel step, one axis at a time
void main(){
  // 9-tap separable gaussian folded into 5 bilinear fetches.
  vec3 c = texture(u_src, v_uv).rgb * 0.2270270270;
  vec2 o1 = u_dir * 1.3846153846;
  vec2 o2 = u_dir * 3.2307692308;
  c += texture(u_src, v_uv + o1).rgb * 0.3162162162;
  c += texture(u_src, v_uv - o1).rgb * 0.3162162162;
  c += texture(u_src, v_uv + o2).rgb * 0.0702702703;
  c += texture(u_src, v_uv - o2).rgb * 0.0702702703;
  f_color = vec4(c, 1.0);
}
"""
)

COMPOSITE_FS: Final = (
    VERSION
    + PALETTE
    + """
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_amount;
uniform float u_time;

void main(){
  vec3 col = texture(u_scene, v_uv).rgb;
  col += texture(u_bloom, v_uv).rgb * u_bloom_amount;

  // Filmic-ish curve. Keeps the hot rim from clipping to flat white, which is
  // what makes bloom read as light instead of as a blown-out screenshot.
  col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);

  // A single frame of dither breaks up banding in the large dark gradients,
  // which is otherwise very visible when this gets filmed off a screen.
  float d = fract(sin(dot(v_uv * 1024.0, vec2(12.9898, 78.233))) * 43758.5453);
  col += (d - 0.5) / 255.0;

  f_color = vec4(col, 1.0);
}
"""
)

BLIT_FS: Final = (
    VERSION
    + PALETTE
    + """
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform vec4 u_rect;      // xy offset, zw scale: the letterboxed 9:16 rect
void main(){
  vec2 uv = (v_uv - u_rect.xy) / u_rect.zw;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    f_color = vec4(BASE * 0.35, 1.0);
    return;
  }
  f_color = vec4(texture(u_src, uv).rgb, 1.0);
}
"""
)

# --------------------------------------------------------------------------- 2D UI

TEXT_VS: Final = (
    VERSION
    + """
in vec2 quad;
in vec4 i_rect;      // x, y, w, h in render-frame uv
in vec4 i_uv;        // atlas u, v, w, h
in vec4 i_color;
in vec2 i_params;    // x = px size (for the SDF width), y = weight bias

out vec2 v_uv;
out vec4 v_color;
out vec2 v_params;

void main(){
  // UI is authored y-down from the top-left, which is how every layout in this
  // app is written. The flip to GL's y-up clip space happens here, once.
  vec2 p = i_rect.xy + quad * i_rect.zw;
  v_uv = i_uv.xy + quad * i_uv.zw;
  v_color = i_color;
  v_params = i_params;
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
"""
)

TEXT_FS: Final = (
    VERSION
    + """
in vec2 v_uv;
in vec4 v_color;
in vec2 v_params;
out vec4 f_color;
uniform sampler2D u_atlas;

void main(){
  float d = texture(u_atlas, v_uv).r - 0.5 + v_params.y;
  // fwidth of the field gives an exactly one-pixel edge at any glyph size,
  // which is the whole reason for storing a distance field instead of coverage.
  float w = max(fwidth(d), 1e-4);
  float a = smoothstep(-w, w, d) * v_color.a;
  if (a <= 0.001) discard;
  f_color = vec4(v_color.rgb * a, a);
}
"""
)

# Rounded rectangles, rings, and the sparkline all come out of this one program.
SHAPE_VS: Final = (
    VERSION
    + """
in vec2 quad;
in vec4 i_rect;      // xy centre, zw half-size (uv)
in vec4 i_color;
in vec4 i_params;    // x corner radius, y mode, z border px, w extra

out vec2 v_local;
out vec4 v_color;
out vec4 v_params;
out vec2 v_half;

void main(){
  vec2 local = quad * 2.0 - 1.0;
  v_local = local * i_rect.zw;
  v_half = i_rect.zw;
  v_color = i_color;
  v_params = i_params;
  vec2 p = i_rect.xy + local * i_rect.zw;      // y-down, same as the text pass
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
"""
)

SHAPE_FS: Final = (
    VERSION
    + """
in vec2 v_local;
in vec4 v_color;
in vec4 v_params;
in vec2 v_half;
out vec4 f_color;
uniform float u_aspect;

float sd_round_box(vec2 p, vec2 b, float r){
  vec2 d = abs(p) - b + r;
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r;
}

void main(){
  // uv y spans 1920px against x's 1080, so one uv-y unit is u_aspect times as
  // long as one uv-x unit. Scaling y UP by u_aspect is what puts the SDF in
  // square units; dividing (the easy mistake) squashes every rounded corner
  // and turns a circle into a full-height ellipse.
  vec2 p = vec2(v_local.x, v_local.y * u_aspect);
  vec2 b = vec2(v_half.x, v_half.y * u_aspect);
  float r = min(v_params.x, min(b.x, b.y));
  float d = sd_round_box(p, b, r);

  float w = max(fwidth(d), 1e-5);
  float a;
  if (v_params.y > 0.5) {
    // ring / border mode
    a = smoothstep(w, -w, abs(d + v_params.z * 0.5) - v_params.z * 0.5);
  } else {
    a = smoothstep(w, -w, d);
  }
  a *= v_color.a;
  if (a <= 0.002) discard;
  f_color = vec4(v_color.rgb * a, a);
}
"""
)

# Progress arc used for the hold-to-lock countdown.
ARC_FS: Final = (
    VERSION
    + """
in vec2 v_local;
in vec4 v_color;
in vec4 v_params;    // x thickness, y progress 0..1, z start angle, w softness
in vec2 v_half;
out vec4 f_color;
uniform float u_aspect;

void main(){
  vec2 p = vec2(v_local.x, v_local.y * u_aspect);
  float radius = min(v_half.x, v_half.y * u_aspect);
  float r = length(p);
  float ring = abs(r - radius * 0.86) - v_params.x;
  float w = max(fwidth(ring), 1e-5);
  float a = smoothstep(w, -w, ring);

  float ang = atan(p.x, -p.y) / 6.2831853 + 0.5;   // 0 at 12 o'clock, clockwise
  a *= step(ang, v_params.y);
  a *= v_color.a;
  if (a <= 0.002) discard;
  f_color = vec4(v_color.rgb * a, a);
}
"""
)
