# pvp/utils/waypoint_viz.py
from panda3d.core import (
    TextNode, LineSegs, Vec4, TransparencyAttrib, ColorAttrib
)

__all__ = ["drop_waypoint_markers", "clear_waypoint_markers"]

def clear_waypoint_markers(env):
    """Remove previously created waypoint markers, line, and the debug parent (if any)."""
    try:
        if hasattr(env, "_wp_markers"):
            for np in env._wp_markers:
                try:
                    np.removeNode()
                except Exception:
                    pass
            env._wp_markers = []
        if hasattr(env, "_wp_line") and env._wp_line is not None:
            try:
                env._wp_line.removeNode()
            except Exception:
                pass
            env._wp_line = None
        if hasattr(env, "_wp_debug_np") and env._wp_debug_np is not None and not env._wp_debug_np.isEmpty():
            try:
                env._wp_debug_np.removeNode()
            except Exception:
                pass
            env._wp_debug_np = None
    except Exception:
        pass

def _ensure_debug_parent(env):
    """Create a dedicated unshaded parent so child nodes render with flat color (no shader/lighting)."""
    if not hasattr(env, "_wp_debug_np") or env._wp_debug_np is None or env._wp_debug_np.isEmpty():
        parent = env.engine.render.attachNewNode("debug_waypoints")
        parent.setShaderOff(1)
        parent.setLightOff(1)
        parent.setTextureOff(1)
        parent.setMaterialOff(1)
        try:
            parent.setColorScaleOff(1)
        except Exception:
            parent.clearColorScale()
        parent.setTransparency(TransparencyAttrib.M_alpha)
        env._wp_debug_np = parent
    return env._wp_debug_np

def drop_waypoint_markers(env, coords, color=(0, 0, 1, 1), scale=0.4, label=True):
    """
    Draw persistent markers at coords=[(x,y,z), ...] and connect them with a line.
    Must be called after env.reset() and when env.engine.render exists.
    """
    if not hasattr(env, "engine") or getattr(env.engine, "render", None) is None:
        print("Engine/render not ready yet — cannot drop markers.")
        return

    clear_waypoint_markers(env)
    parent = _ensure_debug_parent(env)
    env._wp_markers = []

    rgba = Vec4(float(color[0]), float(color[1]), float(color[2]), float(color[3]))

    def _load_marker_model():
        for candidate in ("models/box", "models/smiley", "models/ball", "box"):
            try:
                m = env.engine.loader.loadModel(candidate)
                if m is not None:
                    return m
            except Exception:
                continue
        return None

    model_template = _load_marker_model()

    for i, (x, y, z) in enumerate(coords):
        if model_template is not None:
            marker = model_template.copyTo(parent)
            marker.setScale(scale)
            marker.setPos(float(x), float(y), float(z) + 0.12)
            marker.clearColorScale()
            marker.setAttrib(ColorAttrib.makeFlat(rgba), 1000)  # force flat color
            if rgba[3] < 1.0:
                marker.setTransparency(TransparencyAttrib.M_alpha)
            env._wp_markers.append(marker)
        else:
            tn = TextNode(f"wp{i}")
            tn.setText("■")
            tn.setTextColor(rgba[0], rgba[1], rgba[2], rgba[3])
            tn_np = parent.attachNewNode(tn)
            tn_np.setScale(0.1)
            tn_np.setPos(float(x), float(y), float(z) + 0.15)
            env._wp_markers.append(tn_np)

        if label:
            tn = TextNode(f"wp_label_{i}")
            tn.setText(str(i))
            tn.setAlign(TextNode.A_center)
            tn.setTextColor(1, 1, 1, 1)
            text_np = parent.attachNewNode(tn)
            text_np.setScale(0.5)
            text_np.setPos(float(x), float(y), float(z) + 0.6)
            env._wp_markers.append(text_np)

    if len(coords) >= 2:
        ls = LineSegs()
        ls.setThickness(3.0)
        ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        first = True
        for (x, y, z) in coords:
            px, py, pz = float(x), float(y), float(z) + 0.15
            if first:
                ls.moveTo(px, py, pz); first = False
            else:
                ls.drawTo(px, py, pz)
        line_np = parent.attachNewNode(ls.create())
        line_np.setAttrib(ColorAttrib.makeFlat(rgba), 1000)
        if rgba[3] < 1.0:
            line_np.setTransparency(TransparencyAttrib.M_alpha)
        env._wp_line = line_np
    else:
        env._wp_line = None

    print(f"Dropped {len(coords)} waypoint markers and connected line.")
