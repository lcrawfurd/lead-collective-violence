"""
Shared path configuration for the lead-crime project.
All paths derived from this file's location.
"""
import os
from pathlib import Path

# Project root: one level up from code/
PROJECT = Path(__file__).resolve().parent.parent

# Subdirectories
DATA      = PROJECT / "data"
OUTPUT    = PROJECT / "output"
PANELS    = OUTPUT / "panels"
FIGURES   = OUTPUT / "figures"
TABLES    = OUTPUT / "tables"
CODE      = PROJECT / "code"

# Downloaded public data shared across projects. DHS lives here, not in data/.
# Set the PUBLIC_DATA environment variable to point elsewhere (e.g. a replication run).
PUBLIC_DATA = Path(os.environ.get(
    "PUBLIC_DATA",
    Path.home() / "Library/CloudStorage/Dropbox/0 - Inbox/6_Reference/Public data"))
DHS_DIR   = PUBLIC_DATA / "DHS"

# Ensure output subdirectories exist
for _d in (PANELS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# Overleaf sync (optional — set to None if not available)
_overleaf_candidate = Path.home() / "Library/CloudStorage/Dropbox-CGDEducation/Lee Crawfurd/Apps/Overleaf/Lead and crime"
# Set NO_OVERLEAF=1 to keep all outputs inside output/ (e.g. for a replication run).
OVERLEAF = (_overleaf_candidate if _overleaf_candidate.exists() and not os.environ.get("NO_OVERLEAF")
            else None)

# Overleaf subdirectories (mirroring output structure)
OVERLEAF_FIGS = OVERLEAF / "figures" if OVERLEAF else None
OVERLEAF_TABS = OVERLEAF / "tables"  if OVERLEAF else None
if OVERLEAF_FIGS:
    OVERLEAF_FIGS.mkdir(exist_ok=True)
if OVERLEAF_TABS:
    OVERLEAF_TABS.mkdir(exist_ok=True)

# Key data files
TSIP_PATH    = DATA / "tsip_sites_cleaned.csv"
ACLED_GLOBAL = DATA / "ACLED_global.csv"   # one export, see data/ACLED_export_info.txt
ACLED_KE     = DATA / "acled_kenya.csv"
ACLED_GH     = DATA / "acled_ghana.csv"

# Constants
GRID    = 0.1
POST_YR = 2007
END_YR = 2024   # last complete year: ACLED standard access lags 12 months, so 2025 is partial
BASE_YR = POST_YR - 1


def save_to_overleaf(filename, content_or_path, subfolder=None):
    """Copy a file to the Overleaf sync folder (if available).

    Args:
        filename:        Target filename inside the Overleaf folder.
        content_or_path: Source path (str/Path) or raw string content to write.
        subfolder:       Optional subfolder inside Overleaf ('figures' or 'tables').
                         If None, copies to the Overleaf root (legacy behaviour).
    """
    if OVERLEAF is None:
        return
    import shutil
    if subfolder == "figures":
        dest_dir = OVERLEAF_FIGS
    elif subfolder == "tables":
        dest_dir = OVERLEAF_TABS
    else:
        dest_dir = OVERLEAF
    dest_dir.mkdir(exist_ok=True)
    if isinstance(content_or_path, (str, Path)):
        shutil.copy(str(content_or_path), str(dest_dir / filename))
    else:
        with open(dest_dir / filename, "w") as f:
            f.write(content_or_path)


# ── House figure style: y-axis labels horizontal, above the axis ──────────────
# Lee's rule for every figure: never a y label rotated 90 degrees. Installed as a
# save-time hook so every script that imports config gets it without changes.
# Copied from TVET Rwanda 1_Codes/_style.py (top_ylabels).
def top_ylabels(fig):
    """Move each axis's y label to a horizontal line above the plot area, left-aligned
    (right-aligned for right-hand axes), and push panel titles up to make room."""
    offset = None
    for ax in fig.get_axes():
        text = ax.get_ylabel()
        if not text or getattr(ax, "_top_ylabel_done", False):
            continue
        right = ax.yaxis.get_label_position() == "right"
        fs = ax.yaxis.label.get_fontsize()
        ax.set_ylabel("")
        ax.annotate(text, xy=(1 if right else 0, 1), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    ha="right" if right else "left", va="bottom",
                    fontsize=fs, color=ax.yaxis.label.get_color())
        ax._top_ylabel_done = True
        offset = max(offset or 0, 6 + 1.6 * fs + 4)
    if offset is not None:
        for ax in fig.get_axes():
            if any(t.get_text() for t in (ax.title, ax._left_title, ax._right_title)):
                ax._set_title_offset_trans(offset)


def direct_labels(fig, fontsize=None):
    """House rule: label lines at their right-hand end, in the line's colour, instead of a
    legend. Converts any axes or figure legend whose entries all match plotted lines.
    Legends for markers, bars, patches or maps are left alone. Labels sit in a margin added
    to the right of the axes, are spread vertically so they never overlap, and get a thin
    leader line when moved away from the line end."""
    import numpy as np
    from matplotlib.lines import Line2D
    canvas = fig.canvas
    renderer = canvas.get_renderer()

    def entries(leg):
        return [(t.get_text(), h) for t, h in zip(leg.get_texts(), leg.legend_handles)]

    fig_leg = list(fig.legends)
    for ax in fig.get_axes():
        legs = [ax.get_legend()] if ax.get_legend() else []
        legs += fig_leg
        if not legs:
            continue
        lines = {l.get_label(): l for l in ax.get_lines() if not l.get_label().startswith("_")}
        todo = []
        for leg in legs:
            ent = entries(leg)
            if not ent or not all(isinstance(h, Line2D) and lab in lines for lab, h in ent):
                continue
            todo += [lab for lab, _ in ent if lab not in todo]
            if leg is ax.get_legend():
                leg.remove()
        if not todo:
            continue
        # Same series in a panel further right on the same row? Label there instead.
        pos = ax.get_position()
        row = [o for o in fig.get_axes() if o is not ax and abs(o.get_position().y0 - pos.y0) < 0.02
               and o.get_position().x0 > pos.x0]
        for o in sorted(row, key=lambda o: -o.get_position().x0):
            olab = {l.get_label() for l in o.get_lines()}
            if all(t in olab for t in todo):
                ax = o
                lines = {l.get_label(): l for l in ax.get_lines() if not l.get_label().startswith("_")}
                break
        pos = ax.get_position()
        rightmost = not any(o is not ax and o.get_position().x0 > pos.x1 - 1e-3 and
                            abs(o.get_position().y0 - pos.y0) < 0.02 for o in fig.get_axes())
        fs = fontsize or 9
        pts = []
        for lab in todo:
            ln = lines[lab]
            x = np.asarray(ln.get_xdata(), float); y = np.asarray(ln.get_ydata(), float)
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any() or ln.get_transform() != ax.transData:
                continue
            i = np.where(ok)[0][-1]
            pts.append([lab, ln.get_color(), x[i], y[i]])
        if not pts:
            continue
        # room on the right for the widest label
        widths = [renderer.points_to_pixels(1) * 0 + fig.text(0, 0, p[0], fontsize=fs).get_window_extent(renderer).width for p in pts]
        for t in list(fig.texts)[-len(pts):]:
            t.remove()
        if not rightmost:   # make room inside the panel; a rightmost panel's labels sit outside it
            x0, x1 = ax.get_xlim()
            ax_w = ax.get_window_extent(renderer).width
            xmax_data = max(p[2] for p in pts)
            need_px = max(widths) + 24
            extra = (need_px / ax_w) * (x1 - x0) / max(1e-9, 1 - need_px / ax_w)
            ticks = [t for t in ax.get_xticks() if x0 <= t <= x1]
            ax.set_xlim(x0, max(x1, xmax_data) + extra)
            ax.set_xticks(ticks)
        # spread labels vertically in display space
        to_disp = ax.transData.transform
        disp = [to_disp((p[2], p[3]))[1] for p in pts]
        order = np.argsort(disp)
        gap = fs * 1.25 * fig.dpi / 72
        ybox = ax.get_window_extent(renderer)
        placed = {}
        last = -np.inf
        for k in order:
            y = max(disp[k], last + gap)
            placed[k] = y; last = y
        over = last - ybox.y1
        if over > 0:
            for k in placed: placed[k] -= over
        for k, p in enumerate(pts):
            dy_pts = (placed[k] - disp[k]) * 72 / fig.dpi
            ax.annotate(p[0], xy=(p[2], p[3]), xycoords="data", xytext=(8, dy_pts),
                        textcoords="offset points", ha="left", va="center", fontsize=fs,
                        color=p[1], annotation_clip=False,
                        arrowprops=(dict(arrowstyle="-", color=p[1], lw=0.5, shrinkA=0, shrinkB=2)
                                    if abs(dy_pts) > 1.5 else None))
    for leg in fig_leg:
        ents = entries(leg)
        if ents and all(isinstance(h, Line2D) for _, h in ents) and any(
                any(l.get_label() == lab for l in ax.get_lines()) for ax in fig.get_axes() for lab, _ in ents):
            leg.remove()


def _resolve_overlaps(fig):
    """After labels move above the axes: lift the figure title clear of panel titles and
    labels, and widen the gap between rows when a label runs into the row above."""
    try:
        fig.draw_without_rendering()        # position axis labels before measuring them
        r = fig.canvas.get_renderer()
    except Exception:
        return
    axes = [a for a in fig.get_axes() if a.get_visible()]
    def boxes_top(ax):  # things drawn above an axes: title, top labels
        out = [t.get_window_extent(r) for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
        out += [c.get_window_extent(r) for c in ax.texts if getattr(c, "xycoords", None) == "axes fraction"]
        return out
    def boxes_bottom(ax):  # things drawn below an axes: x label and tick labels
        out = [ax.xaxis.label.get_window_extent(r)] if ax.get_xlabel() else []
        out += [t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_text()]
        return out
    # 1. rows: a top label must not overlap the x label / ticks of an axes above it
    for _ in range(8):
        hit = False
        for a1 in axes:
            for a2 in axes:
                if a1 is a2:
                    continue
                if a2.get_position().y0 >= a1.get_position().y1 - 1e-6:   # a2 sits above a1
                    if any(t.overlaps(b) for t in boxes_top(a1) for b in boxes_bottom(a2)):
                        hit = True
        eng = fig.get_layout_engine()
        if not hit or (eng is not None and not eng.adjust_compatible):
            break
        sp = fig.subplotpars
        fig.subplots_adjust(hspace=(sp.hspace or 0.2) + 0.12)
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()
    # 2. figure title above everything in the axes' top boxes
    st = getattr(fig, "_suptitle", None)
    if st is not None and st.get_text():
        tops = [b.y1 for a in axes for b in boxes_top(a)]
        if tops:
            need = max(tops) + 4 - st.get_window_extent(r).y0
            if need > 0:
                x, y = st.get_position()
                st.set_y(y + need / fig.bbox.height)


def _install_top_ylabel_hook():
    try:
        import matplotlib.figure as _mf
    except ImportError:
        return
    if getattr(_mf.Figure.savefig, "_top_ylabel_hook", False):
        return
    _orig = _mf.Figure.savefig

    def savefig(self, *args, **kwargs):
        st = getattr(self, "_suptitle", None)       # house rule: the caption above the float is the title
        if st is not None:
            st.set_text("")
        if not getattr(self, "_direct_labels_done", False):
            direct_labels(self)
            self._direct_labels_done = True
        top_ylabels(self)
        _resolve_overlaps(self)
        return _orig(self, *args, **kwargs)
    savefig._top_ylabel_hook = True
    _mf.Figure.savefig = savefig


_install_top_ylabel_hook()
