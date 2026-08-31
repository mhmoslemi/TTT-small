"""Jupyter-ready CTA sweep figures.

The cell creates one independent figure per model, plots dog--bird CTA only,
uses rho on the x-axis, and compares only Random with greedy_J.  Nothing is
hidden in a command-line interface or ``if __name__ == "__main__"`` block.

All presentation, layout, legend, curve, marker, axis, and export controls are
collected immediately below.  The data block begins after the controls.
"""

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, NullLocator, ScalarFormatter


# =============================================================================
# USER CONTROLS — edit values in this section only
# =============================================================================

# -----------------------------------------------------------------------------
# Content and ordering
# -----------------------------------------------------------------------------
RHO_VALUES = (1, 2, 5, 10, 20, 40)
MODEL_ORDER = ("ConvNet", "ResNet20", "VGG13")
ATTACK_ORDER = ("BP", "GM", "SAPA")
SELECTION_ORDER = ("random", "greedy_j")
CLASS_PAIR = "dog--bird"                 # provenance only; never drawn

# A title is disabled by default so model/class-pair text can live in the caption.
SHOW_TITLES = False
TITLE_TEXT = {
    "ConvNet": "",
    "ResNet20": "",
    "VGG13": "",
}

# -----------------------------------------------------------------------------
# Figure canvas and axes placement
# -----------------------------------------------------------------------------
FIG_WIDTH = 14.0
FIG_HEIGHT = 8.5
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIGURE_DPI = 150                       # notebook display only
FIGURE_FACE_COLOR = "white"
AXES_FACE_COLOR = "white"
FIGURE_EDGE_COLOR = "white"
USE_CONSTRAINED_LAYOUT = False
USE_TIGHT_LAYOUT = False
TIGHT_LAYOUT_PAD = 0.40
SUBPLOT_LEFT = 0.070
SUBPLOT_RIGHT = 0.995
SUBPLOT_BOTTOM = 0.235
SUBPLOT_TOP = 0.955
SUBPLOT_WSPACE = 0.0                   # retained for easy future multi-axis use
SUBPLOT_HSPACE = 0.0

# -----------------------------------------------------------------------------
# Optional title styling (inactive while SHOW_TITLES is False)
# -----------------------------------------------------------------------------
TITLE_FONT_SIZE = 22
TITLE_FONT_WEIGHT = "bold"
TITLE_FONT_STYLE = "normal"
TITLE_FONT_COLOR = "#111111"
TITLE_ALIGNMENT = "center"
TITLE_PAD = 10
TITLE_Y = None                         # None lets Matplotlib choose

# -----------------------------------------------------------------------------
# Font family and global text rendering
# -----------------------------------------------------------------------------
FONT_FAMILY = "serif"
SERIF_FONT_FALLBACKS = (
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
)
SANS_SERIF_FONT_FALLBACKS = (
    "Arial",
    "Helvetica",
    "DejaVu Sans",
)
MONOSPACE_FONT_FALLBACKS = ("Courier New", "DejaVu Sans Mono")
MATH_FONTSET = "dejavuserif"
BASE_FONT_SIZE = 24
DEFAULT_TEXT_COLOR = "#111111"
PDF_FONT_TYPE = 42
PS_FONT_TYPE = 42
USE_TEX = False

# -----------------------------------------------------------------------------
# Axis labels
# -----------------------------------------------------------------------------
X_LABEL = r"$\rho\, (\times10^{-3})$"
Y_LABEL = "CTA (%)"
AXIS_LABEL_FONT_SIZE = 40
AXIS_LABEL_FONT_WEIGHT = "normal"
AXIS_LABEL_FONT_STYLE = "normal"
AXIS_LABEL_COLOR = "#111111"
X_LABEL_PAD = 7
Y_LABEL_PAD = 10
X_LABEL_HORIZONTAL_ALIGNMENT = "center"
Y_LABEL_VERTICAL_ALIGNMENT = "center"

# -----------------------------------------------------------------------------
# X axis
# -----------------------------------------------------------------------------
X_SCALE = "log"                        # "log" or "linear"
X_LOG_BASE = 2
X_LIMITS = (0.88, 46.0)
X_TICKS = RHO_VALUES
X_TICK_LABELS = tuple(str(value) for value in RHO_VALUES)
X_TICK_FORMATTER = "scalar"            # "scalar", "format", or None
X_TICK_FORMAT = "%g"
SHOW_X_MINOR_TICKS = False
X_MINOR_TICKS = ()
X_TICK_FONT_SIZE = 36
X_TICK_FONT_WEIGHT = "normal"
X_TICK_FONT_STYLE = "normal"
X_TICK_COLOR = "#111111"
X_TICK_ROTATION = 0
X_TICK_HORIZONTAL_ALIGNMENT = "center"
X_TICK_VERTICAL_ALIGNMENT = "top"

# -----------------------------------------------------------------------------
# Y axis — each independent model figure can have its own range and ticks
# -----------------------------------------------------------------------------
Y_LIMITS = {
    "ConvNet": (79.86, 80.64),
    "ResNet20": (83.25, 83.96),
    "VGG13": (84.55, 85.72),
}
Y_TICKS = {
    "ConvNet": (79.9, 80.1, 80.3, 80.5),
    "ResNet20": (83.3, 83.5, 83.7, 83.9),
    "VGG13": (84.7, 85.0, 85.3, 85.6),
}
Y_TICK_FORMAT = "%.1f"
SHOW_Y_MINOR_TICKS = False
Y_MINOR_TICKS = {
    "ConvNet": (),
    "ResNet20": (),
    "VGG13": (),
}
Y_TICK_FONT_SIZE = 36
Y_TICK_FONT_WEIGHT = "normal"
Y_TICK_FONT_STYLE = "normal"
Y_TICK_COLOR = "#111111"
Y_TICK_ROTATION = 0
Y_TICK_HORIZONTAL_ALIGNMENT = "right"
Y_TICK_VERTICAL_ALIGNMENT = "center"

# -----------------------------------------------------------------------------
# Tick marks and tick-label placement
# -----------------------------------------------------------------------------
TICK_DIRECTION = "out"
MAJOR_TICK_LENGTH = 12.0
MAJOR_TICK_WIDTH = 2.5
MINOR_TICK_LENGTH = 4.0
MINOR_TICK_WIDTH = 1.0
TICK_LABEL_PAD = 6.0
TICK_MARK_COLOR = "#111111"
SHOW_BOTTOM_TICKS = True
SHOW_TOP_TICKS = False
SHOW_LEFT_TICKS = True
SHOW_RIGHT_TICKS = False
SHOW_BOTTOM_TICK_LABELS = True
SHOW_TOP_TICK_LABELS = False
SHOW_LEFT_TICK_LABELS = True
SHOW_RIGHT_TICK_LABELS = False

# -----------------------------------------------------------------------------
# Spines
# -----------------------------------------------------------------------------
SPINE_VISIBILITY = {
    "left": True,
    "right": False,
    "bottom": True,
    "top": False,
}
SPINE_COLOR = "#111111"
SPINE_LINE_WIDTH = 1.6
SPINE_ZORDER = 10
SPINE_BOUNDS = {
    "left": None,
    "right": None,
    "bottom": None,
    "top": None,
}
SPINE_OUTWARD_POSITION = {
    "left": 0,
    "right": 0,
    "bottom": 0,
    "top": 0,
}

# -----------------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------------
SHOW_GRID = True
GRID_AXIS = "y"                         # "x", "y", or "both"
GRID_WHICH = "major"                    # "major", "minor", or "both"
GRID_COLOR = "#D5D8DC"
GRID_LINE_STYLE = "-"
GRID_LINE_WIDTH = 1.05
GRID_ALPHA = 0.62
GRID_ZORDER = 0
AXIS_BELOW_GRID = True

# -----------------------------------------------------------------------------
# Attack identity: color and marker shape
# Okabe--Ito-derived colors remain distinguishable in print and for common CVD.
# -----------------------------------------------------------------------------
# Keep this as None to use your custom ATTACK_COLORS mapping below, or replace
# None with one palette name from ATTACK_COLOR_PALETTE_OPTIONS.
SELECTED_ATTACK_COLOR_PALETTE = None

# Ten optional three-color sets.
ATTACK_COLOR_PALETTE_OPTIONS = {
    "option_01_okabe_ito": {
        "BP": "#0072B2", "GM": "#D55E00", "SAPA": "#009E73",
    },
    "option_02_tol_bright": {
        "BP": "#4477AA", "GM": "#EE6677", "SAPA": "#228833",
    },
    "option_03_vibrant": {
        "BP": "#0077BB", "GM": "#EE3377", "SAPA": "#EE7733",
    },
    "option_04_ibm": {
        "BP": "#648FFF", "GM": "#DC267F", "SAPA": "#FFB000",
    },
    "option_05_tableau": {
        "BP": "#4E79A7", "GM": "#E15759", "SAPA": "#59A14F",
    },
    "option_06_dark2": {
        "BP": "#1B9E77", "GM": "#D95F02", "SAPA": "#7570B3",
    },
    "option_07_cool_warm_gold": {
        "BP": "#3B4CC0", "GM": "#B40426", "SAPA": "#F2C14E",
    },
    "option_08_earth": {
        "BP": "#264653", "GM": "#E76F51", "SAPA": "#2A9D8F",
    },
    "option_09_navy_rose_amber": {
        "BP": "#2F4B7C", "GM": "#D45087", "SAPA": "#FFA600",
    },
    "option_10_purple_red_green": {
        "BP": "#6A3D9A", "GM": "#E31A1C", "SAPA": "#33A02C",
    },
}

ATTACK_COLORS = {
    "BP": "#0072B2",                  # blue
    "GM": "#D55E00",                  # vermilion
    "SAPA": "#009E73",                # bluish green
}
if SELECTED_ATTACK_COLOR_PALETTE is not None:
    ATTACK_COLORS = dict(
        ATTACK_COLOR_PALETTE_OPTIONS[SELECTED_ATTACK_COLOR_PALETTE]
    )

ATTACK_MARKERS = {
    "BP": "o",
    "GM": "s",
    "SAPA": "^",
}
ATTACK_LABELS = {
    "BP": "BP",
    "GM": "GM",
    "SAPA": "SAPA",
}
ATTACK_ALPHA = {
    "BP": 1.0,
    "GM": 1.0,
    "SAPA": 1.0,
}
ATTACK_ZORDER_OFFSET = {
    "BP": 0,
    "GM": 0,
    "SAPA": 0,
}

# -----------------------------------------------------------------------------
# Selection identity: line style and marker fill
# Use a Matplotlib linestyle string or a dash tuple, e.g. (0, (5, 2)).
# Marker face can be "attack" to inherit its attack color, or any color string.
# -----------------------------------------------------------------------------
SELECTION_LINE_STYLES = {
    "random": (0, (5.0, 2.2)),
    "greedy_j": "-",
}
SELECTION_LABELS = {
    "random": "Random",
    "greedy_j": r"greedy$_J$",
}

# Legend-only aliases. Change the values freely; data keys and plotting logic
# remain "random" and "greedy_j" regardless of what is displayed in the legend.
LEGEND_SELECTION_ALIASES = {
    "random": "Random",
    "greedy_j": r"greedy$_J$",
}
SELECTION_MARKER_FACES = {
    "random": "white",
    "greedy_j": "attack",
}
SELECTION_MARKER_EDGE_COLORS = {
    "random": "attack",
    "greedy_j": "attack",
}
SELECTION_LINE_WIDTHS = {
    "random": 4.0,
    "greedy_j": 4.4,
}
SELECTION_MARKER_SIZES = {
    "random": 18.0,
    "greedy_j": 18.5,
}
SELECTION_MARKER_EDGE_WIDTHS = {
    "random": 3.0,
    "greedy_j": 0.8,
}
SELECTION_ALPHA = {
    "random": 0.96,
    "greedy_j": 1.0,
}
SELECTION_ZORDER = {
    "random": 4,
    "greedy_j": 5,
}

# -----------------------------------------------------------------------------
# Curve and marker mechanics
# -----------------------------------------------------------------------------
LINE_DRAW_STYLE = "default"
LINE_SOLID_CAP_STYLE = "round"
LINE_DASH_CAP_STYLE = "round"
LINE_SOLID_JOIN_STYLE = "round"
LINE_DASH_JOIN_STYLE = "round"
MARK_EVERY = 1
CLIP_LINES_TO_AXES = True
ANTIALIASED_LINES = True
PICK_RADIUS = 5
MARKER_FILL_STYLE = "full"

# -----------------------------------------------------------------------------
# Legend — drawn only inside the third figure by default
# -----------------------------------------------------------------------------
SHOW_LEGEND = True
LEGEND_MODEL = "VGG13"
LEGEND_LOCATION = "lower center"
LEGEND_BBOX_TO_ANCHOR = (0.27, 0.018)  # axes coordinates; positive y is inside
LEGEND_BBOX_TRANSFORM = "axes"         # "axes" or "figure"
LEGEND_NUMBER_OF_COLUMNS = 2
LEGEND_MODE = None                     # use "expand" to stretch across the axes
LEGEND_ALIGNMENT = "center"
LEGEND_FONT_SIZE = 34
LEGEND_FONT_WEIGHT = "bold"
LEGEND_FONT_STYLE = "normal"
LEGEND_TEXT_COLOR = "#111111"
LEGEND_TITLE = None
LEGEND_TITLE_FONT_SIZE = 25
LEGEND_TITLE_FONT_WEIGHT = "bold"
LEGEND_FRAME_ON = True
LEGEND_FRAME_FACE_COLOR = "white"
LEGEND_FRAME_EDGE_COLOR = "#B8BDC3"
LEGEND_FRAME_ALPHA = 0.0
LEGEND_FRAME_LINE_WIDTH = 0.0
LEGEND_FANCY_BOX = True
LEGEND_SHADOW = False
LEGEND_BORDER_AXES_PAD = 0.25
LEGEND_BORDER_PAD = 0.38
LEGEND_LABEL_SPACING = 0.35
LEGEND_COLUMN_SPACING = 1.15
LEGEND_HANDLE_LENGTH = 2.15
LEGEND_HANDLE_HEIGHT = 0.75
LEGEND_HANDLE_TEXT_PAD = 0.45
LEGEND_MARKER_SCALE = 1.0
LEGEND_REVERSE_ORDER = False
LEGEND_ZORDER = 30

# Appearance of the five proxy handles: BP, GM, SAPA, Random, greedy_J.
LEGEND_ATTACK_HANDLE_LINE_STYLE = "-"
LEGEND_ATTACK_HANDLE_LINE_WIDTH = 4
LEGEND_ATTACK_HANDLE_MARKER_SIZE = 18.0
LEGEND_ATTACK_HANDLE_MARKER_FACE = "attack"
LEGEND_ATTACK_HANDLE_MARKER_EDGE_WIDTH = 1.7
LEGEND_SELECTION_HANDLE_COLOR = "#3A3A3A"
LEGEND_SELECTION_HANDLE_MARKER = "o"
LEGEND_SELECTION_HANDLE_MARKER_SIZE = 18.0
LEGEND_SELECTION_HANDLE_LINE_WIDTH = 4.0
LEGEND_SELECTION_HANDLE_MARKER_EDGE_WIDTH = 2.7

# -----------------------------------------------------------------------------
# Export and notebook display
# -----------------------------------------------------------------------------
SAVE_PDF = True
SAVE_PNG = False
SHOW_PLOTS = True
CLOSE_FIGURES_AFTER_SHOW = False
CREATE_OUTPUT_DIRECTORIES = True
PDF_OUTPUT_DIRECTORY = Path("figures/pdf")
PNG_OUTPUT_DIRECTORY = Path("figures/png")
OUTPUT_FILENAME_PREFIX = "cta"
OUTPUT_FILENAME_SUFFIX = ""
MODEL_FILENAME_SLUGS = {
    "ConvNet": "convnet",
    "ResNet20": "resnet20",
    "VGG13": "vgg13",
}
PDF_EXTENSION = ".pdf"
PNG_EXTENSION = ".png"
SAVE_DPI = 100
SAVE_BBOX_INCHES = "tight"             # None disables tight cropping
SAVE_PAD_INCHES = 0.0
SAVE_FACE_COLOR = "white"
SAVE_EDGE_COLOR = "white"
SAVE_TRANSPARENT = False
SAVE_ORIENTATION = "landscape"
SAVE_PDF_METADATA = {
    "Creator": "Matplotlib",
    "Title": "CTA perturbation-budget sweep",
}
SAVE_PNG_METADATA = {
    "Software": "Matplotlib",
}

# -----------------------------------------------------------------------------
# Validation behavior
# -----------------------------------------------------------------------------
VALIDATE_DATA = True
REQUIRE_FINITE_VALUES = True
EXPECTED_NUMBER_OF_RHO_VALUES = 6


# =============================================================================
# DATA — dog--bird CTA (%) only; Random and greedy_J only
# =============================================================================

CTA = {
    "ConvNet": {
        ("BP", "random"): (80.14, 80.13, 80.11, 80.10, 80.09, 79.98),
        ("BP", "greedy_j"): (80.14, 80.13, 80.22, 80.23, 80.13, 79.92),
        ("GM", "random"): (80.34, 80.38, 80.46, 80.46, 80.38, 80.29),
        ("GM", "greedy_j"): (80.56, 80.39, 80.47, 80.59, 80.53, 80.36),
        ("SAPA", "random"): (80.43, 80.39, 80.30, 80.36, 80.31, 80.30),
        ("SAPA", "greedy_j"): (80.50, 80.42, 80.47, 80.55, 80.57, 80.38),
    },
    "ResNet20": {
        ("BP", "random"): (83.80, 83.73, 83.71, 83.62, 83.56, 83.47),
        ("BP", "greedy_j"): (83.74, 83.73, 83.77, 83.76, 83.81, 83.34),
        ("GM", "random"): (83.76, 83.88, 83.79, 83.73, 83.67, 83.55),
        ("GM", "greedy_j"): (83.86, 83.74, 83.80, 83.81, 83.81, 83.38),
        ("SAPA", "random"): (83.76, 83.63, 83.85, 83.74, 83.62, 83.51),
        ("SAPA", "greedy_j"): (83.78, 83.72, 83.71, 83.75, 83.84, 83.43),
    },
    "VGG13": {
        ("BP", "random"): (85.34, 85.25, 85.42, 85.30, 85.32, 85.02),
        ("BP", "greedy_j"): (85.35, 85.44, 85.56, 85.27, 85.22, 84.71),
        ("GM", "random"): (85.52, 85.54, 85.49, 85.45, 85.29, 85.05),
        ("GM", "greedy_j"): (85.43, 85.44, 85.50, 85.42, 85.43, 84.93),
        ("SAPA", "random"): (85.53, 85.59, 85.49, 85.36, 85.33, 85.04),
        ("SAPA", "greedy_j"): (85.36, 85.42, 85.46, 85.46, 85.25, 84.84),
    },
}


# =============================================================================
# PLOTTING IMPLEMENTATION — normally no edits are needed below this line
# =============================================================================

def _configure_matplotlib():
    """Apply the typography and export controls above."""
    mpl.rcParams.update(
        {
            "figure.facecolor": FIGURE_FACE_COLOR,
            "figure.edgecolor": FIGURE_EDGE_COLOR,
            "axes.facecolor": AXES_FACE_COLOR,
            "font.family": FONT_FAMILY,
            "font.serif": list(SERIF_FONT_FALLBACKS),
            "font.sans-serif": list(SANS_SERIF_FONT_FALLBACKS),
            "font.monospace": list(MONOSPACE_FONT_FALLBACKS),
            "font.size": BASE_FONT_SIZE,
            "text.color": DEFAULT_TEXT_COLOR,
            "text.usetex": USE_TEX,
            "mathtext.fontset": MATH_FONTSET,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.labelweight": AXIS_LABEL_FONT_WEIGHT,
            "axes.labelcolor": AXIS_LABEL_COLOR,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.titleweight": TITLE_FONT_WEIGHT,
            "axes.titlecolor": TITLE_FONT_COLOR,
            "axes.edgecolor": SPINE_COLOR,
            "axes.linewidth": SPINE_LINE_WIDTH,
            "xtick.labelsize": X_TICK_FONT_SIZE,
            "xtick.color": X_TICK_COLOR,
            "ytick.labelsize": Y_TICK_FONT_SIZE,
            "ytick.color": Y_TICK_COLOR,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.frameon": LEGEND_FRAME_ON,
            "legend.fancybox": LEGEND_FANCY_BOX,
            "legend.shadow": LEGEND_SHADOW,
            "lines.antialiased": ANTIALIASED_LINES,
            "pdf.fonttype": PDF_FONT_TYPE,
            "ps.fonttype": PS_FONT_TYPE,
            "savefig.dpi": SAVE_DPI,
            "savefig.facecolor": SAVE_FACE_COLOR,
            "savefig.edgecolor": SAVE_EDGE_COLOR,
            "savefig.transparent": SAVE_TRANSPARENT,
            "savefig.bbox": SAVE_BBOX_INCHES,
            "savefig.pad_inches": SAVE_PAD_INCHES,
        }
    )


def _validate_data():
    """Fail early if a manually edited series no longer matches the x axis."""
    if len(RHO_VALUES) != EXPECTED_NUMBER_OF_RHO_VALUES:
        raise ValueError(
            "RHO_VALUES has "
            f"{len(RHO_VALUES)} entries; expected {EXPECTED_NUMBER_OF_RHO_VALUES}."
        )

    for model in MODEL_ORDER:
        if model not in CTA:
            raise KeyError(f"Missing model in CTA: {model}")
        for attack in ATTACK_ORDER:
            for selection in SELECTION_ORDER:
                key = (attack, selection)
                if key not in CTA[model]:
                    raise KeyError(f"Missing CTA series: {model}, {key}")
                values = CTA[model][key]
                if len(values) != len(RHO_VALUES):
                    raise ValueError(
                        f"{model} {key} has {len(values)} values; "
                        f"expected {len(RHO_VALUES)}."
                    )
                if REQUIRE_FINITE_VALUES:
                    for value in values:
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise ValueError(f"Non-finite CTA value in {model} {key}: {value}")


def _resolved_marker_color(setting, attack):
    """Translate the convenient 'attack' sentinel into the attack's color."""
    return ATTACK_COLORS[attack] if setting == "attack" else setting


def _style_spines(ax):
    for side, spine in ax.spines.items():
        spine.set_visible(SPINE_VISIBILITY[side])
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_LINE_WIDTH)
        spine.set_zorder(SPINE_ZORDER)
        spine.set_position(("outward", SPINE_OUTWARD_POSITION[side]))
        if SPINE_BOUNDS[side] is not None:
            spine.set_bounds(*SPINE_BOUNDS[side])


def _style_ticks(ax, model):
    ax.tick_params(
        axis="both",
        which="major",
        direction=TICK_DIRECTION,
        length=MAJOR_TICK_LENGTH,
        width=MAJOR_TICK_WIDTH,
        pad=TICK_LABEL_PAD,
        color=TICK_MARK_COLOR,
        bottom=SHOW_BOTTOM_TICKS,
        top=SHOW_TOP_TICKS,
        left=SHOW_LEFT_TICKS,
        right=SHOW_RIGHT_TICKS,
        labelbottom=SHOW_BOTTOM_TICK_LABELS,
        labeltop=SHOW_TOP_TICK_LABELS,
        labelleft=SHOW_LEFT_TICK_LABELS,
        labelright=SHOW_RIGHT_TICK_LABELS,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction=TICK_DIRECTION,
        length=MINOR_TICK_LENGTH,
        width=MINOR_TICK_WIDTH,
        color=TICK_MARK_COLOR,
    )
    ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE, labelcolor=X_TICK_COLOR)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE, labelcolor=Y_TICK_COLOR)

    for label in ax.get_xticklabels():
        label.set_fontweight(X_TICK_FONT_WEIGHT)
        label.set_fontstyle(X_TICK_FONT_STYLE)
        label.set_rotation(X_TICK_ROTATION)
        label.set_horizontalalignment(X_TICK_HORIZONTAL_ALIGNMENT)
        label.set_verticalalignment(X_TICK_VERTICAL_ALIGNMENT)

    for label in ax.get_yticklabels():
        label.set_fontweight(Y_TICK_FONT_WEIGHT)
        label.set_fontstyle(Y_TICK_FONT_STYLE)
        label.set_rotation(Y_TICK_ROTATION)
        label.set_horizontalalignment(Y_TICK_HORIZONTAL_ALIGNMENT)
        label.set_verticalalignment(Y_TICK_VERTICAL_ALIGNMENT)

    if SHOW_X_MINOR_TICKS:
        ax.set_xticks(X_MINOR_TICKS, minor=True)
    else:
        ax.xaxis.set_minor_locator(NullLocator())

    if SHOW_Y_MINOR_TICKS:
        ax.set_yticks(Y_MINOR_TICKS[model], minor=True)
    else:
        ax.yaxis.set_minor_locator(NullLocator())


def _legend_handles():
    handles = []

    for attack in ATTACK_ORDER:
        attack_face = _resolved_marker_color(
            LEGEND_ATTACK_HANDLE_MARKER_FACE,
            attack,
        )
        handles.append(
            Line2D(
                [],
                [],
                label=ATTACK_LABELS[attack],
                color=ATTACK_COLORS[attack],
                linestyle=LEGEND_ATTACK_HANDLE_LINE_STYLE,
                linewidth=LEGEND_ATTACK_HANDLE_LINE_WIDTH,
                marker=ATTACK_MARKERS[attack],
                markersize=LEGEND_ATTACK_HANDLE_MARKER_SIZE,
                markerfacecolor=attack_face,
                markeredgecolor=ATTACK_COLORS[attack],
                markeredgewidth=LEGEND_ATTACK_HANDLE_MARKER_EDGE_WIDTH,
            )
        )

    for selection in SELECTION_ORDER:
        neutral_face = SELECTION_MARKER_FACES[selection]
        if neutral_face == "attack":
            neutral_face = LEGEND_SELECTION_HANDLE_COLOR
        neutral_edge = SELECTION_MARKER_EDGE_COLORS[selection]
        if neutral_edge == "attack":
            neutral_edge = LEGEND_SELECTION_HANDLE_COLOR
        handles.append(
            Line2D(
                [],
                [],
                label=LEGEND_SELECTION_ALIASES[selection],
                color=LEGEND_SELECTION_HANDLE_COLOR,
                linestyle=SELECTION_LINE_STYLES[selection],
                linewidth=LEGEND_SELECTION_HANDLE_LINE_WIDTH,
                marker=LEGEND_SELECTION_HANDLE_MARKER,
                markersize=LEGEND_SELECTION_HANDLE_MARKER_SIZE,
                markerfacecolor=neutral_face,
                markeredgecolor=neutral_edge,
                markeredgewidth=LEGEND_SELECTION_HANDLE_MARKER_EDGE_WIDTH,
            )
        )

    return list(reversed(handles)) if LEGEND_REVERSE_ORDER else handles


def _add_legend(fig, ax):
    bbox_transform = ax.transAxes if LEGEND_BBOX_TRANSFORM == "axes" else fig.transFigure
    legend = ax.legend(
        handles=_legend_handles(),
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        bbox_transform=bbox_transform,
        ncol=LEGEND_NUMBER_OF_COLUMNS,
        mode=LEGEND_MODE,
        alignment=LEGEND_ALIGNMENT,
        title=LEGEND_TITLE,
        frameon=LEGEND_FRAME_ON,
        fancybox=LEGEND_FANCY_BOX,
        shadow=LEGEND_SHADOW,
        facecolor=LEGEND_FRAME_FACE_COLOR,
        edgecolor=LEGEND_FRAME_EDGE_COLOR,
        framealpha=LEGEND_FRAME_ALPHA,
        borderaxespad=LEGEND_BORDER_AXES_PAD,
        borderpad=LEGEND_BORDER_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
        columnspacing=LEGEND_COLUMN_SPACING,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        markerscale=LEGEND_MARKER_SCALE,
        prop={
            "size": LEGEND_FONT_SIZE,
            "weight": LEGEND_FONT_WEIGHT,
            "style": LEGEND_FONT_STYLE,
        },
    )
    legend.set_zorder(LEGEND_ZORDER)
    legend.get_frame().set_linewidth(LEGEND_FRAME_LINE_WIDTH)
    for text_item in legend.get_texts():
        text_item.set_color(LEGEND_TEXT_COLOR)
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(LEGEND_TITLE_FONT_SIZE)
        legend.get_title().set_fontweight(LEGEND_TITLE_FONT_WEIGHT)
        legend.get_title().set_color(LEGEND_TEXT_COLOR)
    return legend


def _output_stem(model):
    suffix = f"_{OUTPUT_FILENAME_SUFFIX}" if OUTPUT_FILENAME_SUFFIX else ""
    return f"{OUTPUT_FILENAME_PREFIX}_{MODEL_FILENAME_SLUGS[model]}{suffix}"


def _save_figure(fig, model):
    stem = _output_stem(model)
    common = {
        "dpi": SAVE_DPI,
        "bbox_inches": SAVE_BBOX_INCHES,
        "pad_inches": SAVE_PAD_INCHES,
        "facecolor": SAVE_FACE_COLOR,
        "edgecolor": SAVE_EDGE_COLOR,
        "transparent": SAVE_TRANSPARENT,
        "orientation": SAVE_ORIENTATION,
    }

    if SAVE_PDF:
        if CREATE_OUTPUT_DIRECTORIES:
            PDF_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            PDF_OUTPUT_DIRECTORY / f"{stem}{PDF_EXTENSION}",
            metadata=SAVE_PDF_METADATA,
            **common,
        )

    if SAVE_PNG:
        if CREATE_OUTPUT_DIRECTORIES:
            PNG_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            PNG_OUTPUT_DIRECTORY / f"{stem}{PNG_EXTENSION}",
            metadata=SAVE_PNG_METADATA,
            **common,
        )


def plot_cta_figures():
    """Create and optionally save three separate figures; return both mappings."""
    if VALIDATE_DATA:
        _validate_data()
    _configure_matplotlib()

    figures = {}
    axes = {}

    for model in MODEL_ORDER:
        fig, ax = plt.subplots(
            figsize=FIG_SIZE,
            dpi=FIGURE_DPI,
            constrained_layout=USE_CONSTRAINED_LAYOUT,
            facecolor=FIGURE_FACE_COLOR,
            edgecolor=FIGURE_EDGE_COLOR,
        )
        ax.set_facecolor(AXES_FACE_COLOR)

        for attack in ATTACK_ORDER:
            for selection in SELECTION_ORDER:
                marker_face = _resolved_marker_color(
                    SELECTION_MARKER_FACES[selection],
                    attack,
                )
                marker_edge = _resolved_marker_color(
                    SELECTION_MARKER_EDGE_COLORS[selection],
                    attack,
                )
                ax.plot(
                    RHO_VALUES,
                    CTA[model][(attack, selection)],
                    color=ATTACK_COLORS[attack],
                    linestyle=SELECTION_LINE_STYLES[selection],
                    linewidth=SELECTION_LINE_WIDTHS[selection],
                    alpha=ATTACK_ALPHA[attack] * SELECTION_ALPHA[selection],
                    marker=ATTACK_MARKERS[attack],
                    markersize=SELECTION_MARKER_SIZES[selection],
                    markerfacecolor=marker_face,
                    markeredgecolor=marker_edge,
                    markeredgewidth=SELECTION_MARKER_EDGE_WIDTHS[selection],
                    fillstyle=MARKER_FILL_STYLE,
                    drawstyle=LINE_DRAW_STYLE,
                    solid_capstyle=LINE_SOLID_CAP_STYLE,
                    dash_capstyle=LINE_DASH_CAP_STYLE,
                    solid_joinstyle=LINE_SOLID_JOIN_STYLE,
                    dash_joinstyle=LINE_DASH_JOIN_STYLE,
                    markevery=MARK_EVERY,
                    clip_on=CLIP_LINES_TO_AXES,
                    antialiased=ANTIALIASED_LINES,
                    picker=PICK_RADIUS,
                    zorder=(
                        SELECTION_ZORDER[selection]
                        + ATTACK_ZORDER_OFFSET[attack]
                    ),
                )

        if X_SCALE == "log":
            ax.set_xscale("log", base=X_LOG_BASE)
        else:
            ax.set_xscale(X_SCALE)
        ax.set_xlim(*X_LIMITS)
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels(X_TICK_LABELS)
        if X_TICK_FORMATTER == "scalar":
            ax.xaxis.set_major_formatter(ScalarFormatter())
        elif X_TICK_FORMATTER == "format":
            ax.xaxis.set_major_formatter(FormatStrFormatter(X_TICK_FORMAT))

        ax.set_ylim(*Y_LIMITS[model])
        ax.set_yticks(Y_TICKS[model])
        ax.yaxis.set_major_formatter(FormatStrFormatter(Y_TICK_FORMAT))

        ax.set_xlabel(
            X_LABEL,
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight=AXIS_LABEL_FONT_WEIGHT,
            fontstyle=AXIS_LABEL_FONT_STYLE,
            color=AXIS_LABEL_COLOR,
            labelpad=X_LABEL_PAD,
            loc=X_LABEL_HORIZONTAL_ALIGNMENT,
        )
        ax.set_ylabel(
            Y_LABEL,
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight=AXIS_LABEL_FONT_WEIGHT,
            fontstyle=AXIS_LABEL_FONT_STYLE,
            color=AXIS_LABEL_COLOR,
            labelpad=Y_LABEL_PAD,
            loc=Y_LABEL_VERTICAL_ALIGNMENT,
        )

        if SHOW_TITLES:
            title_kwargs = {
                "fontsize": TITLE_FONT_SIZE,
                "fontweight": TITLE_FONT_WEIGHT,
                "fontstyle": TITLE_FONT_STYLE,
                "color": TITLE_FONT_COLOR,
                "loc": TITLE_ALIGNMENT,
                "pad": TITLE_PAD,
            }
            if TITLE_Y is not None:
                title_kwargs["y"] = TITLE_Y
            ax.set_title(TITLE_TEXT[model], **title_kwargs)

        if SHOW_GRID:
            ax.grid(
                visible=True,
                which=GRID_WHICH,
                axis=GRID_AXIS,
                color=GRID_COLOR,
                linestyle=GRID_LINE_STYLE,
                linewidth=GRID_LINE_WIDTH,
                alpha=GRID_ALPHA,
                zorder=GRID_ZORDER,
            )
        else:
            ax.grid(False)
        ax.set_axisbelow(AXIS_BELOW_GRID)

        _style_spines(ax)
        _style_ticks(ax, model)

        fig.subplots_adjust(
            left=SUBPLOT_LEFT,
            right=SUBPLOT_RIGHT,
            bottom=SUBPLOT_BOTTOM,
            top=SUBPLOT_TOP,
            wspace=SUBPLOT_WSPACE,
            hspace=SUBPLOT_HSPACE,
        )
        if USE_TIGHT_LAYOUT:
            fig.tight_layout(pad=TIGHT_LAYOUT_PAD)

        if SHOW_LEGEND and model == LEGEND_MODEL:
            _add_legend(fig, ax)

        figures[model] = fig
        axes[model] = ax
        _save_figure(fig, model)

    if SHOW_PLOTS:
        plt.show()
    if CLOSE_FIGURES_AFTER_SHOW:
        for figure in figures.values():
            plt.close(figure)

    return figures, axes


# Jupyter execution line: creates three independent figures and exposes their
# objects for later notebook edits.  There is deliberately no main/CLI wrapper.
figures, axes = plot_cta_figures()
