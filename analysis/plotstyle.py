"""
plotstyle.py
------------
Matplotlib settings for publication figures (NIM A single-column width).

Conventions
-----------
* No figure titles — captions belong in the LaTeX source, not in the image.
* Vector PDF is the primary output; PNG is written alongside for quick viewing.
* Serif text at a size that stays legible after the journal scales the figure
  to column width.
* One colour and one marker per overvoltage, used consistently everywhere.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# NIM A single column is 3.35 in; double column 6.9 in.
COLUMN_WIDTH = 3.35
DOUBLE_WIDTH = 6.9

OV_STYLE = {
    'OVfive': dict(label='2.5 V OV', color='#1f5fa8', marker='o'),
    'OVfour': dict(label='4.0 V OV', color='#d1600f', marker='s'),
}


def use_publication_style():
    mpl.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset':   'dejavuserif',
        'font.size':           9,
        'axes.labelsize':      9,
        'axes.titlesize':      9,
        'legend.fontsize':     7.5,
        'xtick.labelsize':     8,
        'ytick.labelsize':     8,
        'axes.linewidth':      0.8,
        'lines.linewidth':     1.3,
        'lines.markersize':    3.5,
        'xtick.direction':     'in',
        'ytick.direction':     'in',
        'xtick.top':           True,
        'ytick.right':         True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'legend.frameon':      True,
        'legend.framealpha':   0.9,
        'legend.edgecolor':    '0.7',
        'grid.linewidth':      0.4,
        'grid.alpha':          0.4,
        'figure.dpi':          150,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.02,
        'errorbar.capsize':    1.5,
    })


def save(fig, path_no_ext):
    """Write both PDF (for the manuscript) and PNG (for quick inspection)."""
    fig.savefig(f"{path_no_ext}.pdf")
    fig.savefig(f"{path_no_ext}.png")
    return f"{path_no_ext}.pdf"


def grid(ax, which='both'):
    ax.grid(True, which=which, linestyle='--', linewidth=0.4, alpha=0.4)
