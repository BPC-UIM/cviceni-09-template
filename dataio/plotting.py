"""Vykreslovani vysledku cviceni 09 -- vse PREDVYPLNENE.

Dve veci se kresli:

* ``NetworkVisualizer`` -- rozhodovaci oblast cele site v rovine ``x1-x2``
  (adaptace ``DecisionBoundaryPlotter`` z Cviceni 08). Navic zakresli primky
  prvni skryte vrstvy (sloupce matice vah) a umi zobrazit aktivaci kazde
  skryte jednotky zvlast -- cte ji ze stopy ``Sequential.io_``.
* ``plot_space_transformation`` -- **klicovy obrazek cviceni**: dva panely
  vedle sebe, body pred vrstvou (``io[0]``) a tytez body po vrstve
  (``io[1]``), obarvene podle stitku. Je videt, ze data neseparovatelna
  v prostoru ``x1-x2`` mohou byt separovatelna v prostoru ``y1-y2``.

Aby modul sel importovat i pred dokoncenim ``src/``, typ ``Sequential`` je
referencovan jen jako retezcovy type hint pod ``TYPE_CHECKING`` -- ``dataio``
za behu ``src`` neimportuje. Sit se pouziva jen pres rozhrani: volani
``network(x)``, atributy ``network.layers`` a ``network.io_`` a u prvni
vrstvy ``layers[0].weights`` / ``layers[0].bias``.

Vsechny funkce pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni nadrazeneho adresare) a
vzdy ji zavrou. Funkce ``plt.show`` se nikdy nevola.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

if TYPE_CHECKING:
    from src.network import Sequential

# --- Paleta (overena pro barvoslepost; trida nikdy neni odlisena jen barvou,
#     vzdy ma i legendu) ------------------------------------------------------
_CLASS_COLORS: dict[int, str] = {0: "#2a78d6", 1: "#eb6834"}   # modra / oranzova
_CLASS_TINTS: tuple[str, str] = ("#d5e5f8", "#fadccf")           # svetle podbarveni oblasti
_OTHER_COLOR = "#898781"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_HAIRLINE = "#c3c2b7"

# Mekky vystup site: trida 0 (modra) -> neutralni seda (0.5) -> trida 1 (oranzova).
_OUTPUT_CMAP = LinearSegmentedColormap.from_list(
    "vystup_site", ["#2a78d6", "#f0efec", "#eb6834"]
)
# Aktivace jedne jednotky (velikost): neutralni sedy odstin svetla -> tmava, aby
# se nepletl s barvami trid bodu nakreslenych pres nej.
_ACTIVATION_CMAP = LinearSegmentedColormap.from_list(
    "aktivace", ["#f7f6f3", "#c3c2b7", "#6f6e69", "#2c2c2a"]
)

# Jmena souradnic skrytych prostoru podle poradi vrstvy (io_[1] -> y, io_[2] -> u, ...).
_SPACE_NAMES: tuple[str, ...] = ("x", "y", "u", "v", "w")


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji.

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    """Sjednoti vzhled os: tlumene ramecky a popisky, zadna mrizka."""
    for spine in ax.spines.values():
        spine.set_color(_HAIRLINE)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(_INK_SECONDARY)
    ax.yaxis.label.set_color(_INK_SECONDARY)
    ax.title.set_color(_INK)


def _space_name(index: int) -> str:
    """Vrati jmeno souradnic prostoru ``io[index]`` (0 -> x, 1 -> y, 2 -> u, ...)."""
    return _SPACE_NAMES[index] if index < len(_SPACE_NAMES) else f"h{index}"


def _point_colors(labels: np.ndarray) -> list[str]:
    """Prevede stitky 0/1 na barvy bodu; jine hodnoty dostanou neutralni sedou."""
    return [_CLASS_COLORS.get(int(label), _OTHER_COLOR) for label in labels]


def _class_legend_handles(labels: np.ndarray, size: float) -> list[Line2D]:
    """Sestavi polozky legendy pro tridy, ktere se ve stitcich skutecne vyskytuji."""
    handles = []
    for cls in np.unique(labels):
        handles.append(
            Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(size) * 0.9,
                   markerfacecolor=_CLASS_COLORS.get(int(cls), _OTHER_COLOR),
                   markeredgecolor="white", label=f"trida {int(cls)}")
        )
    return handles


def _first_output(values: np.ndarray, n_rows: int) -> np.ndarray:
    """Vrati prvni vystupni jednotku site jako vektor ``(n_rows,)``."""
    return np.asarray(values, dtype=np.float64).reshape(n_rows, -1)[:, 0]


class NetworkVisualizer:
    """Drzi 2D data a opakovane kresli, jak rovinu deli navrzena sit.

    Vzor je stejny jako u ``DecisionBoundaryPlotter`` z Cviceni 08: data se
    predaji jednou v konstruktoru, ``set_network`` nastavi (nebo vymeni) sit
    a titulek, ``show`` / ``show_hidden_units`` kresli. Sit se vyhodnoti na
    jemne mrizce bodu pokryvajici data; vystup posledni vrstvy se interpretuje
    jako pravdepodobnost tridy 1 (prah 0.5).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, padding: float = 0.15,
                 resolution: int = 300) -> None:
        """Ulozi data pro opakovane vykreslovani.

        Parametry
        ---------
        x:
            Matice bodu tvaru ``(n, 2)`` -- presne dva priznaky ``x1``, ``x2``.
        y:
            Stitky tvaru ``(n,)`` (ground truth), podle nich se body obarvi.
        padding:
            Okraj kolem dat (v jednotkach souradnic), o ktery se zobrazena
            oblast rozsiri na kazdou stranu.
        resolution:
            Pocet bodu mrizky na jednu osu.
        """
        self.x: np.ndarray = np.asarray(x, dtype=np.float64)
        self.y: np.ndarray = np.asarray(y).ravel()
        self.padding: float = padding
        self.resolution: int = resolution
        self.network: "Sequential | None" = None
        self.title: str = ""

    def set_network(self, network: "Sequential", title: str = "") -> None:
        """Nastavi (nebo vymeni) sit a titulek pro nasledujici vykresleni.

        Parametry
        ---------
        network:
            Instance ``Sequential`` s 2 vstupy; posledni vrstva ma alespon
            jednu jednotku (pouzije se prvni).
        title:
            Titulek grafu; prazdny retezec ponecha vychozi titulek.
        """
        self.network = network
        self.title = title

    # ------------------------------------------------------------------ #
    def _require_network(self) -> "Sequential":
        """Vrati nastavenou sit, nebo vyhodi ``ValueError``."""
        if self.network is None:
            raise ValueError(
                "NetworkVisualizer: nejdrive zavolejte set_network(network, ...)."
            )
        return self.network

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sestavi mrizku ``(xx, yy, body)`` pokryvajici data s okrajem."""
        x_lo, x_hi = self.x[:, 0].min() - self.padding, self.x[:, 0].max() + self.padding
        y_lo, y_hi = self.x[:, 1].min() - self.padding, self.x[:, 1].max() + self.padding
        xx, yy = np.meshgrid(
            np.linspace(x_lo, x_hi, self.resolution),
            np.linspace(y_lo, y_hi, self.resolution),
        )
        return xx, yy, np.c_[xx.ravel(), yy.ravel()]

    def _scatter_data(self, ax: plt.Axes, size: float | None = None, alpha: float = 0.95) -> float:
        """Vykresli datove body obarvene podle stitku; vrati pouzitou velikost."""
        if size is None:
            size = 110.0 if self.x.shape[0] <= 20 else 18.0
        ax.scatter(self.x[:, 0], self.x[:, 1], s=size, c=_point_colors(self.y),
                   edgecolors="white", linewidths=0.8 if size > 50 else 0.4,
                   alpha=alpha, zorder=4)
        return size

    def _draw_first_layer_lines(self, ax: plt.Axes, xx: np.ndarray, yy: np.ndarray) -> None:
        """Zakresli primky jednotek prvni vrstvy a sipkou jejich kladnou stranu.

        Jednotka ``j`` prvni vrstvy ma primku ``w_1j x1 + w_2j x2 + b_j = 0``,
        kde ``(w_1j, w_2j)`` je ``j``-ty sloupec matice vah. Sipka ukazuje ve
        smeru normaly ``(w_1j, w_2j)``, tedy na stranu, kde je aktivace
        jednotky vetsi nez 0.5.
        """
        layer = self._require_network().layers[0]
        weights = np.asarray(layer.weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        if weights.shape[0] != 2:
            return
        n_units = weights.shape[1]
        bias = np.zeros(n_units)
        if layer.bias is not None:
            bias = bias + np.asarray(layer.bias, dtype=np.float64).reshape(-1)

        x_lo, x_hi, y_lo, y_hi = xx.min(), xx.max(), yy.min(), yy.max()
        span = max(x_hi - x_lo, y_hi - y_lo)
        center = np.array([(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0])

        for j in range(n_units):
            normal = weights[:, j]
            norm = float(np.hypot(*normal))
            if norm == 0.0:
                continue
            unit_normal = normal / norm
            direction = np.array([-unit_normal[1], unit_normal[0]])
            # Bod primky nejblize stredu obrazku.
            p0 = center - (normal @ center + bias[j]) / norm**2 * normal
            t = np.array([-2.0 * span, 2.0 * span])
            ax.plot(p0[0] + t * direction[0], p0[1] + t * direction[1],
                    color=_INK_SECONDARY, linestyle=(0, (5, 3)), linewidth=1.3, zorder=3)

            # Popisek a sipka: kazda jednotka jinde podel sve primky, aby se
            # popisky neprekryvaly ani u primek protinajicich se ve stredu.
            offset = (0.30 if j % 2 == 0 else -0.30) * span * (1.0 - 0.25 * (j // 2))
            anchor = p0 + offset * direction
            if not (x_lo <= anchor[0] <= x_hi and y_lo <= anchor[1] <= y_hi):
                anchor = p0
            if not (x_lo <= anchor[0] <= x_hi and y_lo <= anchor[1] <= y_hi):
                continue
            ax.add_patch(FancyArrowPatch(
                tuple(anchor), tuple(anchor + 0.08 * span * unit_normal),
                arrowstyle="-|>", mutation_scale=11, color=_INK_SECONDARY,
                linewidth=1.2, zorder=5,
            ))
            label_pos = anchor - 0.045 * span * unit_normal
            ax.text(label_pos[0], label_pos[1], f"y{j + 1}", color=_INK, fontsize=10,
                    ha="center", va="center", zorder=6,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": _SURFACE,
                          "edgecolor": "none", "alpha": 0.85})

    # ------------------------------------------------------------------ #
    def show(self, save_path: str | None = None, mode: str = "soft",
             draw_hidden_lines: bool = True) -> None:
        """Vykresli rozhodovaci oblast site v rovine ``x1-x2``.

        Parametry
        ---------
        save_path:
            Cesta k vystupnimu PNG, nebo ``None`` (figura se jen zavre).
        mode:
            ``"soft"`` -- spojity odstin podle vystupu site (0 modra, 0.5 seda,
            1 oranzova); ``"hard"`` -- dvoubarevne podbarveni podle predikce
            ``vystup >= 0.5``. V obou rezimech je plnou carou vyznacena
            hranice ``vystup = 0.5``.
        draw_hidden_lines:
            Ma-li sit alespon dve vrstvy, zakresli carkovane primky jednotek
            prvni skryte vrstvy (s popisky ``y1, y2, ...`` a sipkou ke kladne
            strane).

        Vyjimky
        -------
        ``ValueError``:
            Pokud neni nastavena sit, nebo ``mode`` neni ``"soft"``/``"hard"``.
        """
        network = self._require_network()
        if mode not in ("soft", "hard"):
            raise ValueError(f"Neznamy mode: {mode!r}. Povolene hodnoty jsou: 'soft', 'hard'.")

        xx, yy, grid = self._grid()
        output = _first_output(network(grid), grid.shape[0]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(6.4, 5.8))
        fig.patch.set_facecolor(_SURFACE)
        ax.set_facecolor(_SURFACE)

        if mode == "hard":
            ax.contourf(xx, yy, (output >= 0.5).astype(np.float64),
                        levels=[-0.5, 0.5, 1.5], colors=list(_CLASS_TINTS))
        else:
            filled = ax.contourf(xx, yy, output, levels=np.linspace(0.0, 1.0, 21),
                                 cmap=_OUTPUT_CMAP, alpha=0.75, extend="both")
            cbar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("vystup site", color=_INK_SECONDARY)
            cbar.ax.tick_params(colors=_INK_SECONDARY, labelsize=8)
            cbar.outline.set_edgecolor(_HAIRLINE)
        ax.contour(xx, yy, output, levels=[0.5], colors=_INK, linewidths=1.8, zorder=3)

        handles: list[Line2D] = [Line2D([], [], color=_INK, linewidth=1.8,
                                        label="hranice site (vystup = 0.5)")]
        if draw_hidden_lines and len(network.layers) >= 2:
            self._draw_first_layer_lines(ax, xx, yy)
            handles.append(Line2D([], [], color=_INK_SECONDARY, linestyle=(0, (5, 3)),
                                  linewidth=1.3, label="primky 1. skryte vrstvy"))

        size = self._scatter_data(ax)
        handles = _class_legend_handles(self.y, size) + handles

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_aspect("equal")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(self.title if self.title else "Rozhodovaci oblast site", fontsize=12)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                  ncol=2, frameon=False, fontsize=9, labelcolor=_INK_SECONDARY)
        _style_axes(ax)
        _save_and_close(fig, save_path)

    def show_hidden_units(self, save_path: str | None = None, layer_index: int = 1) -> None:
        """Vykresli aktivaci kazde jednotky zvolene vrstvy nad rovinou ``x1-x2``.

        Sit se vyhodnoti na mrizce a z jeji stopy ``network.io_[layer_index]``
        (tvar ``(n_bodu_mrizky, n_jednotek)``) se pro kazdou jednotku vykresli
        samostatny panel. Kazdy panel je tak "jedna souradnice noveho prostoru"
        jako funkce puvodni polohy bodu.

        Parametry
        ---------
        save_path:
            Cesta k vystupnimu PNG, nebo ``None`` (figura se jen zavre).
        layer_index:
            Index do ``io_``: ``1`` = prvni skryta vrstva, ``2`` = druha, ...

        Vyjimky
        -------
        ``ValueError``:
            Pokud neni nastavena sit, sit po pruchodu nema stopu ``io_``,
            nebo ``layer_index`` je mimo rozsah ``1 .. len(io_) - 1``.
        """
        network = self._require_network()
        xx, yy, grid = self._grid()
        network(grid)
        io = network.io_
        if io is None or not 1 <= layer_index < len(io):
            raise ValueError(
                f"layer_index={layer_index} je mimo rozsah stopy io_ "
                f"(delka {None if io is None else len(io)})."
            )

        hidden = np.asarray(io[layer_index], dtype=np.float64).reshape(grid.shape[0], -1)
        n_units = hidden.shape[1]
        name = _space_name(layer_index)
        levels = np.linspace(min(0.0, hidden.min()), max(1.0, hidden.max()), 21)

        fig, axes = plt.subplots(1, n_units, figsize=(4.3 * n_units + 0.8, 4.2), squeeze=False)
        fig.patch.set_facecolor(_SURFACE)
        fig.subplots_adjust(wspace=0.35)
        filled = None
        for j, ax in enumerate(axes[0]):
            filled = ax.contourf(xx, yy, hidden[:, j].reshape(xx.shape), levels=levels,
                                 cmap=_ACTIVATION_CMAP)
            self._scatter_data(ax, size=70.0 if self.x.shape[0] <= 20 else 9.0, alpha=1.0)
            ax.set_aspect("equal")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title(f"souradnice {name}{j + 1}", fontsize=11)
            _style_axes(ax)

        cbar = fig.colorbar(filled, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("aktivace jednotky", color=_INK_SECONDARY)
        cbar.ax.tick_params(colors=_INK_SECONDARY, labelsize=8)
        cbar.outline.set_edgecolor(_HAIRLINE)
        fig.suptitle(
            self.title if self.title else f"Jednotky vrstvy io_[{layer_index}] nad rovinou x1-x2",
            color=_INK, fontsize=12,
        )
        _save_and_close(fig, save_path)


def plot_space_transformation(io: list[np.ndarray], labels: np.ndarray,
                              save_path: str | None = None) -> None:
    """Vykresli tytez body pred vrstvou (``io[0]``) a po ni (``io[1]``).

    Levy panel ukazuje body v prostoru ``io[0]`` (typicky vstup ``x1-x2``),
    pravy panel tytez body v prostoru ``io[1]`` (typicky prvni skryta vrstva
    ``y1-y2``). Barva bodu je jejich stitek (vysledna trida), takze je primo
    videt, zda se tridy v novem prostoru daji oddelit primkou (rovinou).

    Chcete-li zobrazit prechod mezi dalsimi vrstvami, predejte posunutou
    stopu, napr. ``plot_space_transformation(net.io_[1:], labels)`` ukaze
    prechod z prvni do druhe skryte vrstvy.

    Pro nejvyse 8 bodu (rohy hradla) se kazdy bod v obou panelech popise
    svymi souradnicemi v ``io[0]`` -- lze tak sledovat, kam se ktery roh
    zobrazil, i kdyz se vice rohu slije do jednoho bodu.

    Parametry
    ---------
    io:
        Stopa pruchodu ``[io_0, io_1, ...]`` (napr. ``Sequential.io_``);
        pouziji se prvni dva prvky. ``io[0]`` musi mit 2 sloupce, ``io[1]``
        2 nebo 3 sloupce (3 sloupce se vykresli ve 3D).
    labels:
        Stitky tvaru ``(n,)``, podle kterych se body obarvi.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None`` (figura se jen zavre).

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``io`` ma mene nez dva prvky nebo tvary neodpovidaji popisu.
    """
    if len(io) < 2:
        raise ValueError("io musi obsahovat alespon dva prvky (pred a po vrstve).")
    before = np.asarray(io[0], dtype=np.float64)
    after = np.asarray(io[1], dtype=np.float64)
    labels = np.asarray(labels).ravel()
    n_points = before.shape[0]
    if before.ndim != 2 or before.shape[1] != 2:
        raise ValueError(f"io[0] musi mit tvar (n, 2), zadano: {before.shape}")
    if after.ndim == 1:
        after = after.reshape(-1, 1)
    if after.shape[0] != n_points or after.shape[1] not in (2, 3):
        raise ValueError(f"io[1] musi mit tvar (n, 2) nebo (n, 3), zadano: {after.shape}")
    if labels.shape[0] != n_points:
        raise ValueError(f"labels musi mit {n_points} prvku, zadano: {labels.shape[0]}")

    colors = _point_colors(labels)
    size = 120.0 if n_points <= 8 else 16.0
    annotate = n_points <= 8
    in_unit_box = bool(np.all(after >= -0.01) and np.all(after <= 1.01))

    fig = plt.figure(figsize=(12.0, 5.4))
    fig.patch.set_facecolor(_SURFACE)
    ax_before = fig.add_subplot(1, 2, 1)
    is_3d = after.shape[1] == 3
    ax_after = fig.add_subplot(1, 2, 2, projection="3d" if is_3d else None)
    fig.subplots_adjust(wspace=0.5)

    # --- Levy panel: pred vrstvou ------------------------------------------
    ax_before.set_facecolor(_SURFACE)
    ax_before.scatter(before[:, 0], before[:, 1], s=size, c=colors, edgecolors="white",
                      linewidths=0.8 if annotate else 0.3, zorder=3)
    pad = 0.12 * max(np.ptp(before[:, 0]), np.ptp(before[:, 1]), 1e-9)
    ax_before.set_xlim(before[:, 0].min() - pad, before[:, 0].max() + pad)
    ax_before.set_ylim(before[:, 1].min() - pad, before[:, 1].max() + pad)
    ax_before.set_aspect("equal")
    ax_before.set_xlabel("slozka 1")
    ax_before.set_ylabel("slozka 2")
    ax_before.set_title("Pred vrstvou  (prvek io[0])", fontsize=12)
    _style_axes(ax_before)

    # --- Pravy panel: po vrstve --------------------------------------------
    point_names = [f"({before[i, 0]:g}, {before[i, 1]:g})" for i in range(n_points)]
    if is_3d:
        ax_after.scatter(after[:, 0], after[:, 1], after[:, 2], s=size, c=colors,
                         edgecolors="white", linewidths=0.3, depthshade=False)
        if in_unit_box:
            for a in (0.0, 1.0):
                for b in (0.0, 1.0):
                    ax_after.plot([0, 1], [a, a], [b, b], color=_HAIRLINE, linewidth=0.7)
                    ax_after.plot([a, a], [0, 1], [b, b], color=_HAIRLINE, linewidth=0.7)
                    ax_after.plot([a, a], [b, b], [0, 1], color=_HAIRLINE, linewidth=0.7)
            ax_after.set_xlim(0, 1)
            ax_after.set_ylim(0, 1)
            ax_after.set_zlim(0, 1)
        ax_after.set_zlabel("slozka 3", color=_INK_SECONDARY)
        ax_after.view_init(elev=22, azim=-58)
        ax_after.set_xlabel("slozka 1", color=_INK_SECONDARY)
        ax_after.set_ylabel("slozka 2", color=_INK_SECONDARY)
        ax_after.tick_params(colors=_INK_SECONDARY, labelsize=8)
        if annotate:
            seen: dict[tuple[float, ...], int] = {}
            for i in range(n_points):
                key = tuple(np.round(after[i], 2))
                k = seen.get(key, 0)
                seen[key] = k + 1
                ax_after.text(after[i, 0], after[i, 1], after[i, 2] + 0.07 + 0.08 * k,
                              point_names[i], fontsize=8, color=_INK_SECONDARY)
    else:
        ax_after.set_facecolor(_SURFACE)
        if in_unit_box:
            ax_after.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=_HAIRLINE,
                          linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
            ax_after.set_xlim(-0.12, 1.12)
            ax_after.set_ylim(-0.12, 1.12)
        else:
            pad_after = 0.12 * max(np.ptp(after[:, 0]), np.ptp(after[:, 1]), 1e-9)
            ax_after.set_xlim(after[:, 0].min() - pad_after, after[:, 0].max() + pad_after)
            ax_after.set_ylim(after[:, 1].min() - pad_after, after[:, 1].max() + pad_after)
        ax_after.scatter(after[:, 0], after[:, 1], s=size, c=colors, edgecolors="white",
                         linewidths=0.8 if annotate else 0.3, zorder=3)
        ax_after.set_aspect("equal")
        ax_after.set_xlabel("slozka 1")
        ax_after.set_ylabel("slozka 2")
        _style_axes(ax_after)
        if annotate:
            seen_2d: dict[tuple[float, ...], int] = {}
            for i in range(n_points):
                key = tuple(np.round(after[i], 2))
                k = seen_2d.get(key, 0)
                seen_2d[key] = k + 1
                ax_after.annotate(point_names[i], (after[i, 0], after[i, 1]),
                                  xytext=(9, 7 - 13 * k), textcoords="offset points",
                                  fontsize=8, color=_INK_SECONDARY, zorder=4)
    ax_after.set_title("Po vrstve  (prvek io[1])", fontsize=12, color=_INK)

    if annotate:
        for i in range(n_points):
            ax_before.annotate(point_names[i], (before[i, 0], before[i, 1]),
                               xytext=(9, 7), textcoords="offset points",
                               fontsize=8, color=_INK_SECONDARY, zorder=4)

    # --- Sipka "vrstva" mezi panely a legenda --------------------------------
    fig.add_artist(FancyArrowPatch((0.465, 0.52), (0.54, 0.52), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=20, linewidth=1.6,
                                   color=_INK_SECONDARY))
    fig.text(0.5025, 0.56, "vrstva", ha="center", va="bottom", fontsize=10, color=_INK_SECONDARY)
    fig.legend(handles=_class_legend_handles(labels, size), loc="lower center", ncol=2,
               frameon=False, fontsize=10, labelcolor=_INK_SECONDARY,
               bbox_to_anchor=(0.5, -0.02))
    _save_and_close(fig, save_path)
