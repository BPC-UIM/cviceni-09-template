# -*- coding: utf-8 -*-

"""
Created on 15. 09. 2026 at 10:40:00

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Vstupni bod cviceni 09 (vicevrstva sit bez uceni -- rucni navrh vah).
    Pipeline projde pet fazi:

      1. hradlo AND -- jednovrstva sit, VZOROVE RESENI (predvyplneno),
      2. hradlo OR  -- jednovrstva sit, VZOROVE RESENI (predvyplneno),
      3. hradlo XOR -- dvouvrstva sit 2-2-1; skryta vrstva zobrazi rohy
         ctverce do prostoru y1-y2, kde uz XOR linearne separovatelny je
         (UKOL: matice vah a vektory biasu),
      4. trojuhelnik -- sit 2-3-1: tri primky (strany) + vystupni AND
         (UKOL: matice vah a vektory biasu),
      5. motylek -- sit 2-2-2-1: dve poloroviny + XOR nad nimi, tedy dve
         skryte vrstvy (UKOL: matice vah a vektory biasu).

    Kostra kazde faze (sestaveni site, dopredny pruchod, vyhodnoceni,
    vykresleni) je predvyplnena; student ve fazich 3-5 nahradi zastupne
    hodnoty None vlastnimi maticemi vah a vektory biasu.

    Konvence tvaru: vrstva s n_vstupu vstupy a n_jednotek jednotkami ma
    matici vah (n_vstupu, n_jednotek) -- sloupec j jsou vahy j-teho neuronu --
    a vektor biasu (n_jednotek,). Vsechny vrstvy pouzivaji Sigmoid s teplotou
    z config.yaml; vystup site >= 0.5 znamena tridu 1.

    Repozitar bezi v kazdem stavu. Dokud nejsou ukoly hotove (brana src/
    z Cviceni 08, Sequential.forward, navrhy vah), faze se zastavi jen
    hlaskou [NENI HOTOVO] Ukol: ... a pipeline pokracuje dal -- nikdy
    nezpracovanym tracebackem. Spatny tvar navrzene matice ohlasi faze
    hlaskou [CHYBA NAVRHU].
================================================================================
"""

from __future__ import annotations

import sys

import numpy as np

# --- Import guard: srozumitelna hlaska misto holeho ImportError ----------------
try:
    from dataio import (
        ExperimentConfig,
        NetworkVisualizer,
        generate_points,
        inside_butterfly,
        inside_triangle,
        load_config,
        make_gate,
        plot_space_transformation,
    )
    from src import Linear, Neuron, Sequential, Sigmoid
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)

GRAPHS_DIR = "graphs"          # vystupni grafy (.png)

# Jmena souradnic skrytych prostoru v poradi vrstev (io_[1] -> y, io_[2] -> u).
HIDDEN_SPACE_NAMES = ("y", "u", "v")

# Pod touto presnosti na plosnych ulohach vypise pipeline tip k meritku vah.
PRAH_TIPU = 0.98


# ============================================================================ #
#  Pomocne funkce (predvyplneno)                                               #
# ============================================================================ #
def _banner(text: str) -> None:
    """Vypise oddelovaci nadpis faze pipeline."""
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _faze_neni_hotova(exc: NotImplementedError) -> None:
    """Vypise pratelskou hlasku, kdyz faze narazi na nedokonceny ukol."""
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncite v ramci ukolu; pipeline pokracuje dal.")


def _chyba_navrhu(exc: ValueError) -> None:
    """Vypise pratelskou hlasku, kdyz navrzena matice nema spravny tvar."""
    print(f"  [CHYBA NAVRHU] {exc}")
    print("  -> Opravte tvar matice vah/vektoru biasu; pipeline pokracuje dal.")


def _vrstva(weights: np.ndarray, bias: np.ndarray, temperature: float) -> Neuron:
    """Sestavi jednu vrstvu site: ``Linear`` s matici vah a ``Sigmoid``.

    Parametry
    ---------
    weights : np.ndarray
        Matice vah tvaru ``(n_vstupu, n_jednotek)``; sloupec ``j`` jsou vahy
        ``j``-teho neuronu vrstvy.
    bias : np.ndarray
        Vektor biasu tvaru ``(n_jednotek,)``.
    temperature : float
        Teplota sigmoidy (``cfg.sigmoid.temperature``).

    Navratova hodnota
    -----------------
    Neuron
        Vrstva jako slozeni ``Linear`` a injektovane aktivace ``Sigmoid``.
    """
    linear = Linear(np.asarray(weights, dtype=np.float64), np.asarray(bias, dtype=np.float64))
    return Neuron(linear, Sigmoid(temperature=temperature))


def _over_navrh(faze: str, navrh: dict[str, tuple[np.ndarray | None, tuple[int, ...]]]) -> None:
    """Zkontroluje, ze student vyplnil vsechny matice a ze maji ocekavany tvar.

    Parametry
    ---------
    faze : str
        Jmeno faze pro chybovou hlasku.
    navrh : dict[str, tuple[np.ndarray | None, tuple[int, ...]]]
        ``jmeno_promenne -> (hodnota, ocekavany_tvar)``.

    Vyjimky
    -------
    NotImplementedError
        Pokud je nektera hodnota jeste ``None`` (zprava zacina ``Úkol:``).
    ValueError
        Pokud vyplnena hodnota nema ocekavany tvar.
    """
    chybi = [jmeno for jmeno, (hodnota, _) in navrh.items() if hodnota is None]
    if chybi:
        raise NotImplementedError(
            f"Úkol: navrhnete vahy a biasy site ve funkci {faze}() -- nahradte None "
            f"u promennych: {', '.join(chybi)}."
        )
    for jmeno, (hodnota, tvar) in navrh.items():
        if np.shape(hodnota) != tvar:
            raise ValueError(
                f"{faze}(): {jmeno} ma tvar {np.shape(hodnota)}, ocekavan {tvar}."
            )


def _predikce(vystup: np.ndarray) -> np.ndarray:
    """Prevede vystup site ``(n, 1)`` na tvrde tridy 0/1 prahem 0.5."""
    return (np.asarray(vystup, dtype=np.float64).reshape(len(vystup), -1)[:, 0] >= 0.5).astype(int)


def _vypis_pruchod(io: list[np.ndarray], labels: np.ndarray) -> None:
    """Vypise pro kazdy vzorek vstup, souradnice ve skrytych vrstvach, vystup a cil."""
    for i in range(io[0].shape[0]):
        casti = [f"x=({io[0][i, 0]:g}, {io[0][i, 1]:g})"]
        for k, hidden in enumerate(io[1:-1]):
            jmeno = HIDDEN_SPACE_NAMES[k] if k < len(HIDDEN_SPACE_NAMES) else f"h{k + 1}"
            casti.append(f"{jmeno}=(" + ", ".join(f"{v:.3f}" for v in hidden[i]) + ")")
        vystup = float(np.asarray(io[-1])[i].ravel()[0])
        casti.append(f"vystup={vystup:.3f} -> {int(vystup >= 0.5)}")
        casti.append(f"cil={int(labels[i])}")
        print("    " + " | ".join(casti))


def _vyhodnot(
    sit: Sequential,
    x: np.ndarray,
    labels: np.ndarray,
    vypsat_body: bool,
    ) -> list[np.ndarray]:
    """Provede dopredny pruchod, vypise presnost a vrati kopii stopy ``io_``.

    Kopie stopy je nutna: nasledne vykreslovani vola sit na mrizce bodu
    a ``sit.io_`` tim prepise.
    """
    vystup = sit(x)
    io = list(sit.io_)
    if vypsat_body:
        _vypis_pruchod(io, labels)
    tvary = " -> ".join(str(tuple(np.shape(a))) for a in io)
    presnost = float(np.mean(_predikce(vystup) == labels))
    print(f"  tvary stopy io_: {tvary}")
    print(f"  presnost vuci cili: {presnost:.3f}")
    if presnost < PRAH_TIPU and x.shape[0] > 4:
        print("  [TIP] Lezi-li chybne body jen tesne u hranic, primky jsou nejspis spravne, ale")
        print("        prechod sigmoidy je vuci velikosti oblasti prilis pozvolny. Vynasobte")
        print("        vsechny matice vah i biasy stejnou kladnou konstantou k (napr. k = 4):")
        print("        primky se nezmeni a prechod se zostri -- totez jako teplota T/k")
        print("        (README, Teoreticky zaklad, odd. 6).")
    return io


def _vzorova_jednovrstva_faze(gate: str, weights: np.ndarray, bias: np.ndarray,
                              cfg: ExperimentConfig) -> None:
    """Spolecna kostra vzorovych fazi AND/OR: jedna vrstva s jednim neuronem."""
    x, labels = make_gate(gate)
    sit = Sequential([_vrstva(weights, bias, cfg.sigmoid.temperature)])
    print(f"  weights = {weights.tolist()}  (tvar {weights.shape}),  "
          f"bias = {bias.tolist()}  (tvar {bias.shape})")
    try:
        _vyhodnot(sit, x, labels, vypsat_body=True)
        vizualizace = NetworkVisualizer(x, labels, padding=0.35)
        vizualizace.set_network(sit, title=f"Hradlo {gate.upper()}: jednovrstva sit")
        save_path = f"{GRAPHS_DIR}/{gate}_hranice.png"
        vizualizace.show(save_path=save_path)
        print(f"  graf ulozen: {save_path}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


# ============================================================================ #
#  Faze 1-2: vzorova reseni (predvyplneno)                                     #
# ============================================================================ #
def faze_and(cfg: ExperimentConfig) -> None:
    """Faze 1 -- hradlo AND jako jednovrstva sit (VZOROVE RESENI).

    Jedna vrstva s jednim neuronem: primka ``x1 + x2 - 1.5 = 0``. I jediny
    neuron se zde zapisuje v maticove konvenci vrstvy -- vahy tvaru
    ``(2, 1)`` (2 vstupy, 1 jednotka) a bias tvaru ``(1,)`` -- aby byl zapis
    shodny s vicevrstvymi sitemi ve fazich 3-5.
    """
    _banner("Faze 1: Hradlo AND -- jednovrstva sit (vzorove reseni)")
    w_output = np.array([[1.0],
                         [1.0]])       # sloupec = vahy jedineho neuronu (w1, w2)
    b_output = np.array([-1.5])        # bias = zaporne vzaty prah 1.5
    _vzorova_jednovrstva_faze("and", w_output, b_output, cfg)


def faze_or(cfg: ExperimentConfig) -> None:
    """Faze 2 -- hradlo OR jako jednovrstva sit (VZOROVE RESENI).

    Stejna normala jako u AND, jen posunuty prah: primka ``x1 + x2 - 0.5 = 0``.
    """
    _banner("Faze 2: Hradlo OR -- jednovrstva sit (vzorove reseni)")
    w_output = np.array([[1.0],
                         [1.0]])
    b_output = np.array([-0.5])
    _vzorova_jednovrstva_faze("or", w_output, b_output, cfg)


# ============================================================================ #
#  Faze 3-5: navrh vah (UKOL -- nahradte None)                                 #
# ============================================================================ #
def faze_xor(cfg: ExperimentConfig) -> None:
    """Faze 3 -- hradlo XOR jako dvouvrstva sit 2-2-1.

    Skryta vrstva: dva neurony = dve primky v rovine ``x1-x2``; jejich vystupy
    jsou souradnice ``y1, y2`` noveho prostoru. Vystupni vrstva: jeden neuron
    = jedna primka v rovine ``y1-y2``. Navrhnete primky tak, aby se ctyri
    rohy zobrazily do prostoru ``y1-y2``, kde je oddeli jedina primka
    (viz README, Teoreticky zaklad, odd. 3, a priklady_09.md).
    """
    _banner("Faze 3: Hradlo XOR -- dvouvrstva sit 2-2-1 (UKOL: navrh vah)")
    x, labels = make_gate("xor")

    # ------------------------- UKOL: navrh vah a biasu ------------------------
    # Skryta vrstva (2 vstupy -> 2 jednotky y1, y2): sloupec j = vahy neuronu y_j.
    w_hidden: np.ndarray | None = None      # np.array tvaru (2, 2)
    b_hidden: np.ndarray | None = None      # np.array tvaru (2,)
    # Vystupni vrstva (2 vstupy y1, y2 -> 1 jednotka).
    w_output: np.ndarray | None = None      # np.array tvaru (2, 1)
    b_output: np.ndarray | None = None      # np.array tvaru (1,)
    # --------------------------------------------------------------------------

    try:
        _over_navrh("faze_xor", {
            "w_hidden": (w_hidden, (2, 2)), "b_hidden": (b_hidden, (2,)),
            "w_output": (w_output, (2, 1)), "b_output": (b_output, (1,)),
        })
        t = cfg.sigmoid.temperature
        sit = Sequential([_vrstva(w_hidden, b_hidden, t), _vrstva(w_output, b_output, t)])
        io = _vyhodnot(sit, x, labels, vypsat_body=True)

        vizualizace = NetworkVisualizer(x, labels, padding=0.35)
        vizualizace.set_network(sit, title="Hradlo XOR: sit 2-2-1")
        vizualizace.show(save_path=f"{GRAPHS_DIR}/xor_hranice.png")
        vizualizace.show_hidden_units(save_path=f"{GRAPHS_DIR}/xor_skryte_jednotky.png")
        plot_space_transformation(io, labels, save_path=f"{GRAPHS_DIR}/xor_transformace.png")
        print(f"  grafy ulozeny: {GRAPHS_DIR}/xor_hranice.png, xor_skryte_jednotky.png, "
              f"xor_transformace.png")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
    except ValueError as exc:
        _chyba_navrhu(exc)


def faze_trojuhelnik(cfg: ExperimentConfig) -> None:
    """Faze 4 -- trojuhelnik jako sit 2-3-1.

    Kazda strana trojuhelniku je primka = jeden neuron skryte vrstvy, jehoz
    aktivace je blizko 1 na te strane primky, kde lezi trojuhelnik.
    Vystupni neuron realizuje AND tri vstupu: "uvnitr" = uvnitr vsech tri
    polorovin. Skryty prostor ``y1-y2-y3`` je trojrozmerny.

    Navrh odpovida vrcholum ``triangle.vertices`` v config.yaml; zmenite-li
    vrcholy, je nutne navrh prepocitat.
    """
    _banner("Faze 4: Trojuhelnik -- sit 2-3-1 (UKOL: navrh vah)")
    x = generate_points(cfg.data.n_samples, seed=cfg.data.seed)
    labels = inside_triangle(x, np.array(cfg.triangle.vertices))
    print(f"  vrcholy: {cfg.triangle.vertices};  bodu: {x.shape[0]}, "
          f"z toho uvnitr: {int(labels.sum())}")

    # ------------------------- UKOL: navrh vah a biasu ------------------------
    # Skryta vrstva (2 vstupy -> 3 jednotky, jedna na kazdou stranu).
    w_hidden: np.ndarray | None = None      # np.array tvaru (2, 3)
    b_hidden: np.ndarray | None = None      # np.array tvaru (3,)
    # Vystupni vrstva (3 vstupy y1, y2, y3 -> 1 jednotka): AND tri polorovin.
    w_output: np.ndarray | None = None      # np.array tvaru (3, 1)
    b_output: np.ndarray | None = None      # np.array tvaru (1,)
    # --------------------------------------------------------------------------

    try:
        _over_navrh("faze_trojuhelnik", {
            "w_hidden": (w_hidden, (2, 3)), "b_hidden": (b_hidden, (3,)),
            "w_output": (w_output, (3, 1)), "b_output": (b_output, (1,)),
        })
        t = cfg.sigmoid.temperature
        sit = Sequential([_vrstva(w_hidden, b_hidden, t), _vrstva(w_output, b_output, t)])
        io = _vyhodnot(sit, x, labels, vypsat_body=False)

        vizualizace = NetworkVisualizer(x, labels, padding=0.05)
        vizualizace.set_network(sit, title="Trojuhelnik: sit 2-3-1")
        vizualizace.show(save_path=f"{GRAPHS_DIR}/trojuhelnik_hranice.png")
        plot_space_transformation(io, labels,
                                  save_path=f"{GRAPHS_DIR}/trojuhelnik_transformace.png")
        print(f"  grafy ulozeny: {GRAPHS_DIR}/trojuhelnik_hranice.png, "
              f"trojuhelnik_transformace.png")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
    except ValueError as exc:
        _chyba_navrhu(exc)


def faze_motylek(cfg: ExperimentConfig) -> None:
    """Faze 5 -- motylek ``(x2 > 0.5) != (x2 > x1)`` jako sit 2-2-2-1.

    Motylek je XOR dvou polorovin. Prvni skryta vrstva zobrazi rovinu do
    prostoru ``y1-y2`` (y1 ~ "x2 > 0.5", y2 ~ "x2 > x1") -- body se tim sliji
    k rohum ctverce presne s rozlozenim trid hradla XOR. Druha skryta vrstva
    a vystup jsou pak sit XOR z faze 3 (prostor ``u1-u2``). Sit tak vznika
    **skladanim**: XOR sit nasazena na vrstvu polorovin.
    """
    _banner("Faze 5: Motylek -- sit 2-2-2-1 (UKOL: navrh vah)")
    x = generate_points(cfg.data.n_samples, seed=cfg.data.seed)
    labels = inside_butterfly(x)
    print(f"  bodu: {x.shape[0]}, z toho v motylku: {int(labels.sum())}")

    # ------------------------- UKOL: navrh vah a biasu ------------------------
    # 1. skryta vrstva (2 vstupy -> 2 jednotky y1, y2): dve poloroviny.
    w_hidden_1: np.ndarray | None = None    # np.array tvaru (2, 2)
    b_hidden_1: np.ndarray | None = None    # np.array tvaru (2,)
    # 2. skryta vrstva (2 vstupy y1, y2 -> 2 jednotky u1, u2).
    w_hidden_2: np.ndarray | None = None    # np.array tvaru (2, 2)
    b_hidden_2: np.ndarray | None = None    # np.array tvaru (2,)
    # Vystupni vrstva (2 vstupy u1, u2 -> 1 jednotka).
    w_output: np.ndarray | None = None      # np.array tvaru (2, 1)
    b_output: np.ndarray | None = None      # np.array tvaru (1,)
    # --------------------------------------------------------------------------

    try:
        _over_navrh("faze_motylek", {
            "w_hidden_1": (w_hidden_1, (2, 2)), "b_hidden_1": (b_hidden_1, (2,)),
            "w_hidden_2": (w_hidden_2, (2, 2)), "b_hidden_2": (b_hidden_2, (2,)),
            "w_output": (w_output, (2, 1)), "b_output": (b_output, (1,)),
        })
        t = cfg.sigmoid.temperature
        sit = Sequential([
            _vrstva(w_hidden_1, b_hidden_1, t),
            _vrstva(w_hidden_2, b_hidden_2, t),
            _vrstva(w_output, b_output, t),
        ])
        io = _vyhodnot(sit, x, labels, vypsat_body=False)

        vizualizace = NetworkVisualizer(x, labels, padding=0.05)
        vizualizace.set_network(sit, title="Motylek: sit 2-2-2-1")
        vizualizace.show(save_path=f"{GRAPHS_DIR}/motylek_hranice.png")
        # Prechod x -> y (poloroviny) a y -> u (XOR nad polorovinami).
        plot_space_transformation(io, labels,
                                  save_path=f"{GRAPHS_DIR}/motylek_transformace_1.png")
        plot_space_transformation(io[1:], labels,
                                  save_path=f"{GRAPHS_DIR}/motylek_transformace_2.png")
        print(f"  grafy ulozeny: {GRAPHS_DIR}/motylek_hranice.png, "
              f"motylek_transformace_1.png, motylek_transformace_2.png")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
    except ValueError as exc:
        _chyba_navrhu(exc)


def main() -> None:
    """Spusti celou pipeline cviceni 09 s ochrannymi bloky u kazde faze."""
    _banner("CVICENI 09 -- Vicevrstva sit: skladani vrstev a transformace prostoru -- start")

    # --- Config guard -----------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    faze_and(cfg)
    faze_or(cfg)
    faze_xor(cfg)
    faze_trojuhelnik(cfg)
    faze_motylek(cfg)

    _banner("CVICENI 09 -- konec")


if __name__ == "__main__":
    main()
