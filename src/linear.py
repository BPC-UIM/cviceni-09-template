"""Afinni transformace (vazeny soucet vstupu plus volitelny bias) bez aktivace.

BRANA (continuity gate): Zkopirujte sem sve reseni `src/linear.py` z Cviceni 08.
Rozhrani (signatury) je s Cvicenim 08 totozne, staci doplnit tela metod.

`Linear` pocita `z = x @ weights + bias` a nic vic - je to samostatna, na
aktivaci nezavisla vrstva (stejny motiv jako `nn.Linear` v PyTorch, ktery se
sklada s `nn.ReLU()` apod. az zvenku). `Neuron` (modul `src.neuron`) `Linear`
sklada s injektovanou `Activation`.

V Cviceni 09 se `Linear` pouziva s **matici vah** tvaru
`(n_priznaku, n_jednotek)` a s **vektorem biasu** tvaru `(n_jednotek,)` -
jedna instance tak reprezentuje celou vrstvu. Kod z Cviceni 08 se nemeni:
`x @ weights` da `(n_vzorku, n_jednotek)` a vektor biasu se k vysledku pricte
broadcastingem po radcich. Sloupec `j` matice `weights` jsou vahy `j`-teho
neuronu vrstvy (normala jeho rozhodovaci primky).
"""

from __future__ import annotations

import numpy as np


class Linear:
    """Afinni vrstva: `z = x @ weights + bias`, bez aktivace.

    Bias je samostatny, volitelny, PyTorch-style parametr (NE stara "-1
    augmentace" vstupniho vektoru): `bias=None` znamena bez biasu (jako
    PyTorch `bias=False`), float jej zapina. V aditivnim tvaru
    `z = x @ weights + bias` je bias roven zaporne vzatemu prahu - napriklad
    AND s prahem 1.5 odpovida `bias = -1.5`, OR s prahem 0.5 odpovida
    `bias = -0.5`.
    """

    def __init__(self, weights: np.ndarray, bias: float | None = None) -> None:
        """Ulozi vahy a volitelny bias.

        Parametry
        ---------
        weights : np.ndarray
            Vektor vah tvaru `(n_priznaku,)` (jeden neuron), nebo obecneji
            matice tvaru `(n_priznaku, n_jednotek)` pro vice jednotek najednou.
        bias : float | None, optional
            Zaporne vzaty prah v aditivnim tvaru `z = x @ weights + bias`.
            `None` (vychozi) znamena bez biasu, konkretni `float` jej zapina.
        """
        self.weights = weights
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provede dopredny pruchod afinni vrstvou pres davku vstupu.

        Definice
        --------
        z = x @ self.weights + (self.bias, pokud self.bias is not None, jinak 0.0)

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku tvaru `(n_vzorku, n_priznaku)`, kde
            `n_priznaku` odpovida prvnimu rozmeru `self.weights`.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole tvaru `(n_vzorku,)` (pro vektor `weights`), nebo obecneji
            `(n_vzorku, n_jednotek)` (pro matici `weights`).
        """
        # assert  Ověřte, že x je 2D pole (ma atribut ndim == 2)
        # assert  Ověřte, že x.shape[1] odpovida prvnimu rozmeru self.weights
        raise NotImplementedError(
            "Úkol: spocitejte z = x @ self.weights + (self.bias if self.bias "
            "is not None else 0.0) a vratte ho - zadnou aktivaci zde neaplikujte, "
            "Linear pocita jen afinni transformaci."
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zkratka pro `forward` - umoznuje volat instanci jako funkci.

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku predana primo do `forward`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vysledek `self.forward(x)`.
        """
        return self.forward(x)
