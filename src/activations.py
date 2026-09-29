"""Rodina aktivacnich funkci (vzor Strategy) pro neuron a vrstvy site.

BRANA (continuity gate): Zkopirujte sem sve reseni `src/activations.py`
z Cviceni 08. Rozhrani (signatury) je s Cvicenim 08 totozne, staci doplnit
tela metod. Aktivace pracuji prvek po prvku, takze beze zmeny funguji i na
vystupu cele vrstvy tvaru `(n_vzorku, n_jednotek)`.

Trida `Activation` definuje spolecne rozhrani (`forward`, `output_range`);
konkretni potomci (`Step`, `Sigmoid`, `Tanh`, `ReLU`, `ReLU6`) sdileji jen
toto rozhrani, ne kod. Instance `Activation` se injektuje do `Neuron`
(dependency injection, kap. 9 konfiguratoru) - neuron si aktivaci sam
nevytvari, dostava ji hotovou zvenku.

Modul navic definuje tovarni funkci `make_activation` (Strategy + Factory,
kap. 8c konfiguratoru): z ulozeneho jmena aktivace (retezec) vytvori
odpovidajici instanci. Pouziva ji `Neuron.load` pri rekonstrukci ulozeneho
modelu - ulozeny .npz nese jen jmeno aktivace jako metadata, ne cely objekt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Abstraktni zaklad rodiny aktivacnich funkci (plocha ABC hierarchie).

    Jde o vzor Strategy (konfigurator kap. 8a): jednotlive aktivace jsou
    navzajem nezavisle a sdileji pouze spolecne rozhrani `forward` a
    `output_range`, ne implementaci. Diky tomu lze aktivaci, kterou pouziva
    `Neuron`, zamenit bez zasahu do teto tridy - staci injektovat jinou
    instanci potomka (dependency injection, kap. 9 konfiguratoru).
    """

    @property
    @abstractmethod
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot, ktere muze `forward` vratit.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Dvojice `(dolni_mez, horni_mez)` moznych vystupnich hodnot.
        """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provede dopredny pruchod aktivacni funkci.

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole (typicky vysledek vazeneho souctu z `Neuron.forward`).

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` po aplikaci aktivacni funkce.
        """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zkratka pro `forward` - umoznuje volat instanci jako funkci.

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole predane primo do `forward`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vysledek `self.forward(x)`.
        """
        return self.forward(x)


class Step(Activation):
    """Tvrda skokova aktivace (Heaviside step).

    Vystupem je ostry binarni rozhodnuti bez pravdepodobnostni interpretace:
    1, pokud je vstup nezaporny, jinak 0. Pouziva se pro "tvrde" klasifikatory
    (napr. AND/OR hradla v bloku 1 pipeline).
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte skokovou aktivaci.

        Definice
        --------
        f(x) = 1, pokud x >= 0
        f(x) = 0, pokud x < 0

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami z mnoziny {0.0, 1.0}.
        """
        # assert  Ověřte, že x je typu np.ndarray
        # assert  Ověřte, že vystup obsahuje jen hodnoty 0.0 a 1.0
        raise NotImplementedError(
            "Úkol: implementujte f(x) = 1 pro x >= 0, jinak 0, vektorizovane "
            "pomoci np.where (zadna python smycka pres prvky x)."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot skokove aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, 1.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: vratte konstantni tuple (0.0, 1.0) jako obor hodnot Step aktivace."
        )


class Sigmoid(Activation):
    """Hladka (meka) aktivace s nastavitelnou teplotou.

    Logisticka funkce parametrizovana `temperature`, ktera rika, jak ostry
    je prechod mezi 0 a 1. Cim mensi `temperature`, tim strmejsi je prechod a
    tim vice se sigmoida podoba tvrdemu `Step` (v limite `temperature -> 0+`
    se oba prubehy prakticky slevaji); cim vetsi `temperature`, tim je
    prechod plosejsi a rozprostreny pres sirsi rozsah vstupu `x`. Tento
    kontrast (male vs. velke `temperature`) se vyuziva v exploracnim sweepu
    pipeline (blok 3), kde se na stejnych datech ukazuje prechod od tvrde
    k meke rozhodovaci hranici.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Ulozi teplotu sigmoidy.

        Parametry
        ---------
        temperature : float
            Kladne cislo ridici strmost prechodu; vychozi hodnota 1.0.
        """
        self.temperature = temperature

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte sigmoidu s teplotou.

        Definice
        --------
        f(x) = 1 / (1 + exp(-x / temperature))

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v otevrenem intervalu (0.0, 1.0).
        """
        # assert  Ověřte, že x je typu np.ndarray
        # assert  Ověřte, že self.temperature je kladne cislo
        raise NotImplementedError(
            "Úkol: implementujte f(x) = 1 / (1 + exp(-x / self.temperature)) "
            "pomoci np.exp, vektorizovane pres cele pole x."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot sigmoidy.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, 1.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: vratte konstantni tuple (0.0, 1.0) jako obor hodnot Sigmoid aktivace."
        )


class Tanh(Activation):
    """Hyperbolicky tangens jako symetricka meka aktivace.

    Na rozdil od `Sigmoid` je vystup symetricky kolem nuly a nabyva zapornych
    i kladnych hodnot, coz muze byt vyhodne pro rychlejsi konvergenci u
    pozdejsich (uz uceni schopnych) modelu v cv9+.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte hyperbolicky tangens.

        Definice
        --------
        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v otevrenem intervalu (-1.0, 1.0).
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: implementujte f(x) = tanh(x) pomoci np.tanh, vektorizovane "
            "pres cele pole x."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot tanh aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(-1.0, 1.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: vratte konstantni tuple (-1.0, 1.0) jako obor hodnot Tanh aktivace."
        )


class ReLU(Activation):
    """Rectified Linear Unit - useknuti zapornych hodnot na nule.

    Nejpouzivanejsi aktivace v modernich sitich (cv9+): pro zaporny vstup
    vraci 0, pro nezaporny vstup vraci vstup samotny beze zmeny. Vystup je
    shora neomezeny.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte ReLU.

        Definice
        --------
        f(x) = max(0, x)

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v intervalu [0.0, +inf).
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: implementujte f(x) = max(0, x) pomoci np.maximum, "
            "vektorizovane pres cele pole x."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot ReLU aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, np.inf)`.
        """
        # assert  Ověřte, že horni mez je +nekonecno (np.inf), ne konecne cislo
        raise NotImplementedError(
            "Úkol: vratte konstantni tuple (0.0, np.inf) jako obor hodnot ReLU aktivace."
        )


class ReLU6(Activation):
    """ReLU oriznuta shora na hodnotu 6.

    Stejne jako `ReLU` useka zaporne hodnoty na nule, navic ale shora oriznou
    vystup na 6, cimz ziska (na rozdil od `ReLU`) konecny obor hodnot -
    uzitecne napriklad pro numerickou stabilitu nebo kvantizaci v pozdejsich
    sitich (cv9+).
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte ReLU6.

        Definice
        --------
        f(x) = min(max(0, x), 6)

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v intervalu [0.0, 6.0].
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: implementujte f(x) = min(max(0, x), 6) pomoci np.clip "
            "(nebo kombinace np.minimum/np.maximum), vektorizovane pres cele pole x."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot ReLU6 aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, 6.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: vratte konstantni tuple (0.0, 6.0) jako obor hodnot ReLU6 aktivace."
        )


# Registr jmen aktivaci na tridy - podklad pro `make_activation` nize.
_ACTIVATION_REGISTRY: dict[str, type[Activation]] = {
    "Step": Step,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "ReLU": ReLU,
    "ReLU6": ReLU6,
}


def make_activation(name: str, *, temperature: float = 1.0) -> Activation:
    """Tovarni funkce: vytvori instanci `Activation` podle jmena (Strategy + Factory).

    Predvyplnena infrastruktura pro `Neuron.load` (modul `src.neuron`) - ulozeny
    .npz archiv nese jen jmeno aktivace jako textove metadata (`type(x).__name__`),
    ne cely objekt, takze pri nacitani je potreba jmeno zpetne prevest na instanci.

    Parametry
    ---------
    name : str
        Jmeno tridy aktivace, jedno z `"Step"`, `"Sigmoid"`, `"Tanh"`, `"ReLU"`,
        `"ReLU6"` (presne shoda velikosti pismen, odpovida `type(x).__name__`).
    temperature : float, optional
        Teplota pro `Sigmoid` (viz `Sigmoid.__init__`); pro ostatni aktivace
        se ignoruje. Vychozi `1.0`.

    Navratova hodnota
    -----------------
    Activation
        Nova instance prislusne tridy (`Sigmoid` dostane `temperature`,
        ostatni aktivace nemaji zadne parametry konstruktoru).

    Vyjimky
    -------
    ``ValueError``:
        Pokud `name` neni v `_ACTIVATION_REGISTRY`.
    """
    if name not in _ACTIVATION_REGISTRY:
        povolene = ", ".join(sorted(_ACTIVATION_REGISTRY))
        raise ValueError(f"Neznama aktivace: {name!r}. Povolene hodnoty jsou: {povolene}.")

    cls = _ACTIVATION_REGISTRY[name]
    if cls is Sigmoid:
        return Sigmoid(temperature=temperature)
    return cls()
