# -*- coding: utf-8 -*-

"""
Created on 15. 09. 2026 at 10:55:00

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
    Testy pro cviceni 09 -- vicevrstva sit bez uceni (skladani vrstev).

    Spousteni:  pytest -v

    Ve stavu stubu se sada NACTE a testy, ktere volaji nedokoncene ukoly
    (Sequential.forward, brana src/ z Cviceni 08), se oznaci jako xfail
    (ocekavane selhani s NotImplementedError) -- sada nikdy neskonci holym
    tracebackem. Po dokonceni ukolu z nich budou prochazejici testy.

    Testy Sequential pouzivaji DummyLayer -- minimalni vrstvu v cistem numpy,
    NEZAVISLOU na brane src/ z Cviceni 08. Spravnost retezeni se tak overi
    i tehdy, kdyz brana jeste neni vlozena (kap. 10/15 konfiguratoru).

    Sest trid testu:
      TestSequential         -- retezeni vrstev a stopa io_ (DummyLayer)
      TestVrstvaSMaticiVah   -- Neuron s matici vah jako vrstva, navaznost tvaru
      TestDvouvrstvaSit      -- mechanika site na LIBOVOLNYCH vahach (zadna konkretni uloha)
      TestGroundTruth        -- trojuhelnik a motylek (predvyplnene dataio)
      TestGeneratorBodu      -- generate_points: tvar, rozsah, seed
      TestKonfigurace        -- load_config / validate_config
================================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from dataio import (
    generate_points,
    inside_butterfly,
    inside_triangle,
    load_config,
    validate_config,
)
from src import Linear, Neuron, Sequential, Sigmoid, Step

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="studentsky ukol jeste neni dokoncen")


class DummyLayer:
    """Minimalni vrstva ``relu(x @ weights + bias)`` v cistem numpy.

    Nezavisla na ``src/`` -- testy ``Sequential`` tak nevyzaduji hotovou
    branu z Cviceni 08. Pocita volani, aby slo overit, ze kazda vrstva
    probehne prave jednou.
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.maximum(0.0, x @ self.weights + self.bias)


def _mala_sit() -> tuple[Sequential, np.ndarray]:
    """Sit 2-2-1 z DummyLayer se zname vystupem a vstup ke ni.

    x = [[1, 2], [0, 1]]
    vrstva 1: W = [[1, -1], [1, 1]], b = [0, -1]  -> [[3, 0], [1, 0]]
    vrstva 2: W = [[2], [5]],        b = [1]      -> [[7], [3]]
    """
    vrstva_1 = DummyLayer([[1.0, -1.0], [1.0, 1.0]], [0.0, -1.0])
    vrstva_2 = DummyLayer([[2.0], [5.0]], [1.0])
    return Sequential([vrstva_1, vrstva_2]), np.array([[1.0, 2.0], [0.0, 1.0]])


# --------------------------------------------------------------------------- #
#  Sequential (retezeni vrstev)                                              #
# --------------------------------------------------------------------------- #
class TestSequential:
    """Sequential.forward retezi vrstvy a uklada stopu io_ = [x, out_1, ...]."""

    def test_init_uklada_vrstvy_a_io_je_none(self) -> None:
        """__init__ je PREDVYPLNENY, neni STUB."""
        sit, _ = _mala_sit()
        assert len(sit.layers) == 2
        assert sit.io_ is None

    @STUB
    def test_forward_retezi_vrstvy(self) -> None:
        sit, x = _mala_sit()
        vystup = sit.forward(x)
        assert np.allclose(vystup, np.array([[7.0], [3.0]]))

    @STUB
    def test_call_deleguje_na_forward(self) -> None:
        sit, x = _mala_sit()
        assert np.allclose(sit(x), np.array([[7.0], [3.0]]))

    @STUB
    def test_io_ma_delku_1_plus_pocet_vrstev(self) -> None:
        vrstvy = [DummyLayer(np.eye(2), np.zeros(2)) for _ in range(3)]
        sit = Sequential(vrstvy)
        sit(np.ones((5, 2)))
        assert len(sit.io_) == 1 + len(vrstvy)

    @STUB
    def test_io_zacina_vstupem_a_konci_vystupem(self) -> None:
        sit, x = _mala_sit()
        vystup = sit(x)
        assert np.array_equal(sit.io_[0], x)
        assert np.array_equal(sit.io_[-1], vystup)

    @STUB
    def test_mezivystup_je_vstupem_dalsi_vrstvy(self) -> None:
        sit, x = _mala_sit()
        sit(x)
        assert np.allclose(sit.io_[1], np.array([[3.0, 0.0], [1.0, 0.0]]))
        for k, vrstva in enumerate(sit.layers):
            assert np.allclose(sit.io_[k + 1], DummyLayer(vrstva.weights, vrstva.bias)(sit.io_[k]))

    @STUB
    def test_kazda_vrstva_probehne_prave_jednou(self) -> None:
        sit, x = _mala_sit()
        sit(x)
        assert [vrstva.calls for vrstva in sit.layers] == [1, 1]

    @STUB
    def test_poradi_vrstev_zalezi(self) -> None:
        """Skladani neni komutativni: A pak B dava obecne jiny vysledek nez B pak A."""
        a = DummyLayer([[1.0, 0.0], [0.0, 1.0]], [-1.0, 0.0])
        b = DummyLayer([[0.0, 1.0], [1.0, 0.0]], [0.0, 0.0])
        x = np.array([[0.0, 2.0]])
        assert not np.allclose(Sequential([a, b])(x), Sequential([b, a])(x))

    @STUB
    def test_novy_pruchod_prepise_stopu(self) -> None:
        sit, x = _mala_sit()
        sit(x)
        sit(x[:1])
        assert sit.io_[0].shape == (1, 2)
        assert len(sit.io_) == 3


# --------------------------------------------------------------------------- #
#  Vrstva = Neuron s matici vah (vyzaduje branu z Cviceni 08)               #
# --------------------------------------------------------------------------- #
class TestVrstvaSMaticiVah:
    """Neuron, jehoz Linear nese matici (n_vstupu, n_jednotek), je celou vrstvou."""

    @STUB
    def test_vektor_biasu_se_pricita_po_jednotkach(self) -> None:
        linear = Linear(np.eye(2), np.array([10.0, -10.0]))
        z = np.asarray(linear(np.array([[0.0, 0.0], [1.0, 2.0]])))
        assert np.allclose(z, np.array([[10.0, -10.0], [11.0, -8.0]]))

    @STUB
    def test_vystup_vrstvy_ma_tvar_n_krat_k(self) -> None:
        vrstva = Neuron(Linear(np.ones((2, 3)), np.zeros(3)), Sigmoid())
        assert np.asarray(vrstva(np.zeros((5, 2)))).shape == (5, 3)

    @STUB
    def test_vystup_prvni_vrstvy_je_platnym_vstupem_druhe(self) -> None:
        skryta = Neuron(Linear(np.ones((2, 3)), np.zeros(3)), Sigmoid())
        vystupni = Neuron(Linear(np.ones((3, 1)), np.zeros(1)), Sigmoid())
        sit = Sequential([skryta, vystupni])
        vystup = sit(np.zeros((5, 2)))
        assert np.asarray(sit.io_[1]).shape == (5, 3)
        assert np.asarray(vystup).shape == (5, 1)


# --------------------------------------------------------------------------- #
#  Mechanika dvouvrstve site na libovolnych vahach                           #
# --------------------------------------------------------------------------- #
# Vahy nize NEresi zadnou ulohu ze cviceni -- jsou zvolene libovolne, jen tak,
# aby zadny vzorek nelezel presne na hranici (vsechna |z| >= 0.15). Overuje se
# jen to, ze sit pocita to, co ma.
_W_SKRYTA = np.array([[2.0, -1.0, 0.5],
                      [1.0, 1.0, -2.0]])
_B_SKRYTA = np.array([-1.0, 0.25, 0.5])
_W_VYSTUP = np.array([[1.0], [-2.0], [3.0]])
_B_VYSTUP = np.array([0.5])
_VZORKY = np.array([[0.0, 0.0], [1.0, 0.25], [0.25, 1.0], [0.9, 0.8]])


def _sit_2_3_1(aktivace_factory) -> Sequential:
    """Sit 2-3-1 z `Neuron`u s libovolne zvolenymi vahami."""
    skryta = Neuron(Linear(_W_SKRYTA, _B_SKRYTA), aktivace_factory())
    vystupni = Neuron(Linear(_W_VYSTUP, _B_VYSTUP), aktivace_factory())
    return Sequential([skryta, vystupni])


class TestDvouvrstvaSit:
    """Sit slozena ze skutecnych `Neuron`u pocita spravne hodnoty.

    Testy zamerne neoveruji zadnou konkretni ulohu (XOR, trojuhelnik, motylek).
    Navrh vah je studentsky ukol, ktery se hodnoti presnosti a grafy z pipeline;
    zde jde jen o vecnou spravnost mechaniky pro libovolne vahy a prahy.
    """

    @STUB
    def test_skryta_vrstva_dava_ocekavane_hodnoty(self) -> None:
        """Rucne dopoctene hodnoty prvni vrstvy (Step) pro ctyri vzorky."""
        sit = _sit_2_3_1(Step)
        sit(_VZORKY)
        ocekavano = np.array([[0.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0],
                              [1.0, 1.0, 0.0]])
        assert np.allclose(np.asarray(sit.io_[1]), ocekavano)

    @STUB
    def test_vystup_site_dava_ocekavane_hodnoty(self) -> None:
        sit = _sit_2_3_1(Step)
        vystup = np.asarray(sit(_VZORKY))
        assert vystup.shape == (4, 1)
        assert np.allclose(vystup.ravel(), np.array([1.0, 1.0, 0.0, 0.0]))

    @STUB
    def test_vystup_odpovida_primemu_vypoctu_pro_nahodne_vahy(self) -> None:
        """Nezavisla kontrola tychz vzorcu spoctenych primo v numpy."""
        rng = np.random.default_rng(0)
        w_skryta, b_skryta = rng.normal(size=(2, 4)), rng.normal(size=4)
        w_vystup, b_vystup = rng.normal(size=(4, 1)), rng.normal(size=1)
        x = rng.uniform(size=(6, 2))
        sit = Sequential([
            Neuron(Linear(w_skryta, b_skryta), Sigmoid()),
            Neuron(Linear(w_vystup, b_vystup), Sigmoid()),
        ])
        skryta = 1.0 / (1.0 + np.exp(-(x @ w_skryta + b_skryta)))
        ocekavano = 1.0 / (1.0 + np.exp(-(skryta @ w_vystup + b_vystup)))
        assert np.allclose(np.asarray(sit(x)), ocekavano)
        assert np.allclose(np.asarray(sit.io_[1]), skryta)

    @STUB
    def test_nizka_teplota_sigmoidy_dava_totez_co_step(self) -> None:
        tvrde = np.asarray(_sit_2_3_1(Step)(_VZORKY)).ravel()
        mekke = np.asarray(_sit_2_3_1(lambda: Sigmoid(temperature=0.05))(_VZORKY)).ravel()
        assert np.array_equal(tvrde, (mekke >= 0.5).astype(float))

    @STUB
    def test_vystup_sigmoidove_site_lezi_mezi_nulou_a_jednickou(self) -> None:
        vystup = np.asarray(_sit_2_3_1(Sigmoid)(_VZORKY))
        assert np.all(vystup > 0.0) and np.all(vystup < 1.0)

    @STUB
    def test_jine_vahy_vystupni_vrstvy_daji_jiny_vystup(self) -> None:
        """Vystupni neuron skryte souradnice skutecne kombinuje, nejen propousti."""
        sit = Sequential([
            Neuron(Linear(_W_SKRYTA, _B_SKRYTA), Step()),
            Neuron(Linear(-_W_VYSTUP, -_B_VYSTUP), Step()),
        ])
        assert np.allclose(np.asarray(sit(_VZORKY)).ravel(), np.array([0.0, 0.0, 1.0, 1.0]))


# --------------------------------------------------------------------------- #
#  Ground truth oblasti (predvyplnene dataio)                                #
# --------------------------------------------------------------------------- #
class TestGroundTruth:
    """Referencni oblasti, se kterymi se porovnava navrzena sit."""

    VRCHOLY = np.array([[0.5, 0.2], [0.1, 0.6], [0.9, 0.6]])

    def test_motylek_protilehle_kliny(self) -> None:
        body = np.array([[0.9, 0.6], [0.1, 0.4], [0.2, 0.9], [0.8, 0.2]])
        assert np.array_equal(inside_butterfly(body), np.array([1, 1, 0, 0]))

    def test_motylek_zabira_ctvrtinu_ctverce(self) -> None:
        body = generate_points(20000, seed=0)
        assert inside_butterfly(body).mean() == pytest.approx(0.25, abs=0.015)

    def test_trojuhelnik_teziste_uvnitr_body_venku(self) -> None:
        body = np.array([self.VRCHOLY.mean(axis=0), [0.5, 0.1], [0.5, 0.7], [0.1, 0.2]])
        assert np.array_equal(inside_triangle(body, self.VRCHOLY), np.array([1, 0, 0, 0]))

    def test_trojuhelnik_nezavisi_na_poradi_vrcholu(self) -> None:
        body = generate_points(500, seed=1)
        assert np.array_equal(inside_triangle(body, self.VRCHOLY),
                              inside_triangle(body, self.VRCHOLY[::-1]))


# --------------------------------------------------------------------------- #
#  Generator bodu                                                            #
# --------------------------------------------------------------------------- #
class TestGeneratorBodu:
    """generate_points: tvar, rozsah [0, 1] a reprodukovatelnost pres seed."""

    def test_tvar_a_rozsah(self) -> None:
        body = generate_points(100, seed=42)
        assert body.shape == (100, 2)
        assert body.min() >= 0.0 and body.max() <= 1.0

    def test_stejny_seed_stejne_body(self) -> None:
        assert np.array_equal(generate_points(10, seed=7), generate_points(10, seed=7))


# --------------------------------------------------------------------------- #
#  Konfigurace                                                               #
# --------------------------------------------------------------------------- #
class TestKonfigurace:
    """load_config nacte vychozi config.yaml; validate_config odmitne nesmysly."""

    def test_vychozi_konfigurace_se_nacte(self) -> None:
        cfg = load_config()
        assert cfg.data.n_samples >= 1
        assert cfg.sigmoid.temperature > 0
        assert len(cfg.triangle.vertices) == 3

    def test_nekladna_teplota_je_chyba(self) -> None:
        cfg = load_config()
        cfg.sigmoid.temperature = 0.0
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_spatny_tvar_vrcholu_je_chyba(self) -> None:
        cfg = load_config()
        cfg.triangle.vertices = [[0.0, 0.0], [1.0, 1.0]]
        with pytest.raises(ValueError):
            validate_config(cfg)
