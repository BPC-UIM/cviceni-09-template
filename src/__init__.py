"""Verejne API balicku ``src`` pro cviceni 09.

Obsahuje:

* ``Activation`` (ABC) -- spolecne rozhrani rodiny aktivacnich funkci
  (brana z Cviceni 08).
* ``Step`` / ``Sigmoid`` / ``Tanh`` / ``ReLU`` / ``ReLU6`` -- konkretni
  aktivace (brana z Cviceni 08).
* ``make_activation`` -- tovarni funkce: jmeno aktivace -> instance
  ``Activation`` (brana z Cviceni 08).
* ``Linear`` -- afinni vrstva ``x @ weights + bias``; s matici vah
  ``(n_vstupu, n_jednotek)`` reprezentuje celou vrstvu (brana z Cviceni 08).
* ``Neuron`` -- slozeni ``Linear`` a injektovane ``Activation``; v tomto
  cviceni slouzi jako vrstva site (brana z Cviceni 08).
* ``Sequential`` -- retezeni vrstev se zaznamem mezivystupu ``io_`` (NOVE).

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from src.activations import Activation, ReLU, ReLU6, Sigmoid, Step, Tanh, make_activation
from src.linear import Linear
from src.network import Sequential
from src.neuron import Neuron

__all__ = [
    "Activation",
    "Step",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "ReLU6",
    "make_activation",
    "Linear",
    "Neuron",
    "Sequential",
]
