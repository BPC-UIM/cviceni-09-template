# Papírové příklady — Cvičení 9

Vícevrstvá síť bez učení: skládání vrstev, transformace prostoru skrytou
vrstvou a ruční návrh vah z geometrie. Příklady procvičují **přesně tu
notaci a ty výpočty**, se kterými pracujete v `src/network.py` a ve fázích
`cviceni_09.py`. Čísla jsou volena tak, aby se dala spočítat na papíře.

**Řešení nejsou součástí repozitáře.** Výsledky si ověřte u vyučujícího nebo
výpočtem v `numpy`.

Značení je stejné jako v kódu:

- `x`: matice vstupů tvaru `(N, n)`, jeden vzorek je jeden **řádek**.
- `W`: matice vah vrstvy tvaru `(n_vstupů, n_jednotek)`. **Sloupec $j$** jsou
  váhy $j$-tého neuronu, tedy normála jeho přímky.
- `b`: vektor biasu tvaru `(n_jednotek,)`.
- Vrstva počítá `z = x @ W + b` a pak aktivaci prvek po prvku. Pro jeden
  vzorek $\mathbf{x} = (x_1, x_2)$ a neuron $j$ je
  $z_j = W_{1j}\,x_1 + W_{2j}\,x_2 + b_j$.
- `Step`: $z \ge 0 \Rightarrow 1$, jinak $0$ (neostrá nerovnost).
  `Sigmoid(T)`: $\sigma_T(z) = 1/(1+e^{-z/T})$.
- `io_ = [x, h1, h2, …]`: stopa `Sequential.forward`. Souřadnice první skryté
  vrstvy značíme $y_1, y_2, \dots$, druhé skryté vrstvy $u_1, u_2, \dots$
- Výstup sítě $\ge 0{,}5$ znamená třídu 1.

Pořadí rohů jednotkového čtverce (shodné s `dataio/gates.py`):
$(0,0),\ (0,1),\ (1,0),\ (1,1)$.

---

## Příklad 1 — Ruční dopředný průchod sítí 2-2-1

Je dána síť `Sequential([vrstva_1, vrstva_2])` s aktivací `Step` v obou vrstvách:

$$W_1 = \begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix},\quad \mathbf{b}_1 = (-1,\ 0), \qquad W_2 = \begin{pmatrix} 1 \\ -2 \end{pmatrix},\quad \mathbf{b}_2 = (0{,}5).$$

Vstupní dávka:

| vzorek | $x_1$ | $x_2$ |
|:---:|:---:|:---:|
| A | $0$ | $0$ |
| B | $1$ | $0$ |
| C | $0{,}25$ | $0{,}5$ |
| D | $0$ | $1$ |

**Úkoly:**

- **a)** Z matice $W_1$ vypište váhy a bias **každého** ze dvou skrytých
  neuronů zvlášť a zapište rovnici jejich přímek ve tvaru
  $w_1 x_1 + w_2 x_2 + b = 0$. Který prvek `W1` je $W_{21}$ a ke kterému
  neuronu a vstupu patří?
- **b)** Spočítejte matici `z1 = x @ W1 + b1` pro celou dávku (tvar?) a z ní
  `io_[1] = Step(z1)`.
- **c)** Vzorek C leží na přímce jednoho ze skrytých neuronů. Kterého?
  Jakou hodnotu dá `Step` a proč?
- **d)** Spočítejte `io_[2]` a zapište výslednou třídu každého vzorku.
- **e)** Vypište tvary všech prvků `io_` a délku seznamu. Jak by se změnily
  pro dávku $N = 400$ vzorků?
- **f)** Prohoďte v `W1` oba sloupce a zároveň oba prvky `b1`. Co se stane
  s `io_[1]`? Jak musíte upravit `W2`, aby výstup sítě zůstal stejný?
  Co z toho plyne o „jménech" skrytých neuronů?
- **g)** Vysvětlete, proč `Sequential.forward` musí volat vrstvy **v pořadí**
  a proč nestačí vrátit jen poslední výstup (k čemu slouží `io_`).

---

## Příklad 2 — Návrh sítě XOR z geometrie

Hradlo XOR má výstupy $0, 1, 1, 0$. Síť má tvar 2-2-1 s aktivací `Step`.

**Úkoly:**

- **a)** Nakreslete čtyři rohy. Zdůvodněte, proč lze rohy třídy 1 oddělit
  od rohů třídy 0 **dvojicí rovnoběžných přímek** $x_1 + x_2 = c_1$
  a $x_1 + x_2 = c_2$ s $c_1 < c_2$. Určete **intervaly** přípustných $c_1$
  a $c_2$ (pozor na neostrou nerovnost ve `Step`) a jejich středy.
- **b)** Zapište neuron $y_1$, který dává 1 **nad** dolní přímkou, a neuron
  $y_2$, který dává 1 **pod** horní přímkou, jako dvojice (váhy, bias).
  Kterým logickým hradlům odpovídají?
- **c)** Sestavte `w_hidden` (tvar `(2, 2)`) a `b_hidden` (tvar `(2,)`) přesně
  v konvenci `faze_xor`.
- **d)** Doplňte tabulku souřadnic ve skrytém prostoru:

  | roh | $y_1$ | $y_2$ | XOR |
  |:---:|:---:|:---:|:---:|
  | $(0,0)$ | ? | ? | 0 |
  | $(0,1)$ | ? | ? | 1 |
  | $(1,0)$ | ? | ? | 1 |
  | $(1,1)$ | ? | ? | 0 |

- **e)** Pro výstupní neuron s váhami $(1, 1)$ určete interval přípustných
  biasů a jeho střed. Sestavte `w_output` `(2, 1)` a `b_output` `(1,)`.
- **f)** Alternativní konstrukce: skryté neurony $x_1 \wedge \neg x_2$
  a $\neg x_1 \wedge x_2$, výstup OR. Navrhněte váhy, spočítejte tabulku
  z bodu d) pro tuto síť a porovnejte, kam se nyní zobrazí rohy. Jsou oba
  skryté prostory stejné?
- **g)** Kolik parametrů (vah a biasů) má síť 2-2-1? Proč to neodporuje
  důkazu z Cvičení 08, že XOR nelze realizovat jedním neuronem?

---

## Příklad 3 — Transformace prostoru a nutnost nelinearity

Uvažujte skrytou vrstvu z příkladu 2 c).

**Úkoly:**

- **a)** Které rohy se ve skrytém prostoru (se `Step`) zobrazí do téhož bodu?
  Je zobrazení prosté? Vadí to klasifikaci? Zdůvodněte.
- **b)** Nahraďte `Step` aktivací `Sigmoid` s $T = 0{,}08$. Spočítejte
  souřadnice $(y_1, y_2)$ rohů $(0,0)$ a $(0,1)$ na tři desetinná místa.
  Pomůcky: $e^{-6{,}25} \approx 1{,}93\cdot10^{-3}$, $e^{-18{,}75} \approx 7{,}2\cdot10^{-9}$.
  Jak daleko jsou od vrcholů čtverce?
- **c)** Odstraňte aktivaci úplně (identita). Spočítejte obrazy všech čtyř
  rohů $\mathbf{y} = \mathbf{x}W + \mathbf{b}$ a zakreslete je. Proč je nyní
  **žádná přímka** v prostoru $y_1$–$y_2$ neoddělí?
- **d)** Obecně: ukažte, že dvě vrstvy bez aktivace
  $(\mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2$ jsou jedna vrstva
  $\mathbf{x}W' + \mathbf{b}'$. Vyjádřete $W'$ a $\mathbf{b}'$ a ověřte tvary.
- **e)** Pro síť z příkladu 1 (bez aktivací) spočítejte $W'$ a $\mathbf{b}'$
  číselně a zapište rovnici přímky, kterou taková „dvouvrstvá" síť realizuje.
- **f)** Dokažte: jestliže přímka $\mathbf{y}\mathbf{v} + c = 0$ odděluje
  třídy v prostoru $\mathbf{y} = \mathbf{x}W + \mathbf{b}$, pak existuje přímka,
  která je odděluje v prostoru $\mathbf{x}$. Co z toho plyne pro XOR a
  libovolně hlubokou lineární síť?

---

## Příklad 4 — Trojúhelník a měřítko vah

Trojúhelník má vrcholy $V_1 = (0, 0)$, $V_2 = (2, 0)$, $V_3 = (1, 2)$.

**Úkoly:**

- **a)** Pro každou stranu $V_1V_2$, $V_1V_3$, $V_2V_3$ napište rovnici přímky
  ve tvaru $w_1 x_1 + w_2 x_2 + b = 0$ s celočíselnými koeficienty.
- **b)** Spočítejte těžiště a pomocí něj orientujte normály **dovnitř**
  (těžiště musí dát $z > 0$ pro všechny tři neurony). Sestavte `w_hidden`
  `(2, 3)` a `b_hidden` `(3,)`.
- **c)** Výstupní neuron má váhy $(1, 1, 1)$. Odvoďte interval přípustných
  biasů pro AND tří binárních vstupů a zobecněte ho pro $k$ vstupů.
- **d)** Klasifikujte body $P = (1, 1)$, $Q = (0{,}2;\ 1)$, $R = (1, 0)$,
  $S = (1;\ 1{,}9)$, $U = (1{,}5;\ 1{,}2)$: vypište $\mathbf{z}$, $\mathbf{y}$
  i výstup. Který bod leží na hranici a jak ho `Step` klasifikuje?
- **e)** Kolik skrytých neuronů potřebuje konvexní $k$-úhelník? Co by bylo
  potřeba pro sjednocení **dvou** disjunktních trojúhelníků?
- **f)** *(měkká hranice)* Nyní použijte `Sigmoid` s $T = 0{,}08$ a normály
  **jednotkové délky** (z je pak přímo vzdálenost od přímky). Vnitřní bod leží
  ve vzdálenosti $0{,}05$ od jedné strany a daleko (vzdálenost $\approx 1$)
  od ostatních dvou. Spočítejte aktivace skrytých neuronů, vnitřní potenciál
  výstupního neuronu ($b_{\text{out}} = -2{,}5$) a výstup sítě. Pomůcky:
  $e^{-0{,}625} \approx 0{,}535$, $e^{-12{,}5} \approx 3{,}7\cdot10^{-6}$,
  $e^{-1{,}89} \approx 0{,}151$.
- **g)** Totéž pro bod ve vzdálenosti $0{,}05$ od **dvou** stran (například
  u vrcholu). Pomůcka: $e^{2{,}47} \approx 11{,}8$. Je klasifikace správná?
- **h)** Vynásobte všechny váhy i biasy obou vrstev konstantou $k = 4$
  a zopakujte bod g). Pomůcka: $e^{-2{,}5} \approx 0{,}0821$. Změnila se
  poloha některé přímky? Dokažte obecně, že $\sigma_T(kz) = \sigma_{T/k}(z)$.

---

## Příklad 5 — Motýlek jako složení sítí

Motýlek: třída 1 právě když $(x_2 > 0{,}5) \oplus (x_2 > x_1)$.

**Úkoly:**

- **a)** Označte $s_1 = [x_2 > 0{,}5]$ a $s_2 = [x_2 > x_1]$. Pro každý ze čtyř
  klínů (vlevo nahoře, vpravo nahoře, vlevo dole, vpravo dole) určete
  $(s_1, s_2)$ a třídu. Ověřte, že třída je $s_1 \oplus s_2$.
- **b)** Zdůvodněte, proč **jedna** skrytá vrstva složená z neuronů pro tyto
  dvě poloroviny nestačí, ať zvolíte výstupní neuron jakkoli.
- **c)** Navrhněte první vrstvu: `w_hidden_1` `(2, 2)` a `b_hidden_1` `(2,)`.
- **d)** Druhou a třetí vrstvu převezměte ze sítě XOR (příklad 2). Vypište
  tvary všech matic a délku stopy `io_`.
- **e)** Pro body $P = (0{,}9;\ 0{,}6)$, $Q = (0{,}2;\ 0{,}9)$,
  $R = (0{,}1;\ 0{,}4)$, $S = (0{,}8;\ 0{,}2)$ spočítejte (se `Step`) celou
  stopu `io_[1]`, `io_[2]`, `io_[3]` a porovnejte s definicí motýlka.
- **f)** Se sigmoidou: jaké souřadnice $(y_1, y_2)$ má přesně bod $(0{,}5;\ 0{,}5)$?
  Vysvětlete, proč síť zaobluje hranici v okolí průsečíku přímek a čím lze
  zaoblenou oblast zmenšit.
- **g)** Kolik parametrů má síť 2-2-2-1? Porovnejte s trojúhelníkem 2-3-1
  a vysvětlete, v čem je motýlek „těžší", přestože má menší plochu hranice.

---

## Příklad 6 — Tvary matic a počty parametrů

**Úkoly:**

- **a)** Doplňte tabulku (počet parametrů = všechny váhy + všechny biasy):

  | architektura | tvary matic vah | tvary biasů | počet parametrů |
  |:---|:---|:---|:---:|
  | 2-2-1 (XOR) | ? | ? | ? |
  | 2-3-1 (trojúhelník) | ? | ? | ? |
  | 2-2-2-1 (motýlek) | ? | ? | ? |
  | 30-16-8-1 | ? | ? | ? |

- **b)** Síť 30-16-8-1 zpracuje najednou $N = 569$ pacientů datasetu Breast
  Cancer Wisconsin (30 příznaků). Vypište tvary všech prvků `io_`.
- **c)** Obecně: vrstva s $n$ vstupy a $k$ jednotkami má kolik parametrů?
  Jak roste počet parametrů sítě, zdvojnásobíte-li šířku všech skrytých vrstev?
- **d)** Assert v `Linear.forward` kontroluje `x.shape[1] == weights.shape[0]`.
  Který z tvarů v tabulce by selhal, kdybyste omylem zapsali neurony do
  **řádků** místo sloupců? Najděte architekturu, u které by chyba **nebyla**
  odhalena, a vysvětlete proč.

---

## Příklady k procvičení

Devět bodů na téže mřížce $\{0;\,0{,}5;\,1\} \times \{0;\,0{,}5;\,1\}$ jako
v Cvičení 08 a deset sloupců štítků $y_1$–$y_{10}$. Na rozdíl od Cvičení 08
**nelze žádný z těchto sloupců realizovat jediným neuronem** — každý vyžaduje
síť s **alespoň dvěma neurony v první (skryté) vrstvě** a jedním **slučovacím
neuronem** ve vrstvě výstupní, tedy architekturu 2-$k$-1 s $k \ge 2$
(u některých sloupců se hodí $k = 3$). Aktivací je všude `Step`.

Pro každý sloupec postupujte stejně jako ve fázích `cviceni_09.py`:

1. Zakreslete devět bodů a rozhodněte, **kolik přímek** je potřeba, aby každá
   oblast vzniklá jejich rozřezáním obsahovala body jediné třídy.
2. Každou přímku zapište jako sloupec matice `w_hidden` a odpovídající prvek
   `b_hidden`; normálu orientujte dosazením konkrétního bodu.
3. Sestavte tabulku souřadnic devíti bodů ve skrytém prostoru
   ($y_1, \dots, y_k$ po aktivaci `Step`).
4. V tomto prostoru najděte **slučovací neuron** (typicky AND, OR, případně
   jejich kombinaci se zápornými vahami) a zapište `w_output`, `b_output`.
5. Ověřte, že `Step` celé sítě dá na všech devíti bodech přesně daný sloupec.
6. Uveďte počet parametrů sítě a zdůvodněte, proč jediný neuron nestačí
   (argumentem typu příklad 2 g) nebo příklad 5 b)).

*Nápověda k typickým vzorům:* dvě přímky dávají podle orientace normál **pás**
(AND dvou polorovin, viz XOR), **klín** či **roh** (AND), **sjednocení dvou
polorovin** (OR) nebo **XOR-vzor** dvou protilehlých oblastí (ten vyžaduje
ještě jednu vrstvu navíc, viz motýlek).

| $x_1$ | $x_2$ | $y_1$ | $y_2$ | $y_3$ | $y_4$ | $y_5$ | $y_6$ | $y_7$ | $y_8$ | $y_9$ | $y_{10}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 0.0 | 0.5 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 0.0 | 1.0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 0.5 | 0.0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| 0.5 | 0.5 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 |
| 0.5 | 1.0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| 1.0 | 0.0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| 1.0 | 0.5 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| 1.0 | 1.0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |

U alespoň jednoho sloupce navíc rozhodněte, zda by **stačilo méně** skrytých
neuronů, než kolik jste použili, a svou odpověď zdůvodněte — stejná otázka
jako u jednoznačnosti řešení v Cvičení 08, tentokrát ale o **počtu přímek**.
