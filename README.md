# Kirchhoff_plate

Jednoduchý program pro výpočet Kirchhoffovy desky pomocí
jednoduchých prvků Morley. Jedná se o nejjednodušší prvek MPK, kterým
lze řešit tuto problematiku. 
Ohybové a krouticí momenty jsou po prvku aproximovány konstantní funkcí.

![input data](images/Clipboard.png)

## Quick start

```bash
git clone https://github.com/xvokac/Kirchhoff_plate
cd Kirchhoff_plate
pip install PyQt5 numpy scikit-fem matplotlib gmsh meshio
python kirchhoff_plate_gui.py
```

---

## Cíl

Hlavním cílem je:

- vytvořit výukový program,
- demonstrovat použití MKP,
- demonstrovat ohybové momenty na desce včetně krouticích,
- ukázat problematiku dimenzačních momentů v železobetonu.


## Klíčové soubory

1. **kirchhoff_plate_gui.py** – GUI pro zadání vstupů výpočtu a ovládání programu,
2. **kirchhoff_plate.py** – solver.

## Jak začít 

Příkazy jsou použity pro systém Windows. V případě Linux nebo macOS je třeba
použít odpovídající ekvivalenty pro tyto systémy, např. `python3` místo `python` atd.

Je třeba mít instalován Python a odpovídající závislosti, které lze instalovat příkazem:

```bash
pip install PyQt5 numpy scikit-fem matplotlib gmsh meshio
```

V adresáři, kde jsou uloženy klíčové soubory, lze spusti aplikaci např. příkazem 

```bash
python kirchhoff_plate_gui.py
```


## Vytvoření `.exe` pomocí PyInstalleru

Aplikace je připravená na spuštění výpočtu i po zabalení do jednoho `exe` souboru. 
Výhodou může být, že tento soubor `.exe` lze spustit na PC, kde není instalovýn Python. 
Ve Windows lze použít příkazy:

```bash
pip install pyinstaller
python -m PyInstaller --onefile --windowed kirchhoff_plate_gui.py
```

Po dokončení bude spustitelný soubor ve složce `dist/`. 
Je třeba počítat s jeho větší velikostí, protože obsahuje všechny knihovny. 
Případně je možné volit jinou metodu, např. rozbalit UPX pro optimalizaci z `https://upx.github.io/` do  `C:\upx` a použít:

```bash 
python -O -m PyInstaller ^
  --onedir --windowed kirchhoff_plate_gui.py ^
  --upx-dir C:\upx  
```

## Omezení

- lineární pružný model
- řešenou oblastí je uzavřený polygon
- liniové podpory (či podmínka symetrie) po obvodu řešené oblasti
- neřeší vnitřní podpory
- neřeší trhliny ani nelinearity
- není určen pro projektování konstrukcí
- slouží pouze pro výuku a demonstraci

---

## Hlavní okno

Vstupní data se zadávají pomocí hlavního okna aplikace.

![input data](images/Clipboard01.png)

Hlavní okno obsahuje několik položek:

- Jméno projektu - vytvoří v aktuálním adresáři podadresář s tímto názvem pro uložení vstupních a výstupních souborů programu
- Spojité zatížení na desku
- Poissonův poměr
- Délka strany prvku - generuje se síť MKP s tímto parametrem
- Geometrie desky - pomocí tabulky se zadává geometrie uzavřeného polygonu ohraničující řešenou oblast. Každá řádek začíná souřadnicemi 
vrcholu polygonu (X, Y) a následují boolen proměnné k následující 
straně polygonu určují okrajové podmínky, resp. uložení na této hraně ($w=0$, $\varphi_n=0$). Např. kombinace okrajových podmínek [0,0] je pro volný okraj desky, [1,1] vetknutí, [0,1] podmínka osy symetrie, [1,0] válcový kloub.
- Parametry pro liniové grafy - je možné zadat úsečku rovnoběžnou s osou $X$ a $Y$, pro kterou se vykreslí grafy průběhu dimenzačních momentů. Souřadnice musí být v dané oblasti řešení desky, jinak se výpočet programu zastaví a oznámí chybové hlášení, které se loguje také do `kirchhoff_gui.log`
- Počet bodů liniového grafu - stanoví se počet bodů na úcečce, pro které se budou hledat výsledky v síti MKP.

Pomocí tlačítek **Uložit zadání (JSON)** a **Načíst zadání (JSON)** lze aktuální vstupní data uložit do souboru ve formátu `JSON` a při dalším spuštění programu je znovu načíst.

Tlačítko **Generovat síť** vygeneruje a zobrazí síť prvků a označí i okrajové podmínky. Slouží pro kontrolu zadané geometrie a sítě prvků. 
Výpočet se neprovádí.

Tlačítko **Spustit výpočet** zahájí výpočetní proces a zobrazení výstupních grafů a uložení výsledků do adresáře projektu. 
V závislosti na velikosti úlohy může proces na pozadí trvat delší dobu.

Tlačítko **Zavřít všechny grafy** uzavře všechna grafická okna solveru a je potom možné spustit další výpočet.

Výpočet běží v samostatném procesu, takže hlavní panel GUI zůstává aktivní i při otevřených grafech. 
Nový výpočet je ale možné spustit až po uzavření všech grafických oken předchozího výpočtu.

Dimenzační momenty jsou počítány metodou pro dolní výztuž $M_{x,dim,lower} = M_x + |M_{xy}|$ a dále pro horní výztuž $M_{x,dim,upper} = M_x - |M_{xy}|$.  
Analogicky platí vztahy pro směr $Y$. Předopkládá se orientace nosné výztuže ve směru os $X$ a $Y$.

Pro výpočet ohybových momentů nemá vliv volba parametrů tloušťky desky a modulu pružnosti.
Proto tyto parametry nejsou zadávány v okně GUI, výpočet používá implicitní hodnoty.
Tvar průhybu (deformace) je pouze orientační, platil by pro lineární pružný materiál, což ovšem beton nebo železobeton není.
Deformace, resp. průhyb $w(x,y)$, je proto prezentována pro jednotkovou deskovou tuhost.
Na průběh momentů má ale vliv zadaný poissonův součinitel $\mu$ .

Výsledky se zapíší do adresáře projektu do souboru `kirchhoff_report.pdf` a grafy se uloží ve formátu PNG do podadresáře `kirchhoff_plots`. 
Do adresáře projektu se také zapíše soubor `kirchhoff_input.json` se zadáním výpočtu, které lze programem znovu načíst při dalším spuštění. 
Text souboru `kirchhoff_input.json` je také v úvodu `kirchhoff_report.pdf` a obsahuje také extrémy ohybových momentů (minima, maxima) a jejch souřadnice X a Y.

---

## Example_01

Jedná se o implicitní zadání, které se načte po startu programu. 

Jedná se o lichoběžníkový tvar desky, se souřadnicemi vrcholů [0, 0]; [4, 0]; [3, 3] a [0, 3]. 
Tyto údaje jsou v prvních dvou sloupcích zadání geometrie.

Okrajové podmínky (způsob podepření desky) jsou:
- první strana mezi body č. 1 a 2: vetknutí [1, 1];
- druhá strana mezi body č. 2 a 3: válcový kloub [1, 0];
- třetí strana mezi body č. 3 a 4: volný okraj [0, 0];
- čtvrtá strana mezi body č. 4 a 1: vetknutí [1, 1];
Tyto údaje jsou ve třetím a čtvrtým sloupci zadání geometrie desky.

Liniové grafy jsou voleny v místech s velkou hodnotou krouticího moementu tak, aby bylo možné ukázat jejich vliv na dimenzování výztuže.

Grafické výstupy jsou následující. 

<p align="center">
  <img src="Example_01/kirchhoff_plots/plot_01.png" width="45%">
  <img src="Example_01/kirchhoff_plots/plot_02.png" width="45%">
</p>

![input data](Example_01/kirchhoff_plots/plot_03.png)

![input data](Example_01/kirchhoff_plots/plot_04.png)

![input data](Example_01/kirchhoff_plots/plot_05.png)

<p align="center">
  <img src="Example_01/kirchhoff_plots/plot_06.png" width="45%">
  <img src="Example_01/kirchhoff_plots/plot_07.png" width="45%">
</p>


## Example_02

Jedná se o desku vetknutou na rozpětí $L_y = 2$ m se zatížemím 10 kN/m^2. 
Ohybový moment $m_y$ na 1 m šířky desky ve vetknutí by měl být $(1/12)qL_y^2 = 3,33$ kNm/m a v poli $(1/24)qL_y^2 = 1,67$ kNm/m. 
Deska je dlouhá, kolmý rozměr na nosnou výstuž je $L_x = 6$ m, a proto velikost $m_x = m_y * \mu$. 
Rozdělovací výztuž se proto provádí jako $\mu$ násobek hlavní nosné výztuže.

<p align="center">
  <img src="Example_02/kirchhoff_plots/plot_01.png" width="80%">
</p>

<p align="center">
  <img src="Example_02/kirchhoff_plots/plot_07.png" width="45%">
  <img src="Example_02/kirchhoff_plots/plot_06.png" width="45%">
</p>


## Example_03

Jedná se o desku prostě podepřenou na $L_x = 6$ m se zatížením 10 kN/m^2. Využívá se podmínky symetrie, proto se řeší jen levá polovina konstrukce. 
Ohybový moment $m_x$ na šířku desky 1 m má být $m_x = (1/8) qL_x^2 = 45,0$ kNm/m.
Kolmý směr je velmi malý, je pouze $L_y = 1$ m. Působení má proto blíže nosníku a příčné přetvořené (Poissonův poměr) se příliš neuplatní.

<p align="center">
  <img src="Example_03/kirchhoff_plots/plot_01.png" width="80%">
</p>

<p align="center">
  <img src="Example_03/kirchhoff_plots/plot_06.png" width="45%">
  <img src="Example_03/kirchhoff_plots/plot_07.png" width="45%">
</p>


## Example_04

Jedná se o čtvercovou desku  $L_x = L_x = 6$ m se zatížením 20 kN/m^2. 
Po celém obvodu je válcový kloub. 
Při zjednodušeném postupu, kde se usuzuje na velikost $q_x$ a $q_y$ z rovnosti průhybů nosníků ve středu desky ve směru $X$ a $Y$ a řeší se každý směr odděleně, 
by v tomto případě bylo možné usuzovat na $q_x = q_y = 10$ kN/m^2 a $m_x = m_y = (1/8) q_x L_x^2 = 45,0$ kNm/m.
Z výstupů programu je patrné, že zjednodušený postup pro křížem vyztuženou desku v tomto případě dimenzační momenty nadhodnocuje.

<p align="center">
  <img src="Example_04/kirchhoff_plots/plot_01.png" width="45%">
  <img src="Example_04/kirchhoff_plots/plot_06.png" width="45%">
</p>


## Example_05

Obdoba předchozího příkladu, po obvodu je ale v tomto případě vetknutí.
Zjednodušeným postupem by byl ohybový moment v poli $m_x = m_y = (1/24) q_x L_x^2 = 15,0$ kNm/m a ve vetknutí $m_x = m_y = (1/12) q_x L_x^2 = 30,0$ kNm/m.
Zde je přibližná metoda výpočtu ve větší shodě a její výsledky jsou tentokrát menší než hodnoty z MKP.

<p align="center">
  <img src="Example_05/kirchhoff_plots/plot_01.png" width="45%">
  <img src="Example_05/kirchhoff_plots/plot_06.png" width="45%">
</p>


## Example_06

U složitějších úloh je třeba počítat se vznikem singularit, viz následující příklad. Jeden prvek má výrazně větší hodnotu ohybového momentu. 
V takovém případě je možné pro lepší čitelnost izolinií v grafu oříznou zobrazované hodnoty ohybových momentů následujícím způsobem.

![input data](images/Clipboard21.png)

![input data](images/Clipboard22.png)

<p align="center">
  <img src="Example_06/kirchhoff_plots/plot_01.png" width="45%">
  <img src="Example_06/kirchhoff_plots/plot_06.png" width="45%">
</p>




