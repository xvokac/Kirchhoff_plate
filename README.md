# Kirchhoff_plate

Jednoduchý program pro výpočet Kirchhoffovy desky pomocí
jednoduchých prvků Morley. Jedná se o nejjednodušší prvek MPK, kterým
lze řešit tuto problematiku. 
Ohybové a krouticí momenty jsou po prvku aproximovány konstantní funkcí.

---

## Cíl

Hlavním cílem je:

- výukový program,
- demonstrovat použití MKP,
- demonstrovat ohybové momenty na desce včetně krouticích,
- ukázat problematiku dimenzačních mometů v železobetonu.


## Klíčové soubory

1. **kirchhoff_plate.py** – vlastní výpočet,
2. **kirchhoff_plate_gui.py** – jednoduché GUI pro zadání vstupů výpočtu a ovládání programu.

## Jak začít

Příkazy jsou použity pro systém Windows. V případě Linux nebo macOS je třeba
použít odpovídající ekvivalenty pro tyto systémy, např. `python3` místo `python` atd.

Je třeba mít instalován Python a odpovídající závislosti, které lze instalovet příkazem:

```bash
pip install PyQt5 numpy scikit-fem matplotlib gmsh
```

Spuštění aplikace např. příkazem 
```bash
python kirchhoff_plate_gui.py
```


## Vytvoření `.exe` pomocí PyInstalleru

Aplikace je připravená na spuštění výpočtu i po zabalení do jednoho `exe` souboru. Ve Windows lze použít příkazy:

```bash
pip install pyinstaller
python -m PyInstaller --onefile --windowed kirchhoff_plate_gui.py
```

Po dokončení bude spustitelný soubor ve složce `dist/` (např. `dist/kirchhoff_plate_gui.exe`). 
Je třeba ale potom počítat s jeho větší velikostí, protože obsahuje všechny knihovny, i ty co nutně nepotřebuje. 
Proto může být vhodnější pro běžného uživatele Python spouštět skript přímo v programu Python, jak je popsáno výše. 
Případně je možné volit jinou metodu, např. rozbalit UPX pro optimalizaci z `https://upx.github.io/` do  `C:\upx` a použít:

```bash 
python -O -m PyInstaller ^
  --onedir --windowed kirchhoff_plate_gui.py ^
  --upx-dir C:\upx  
```

V tomto případě přepínač `--onedir` způsobí vytvoření adresáře, kde bude `.exe` soubor a potřebné knihovny k běhu programu budou v podadresáři `_internal`.

Při použití přepínače `--windowed` se neotevře konzolové černé okno, kde by mohly být zobrazovány průběžné informace o běhu programu a chybová hlášení. 
Tyto výstupy se proto logují do souboru `kirchhoff_gui.log`.

---

## Hlavní okno

Vstupní data se zadávají pomocí hlavního okna aplikace.

![input data](images/Clipboard01.png)

Hlavní okno obsahuje několik položek:

- Jméno projektu - vytvoří v aktuálním adresáři podadresář s tímto názvem pro uložení vstupních a výstupních souborů programu
- Spojité zatížení na desku
- Tloušťka desky
- Modul pružnosti
- Poissonův poměr
- Délka strany prvku - generuje se síť MKP s tímto parametrem
- Geometrie desky - se zadává pomocí tabulky, každá řádek začíná souřadnicemi 
vrcholu polygonu a následují boolen proměnné, které se váže k následující 
straně polygonu a určují okrajové podmínky. Např. kombinace [0,0] je pro volný okraj desky, [1,1] vetknutí, [0,1] podmínka osy symetrie, [1,0] válcový kloub.
- Parametry pro liniové grafy - je možné zadat úsečku rovnoběžnou s osou X a Y, pro kterou se vykreslí grafy průběhu dimenzačních mometů. Souřadnice musí být v dané oblasti řešení desky, jinak se výpočet programu zastaví a oznámí chybové hlášení, které se loguje také do `kirchhoff_gui.log`
- Počet bodů liniového grafu - stanoví se počet bodů na úcečce, pro které se budou hledat výsledky v síti MKP.

Tlačítko **Generovat síť** vygeneruje a zobrazí síť prvků a označí i okrajové podmínky. Slouží pro kontrolu zadané geometrie a sítě prvků. 

Tlačítko **Spustit výpočet** zahájí výpočetní proces a zobrazení výstupních grafů a uložení výsledků do adresáře projektu. V závislti na velikosti úlohy může proces trvat delší dobu.

Pomocí tlačítek **Uložit zadání (JSON)** a **Načíst zadání (JSON)** lze aktuální vstupní data uložit do souboru ve formátu `JSON` a při dalším spuštění programu je znovu načíst.

Výpočet běží v samostatném procesu, takže hlavní panel GUI zůstává aktivní i při otevřených grafech. Nový výpočet je ale možné spustit až po uzavření všech grafických oken předchozího výpočtu.

Dimenzační momenty jsou počítány metodou $M_{x,dim,lower} = M_x + |M_{xy}|$, $M_{x,dim,upper} = M_x - |M_{xy}|$ a podobně pro směr $Y$. Předopkládá se orientace nosné výztuže ve směru os X a Y.

Pro výpočet ohybových mometů nemá vliv volba parametrů tloušťky desky a modulu pružnosti. 
Na průběh momentů má vliv zadaný poissonův součinitel.

Tvar průhybu (deformace) je pouze orientační, je pro lineární materiál, což ovšem beton není. 
Proto desková tuhost vypočtená z tloušťky desky a modulu pružnosti jako u lineárního pružného materiálu není reálná.

Výsledky se zapíší do adresáře projektu do souboru `kirchhoff_report.pdf` a grafy se uloží ve formátu PNG do podadresáře `kirchhoff_plots`. Do adresáře projektu se také zapíše soubor `kirchhoff_input.json` se zadáním výpočtu, které lze programem znovu načíst při dalším spuštění. Text souboru `kirchhoff_input.json` je také v úvodu `kirchhoff_report.pdf` a obsahuje také extrémy ohybových momentů (minima, maxima) a jejch souřadnice X a Y.

![input data](kirchhoff_plots/plot_01.png)

![input data](kirchhoff_plots/plot_02.png)

![input data](kirchhoff_plots/plot_03.png)

![input data](kirchhoff_plots/plot_04.png)

![input data](kirchhoff_plots/plot_05.png)

![input data](kirchhoff_plots/plot_06.png)

![input data](kirchhoff_plots/plot_07.png)







