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
2. **kirchhoff_plate_gui.py** – jednoduché GUI pro zadání vstupů výpočtu.

## Jak začít

Příkazy jsou použity pro systém Windows. V případě Linux nebo macOS je třeba
použít odpovídající ekvivalenty pro tyto systémy, např. `python3` místo `python` atd.

Je třeba mít instalován Python a odpovídající závislosti.

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
Proto může být vhodnější spouštět skript v programu Python. Případně optimalizovat a volit jinou metodu, např.

```bash
REM je treba rozbalit UPX  z https://upx.github.io/ do  C:\upx
python -O -m PyInstaller ^
  --onedir --windowed kirchhoff_plate_gui.py ^
  --upx-dir C:\upx  
```
## Hlavní okno

Vstupní data se zadávají pomocí hlavního okna aplikace.

![input data](images/Clipboard01.png)

Hlavní okno obsahuje několik položek:

- Spojité zatížení na desku
- Tloušťka desky
- Modul pružnosti
- Poissonův poměr
- Délka strany prvku - generuje se síť MKP s tímto parametrem
- Geometrie desky - se zadává pomocí tabulky, každá řádek začíná souřadnicemi 
vrcholu polygonu a následují boolen proměnné, které se váže k následující 
straně polygonu a určují okrajové podmínky. Např. kombinace [0,0] je pro volný okraj desky, [1,1] vetknutí, [0,1] podmínka osy symetrie, [1,0] válcový kloub.
- Parametry pro liniové grafy - je možné zadat úsečky rovnoběžnou s osou X a Y, pro kterou se vykreslí grafy průběhu dimenzačních mometů.
- Počet bodů liniového grafu - stanový počet bodů na úcečce, pro které se budou hledat výsledky v síti MKP.

Tlačítko **Spustit výpočet** zahájí výpočetní proces a zobrazení grafů.

Pomocí tlačítek **Uložit zadání (JSON)** a **Načíst zadání (JSON)** lze vstupní data uložit do souboru a při dalším spuštění programu je znovu načíst.

Výpočet běží v samostatném procesu, takže hlavní panel GUI zůstává aktivní i při otevřených grafech. Nový výpočet je ale možné spustit až po uzavření všech grafických oken.

Dimenzační momenty jsou počítány nejjednodušší metodou, tj. $M_{x,dim,lower} = M_x + |M_{xy}|$, $M_{x,dim,upper} = M_x - |M_{xy}|$ a podobně pro směr $Y$.

Pro výpočet ohybových mometů nemá vliv volba parametrů tloušťky desky a modulu pružnosti. 
Na průběh momentů má vliv zadaný poissonův součinitel.

Tvar průhybu (deformace) je pouze orientační, je pro lineární materiál, což ovšem beton není. 
Proto desková tuhost vypočtená z tloušťky desky a modulu pružnosti jako u lineárního pružného materiálu není reálná.

Výsledky se zapíší do aktuálního adresáře do souboru `kirchhoff_report.pdf` a grafy se uloží ve formátu PNG do adresáře `kirchhoff_plots`.

![input data](kirchhoff_plots/plot_01.png)

![input data](kirchhoff_plots/plot_02.png)

![input data](kirchhoff_plots/plot_03.png)

![input data](kirchhoff_plots/plot_04.png)

![input data](kirchhoff_plots/plot_05.png)

![input data](kirchhoff_plots/plot_06.png)

![input data](kirchhoff_plots/plot_07.png)







