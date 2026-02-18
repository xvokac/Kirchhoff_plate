"""Jednoduché GUI pro zadání vstupů výpočtu Kirchhoffovy desky."""

import json
import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_POLYGON = [
    [0.0, 0.0],
    [4.0, 0.0],
    [3.0, 3.0],
    [0.0, 3.0],
]

DEFAULT_ULOZENI = [
    [1, 1],
    [1, 0],
    [0, 0],
    [1, 1],
]

DEFAULT_LINE_PAR_X = [0.0, 3.6, 1.0]
DEFAULT_LINE_PAR_Y = [0.0, 3.0, 1.0]


def _build_default_edges_text():
    rows = ["# x,y,w,phi_n"]
    for point, bc in zip(DEFAULT_POLYGON, DEFAULT_ULOZENI):
        rows.append(f"{point[0]},{point[1]},{bc[0]},{bc[1]}")
    return "\n".join(rows)


class KirchhoffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kirchhoffova deska – zadání vstupů")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        form = QFormLayout()

        self.q_input = self._create_double(15.0, 0.0, 1e6, 2)
        self.lc_input = self._create_double(0.06, 0.005, 1.0, 3)
        self.d_input = self._create_double(0.2, 0.01, 5.0, 3)
        self.e_input = self._create_double(35e6, 1e3, 1e9, 0)
        self.nu_input = self._create_double(0.25, 0.0, 0.49, 3)
        self.n_pts_input = QSpinBox()
        self.n_pts_input.setRange(10, 2000)
        self.n_pts_input.setValue(100)

        self.line_x_start_input = self._create_double(DEFAULT_LINE_PAR_X[0], -1e4, 1e4, 3)
        self.line_x_stop_input = self._create_double(DEFAULT_LINE_PAR_X[1], -1e4, 1e4, 3)
        self.line_x_const_input = self._create_double(DEFAULT_LINE_PAR_X[2], -1e4, 1e4, 3)

        self.line_y_start_input = self._create_double(DEFAULT_LINE_PAR_Y[0], -1e4, 1e4, 3)
        self.line_y_stop_input = self._create_double(DEFAULT_LINE_PAR_Y[1], -1e4, 1e4, 3)
        self.line_y_const_input = self._create_double(DEFAULT_LINE_PAR_Y[2], -1e4, 1e4, 3)

        form.addRow("Spojité zatížení q [kN/m²]", self.q_input)
        form.addRow("Tloušťka desky d [m]", self.d_input)
        form.addRow("Modul pružnosti E [kPa]", self.e_input)
        form.addRow("Poissonův poměr nu [-]", self.nu_input)
        form.addRow("Délka strany prvku lc [m]", self.lc_input)
        

        self.edges_input = QTextEdit()
        self.edges_input.setPlaceholderText("x,y,w,phi_n")
        self.edges_input.setPlainText(_build_default_edges_text())

        layout.addLayout(form)
        layout.addWidget(QLabel("Geometrie + okrajové podmínky (1 řádek = 1 vrchol + podmínka následující hrany):"))
        layout.addWidget(QLabel("Formát: x, y, w=0?, phi_n=0?   | řádky s komentářem začínají #"))
        layout.addWidget(self.edges_input)

        line_form = QFormLayout()
        line_form.addRow("Linie X: x_start", self.line_x_start_input)
        line_form.addRow("Linie X: x_stop", self.line_x_stop_input)
        line_form.addRow("Linie X: y_konstantní", self.line_x_const_input)
        line_form.addRow("Linie Y: y_start", self.line_y_start_input)
        line_form.addRow("Linie Y: y_stop", self.line_y_stop_input)
        line_form.addRow("Linie Y: x_konstantní", self.line_y_const_input)
        line_form.addRow("Počet bodů liniových grafů", self.n_pts_input)

        layout.addWidget(QLabel("Parametry pro liniové grafy:"))
        layout.addLayout(line_form)

        self.status = QLabel("Nastavte hodnoty a spusťte výpočet.")

        self.run_button = QPushButton("Spustit výpočet")
        self.run_button.clicked.connect(self.run_solver)
        self._solver_process = None

        layout.addWidget(self.run_button)
        layout.addWidget(self.status)
        self.setCentralWidget(central)

    @staticmethod
    def _create_double(value, minimum, maximum, decimals):
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setSingleStep(10 ** (-decimals) if decimals > 0 else 1000.0)
        return box

    def _parse_edges_text(self):
        polygon = []
        ulozeni = []
        lines = self.edges_input.toPlainText().splitlines()

        for index, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                QMessageBox.critical(
                    self,
                    "Neplatný řádek",
                    f"Řádek {index}: očekáván formát x,y,w,phi_n.",
                )
                return None

            try:
                x = float(parts[0])
                y = float(parts[1])
                w = int(parts[2])
                phi_n = int(parts[3])
            except ValueError:
                QMessageBox.critical(
                    self,
                    "Neplatná data",
                    f"Řádek {index}: x,y musí být čísla a w,phi_n musí být 0/1.",
                )
                return None

            if w not in (0, 1) or phi_n not in (0, 1):
                QMessageBox.critical(
                    self,
                    "Neplatná data",
                    f"Řádek {index}: w a phi_n mohou být pouze 0 nebo 1.",
                )
                return None

            polygon.append([x, y])
            ulozeni.append([w, phi_n])

        if len(polygon) < 3:
            QMessageBox.critical(self, "Neplatný polygon", "Je potřeba zadat alespoň 3 vrcholy.")
            return None

        return polygon, ulozeni

    def _build_line_params(self):
        line_par_x = [
            self.line_x_start_input.value(),
            self.line_x_stop_input.value(),
            self.line_x_const_input.value(),
        ]
        line_par_y = [
            self.line_y_start_input.value(),
            self.line_y_stop_input.value(),
            self.line_y_const_input.value(),
        ]
        return line_par_x, line_par_y

    def run_solver(self):
        if self._solver_process is not None and self._solver_process.poll() is None:
            self.status.setText("Výpočet už běží. Počkejte na dokončení aktuálního běhu.")
            return

        parsed = self._parse_edges_text()
        if parsed is None:
            return

        polygon, ulozeni = parsed
        line_par_x, line_par_y = self._build_line_params()

        env = os.environ.copy()
        env["KIRCHHOFF_Q"] = str(self.q_input.value())
        env["KIRCHHOFF_LC"] = str(self.lc_input.value())
        env["KIRCHHOFF_D"] = str(self.d_input.value())
        env["KIRCHHOFF_E"] = str(self.e_input.value())
        env["KIRCHHOFF_NU"] = str(self.nu_input.value())
        env["KIRCHHOFF_N_QUERY_PTS"] = str(self.n_pts_input.value())
        env["KIRCHHOFF_POLYGON"] = json.dumps(polygon)
        env["KIRCHHOFF_ULOZENI"] = json.dumps(ulozeni)
        env["KIRCHHOFF_LINE_PAR_X"] = json.dumps(line_par_x)
        env["KIRCHHOFF_LINE_PAR_Y"] = json.dumps(line_par_y)

        self.status.setText("Výpočet běží…")
        self.repaint()

        process = self._launch_solver_process(env)
        if process is None:
            self.status.setText("Výpočet selhal – nepodařilo se spustit výpočetní proces.")
            return

        self._solver_process = process
        self.status.setText(
            "Výpočet byl spuštěn na pozadí. Grafická okna se otevřou samostatně a GUI zůstává aktivní. Další výpočet možný po uzavření všech grafických oken."
        )

    def _launch_solver_process(self, env):
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--run-solver"]
        else:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-solver"]

        try:
            # Důležité: proces nespouštíme jako "detached".
            # U některých prostředí (hlavně Windows + matplotlib) to bránilo otevření grafických oken.
            return subprocess.Popen(cmd, env=env)
        except OSError as error:
            QMessageBox.critical(self, "Chyba spuštění", f"Nepodařilo se spustit výpočet: {error}")
            return None


def run_solver_entrypoint():
    import runpy

    # Spuštění přes runpy zajistí, že se v kirchhoff_plate.py vykoná
    # i blok `if __name__ == "__main__":`, který ukládá report, obrázky
    # a otevírá matplotlib okna.
    runpy.run_module("kirchhoff_plate", run_name="__main__")


def main():
    if "--run-solver" in sys.argv:
        run_solver_entrypoint()
        return

    app = QApplication(sys.argv)
    window = KirchhoffWindow()
    window.resize(760, 820)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
