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
    [3.0, 0.0],
    [3.0, 3.0],
    [0.0, 3.0],
    [0.0, 0.2],
    [0.2, 0.2],
    [0.2, 0.0],
]

DEFAULT_ULOZENI = [
    [0, 1],
    [0, 1],
    [0, 1],
    [1, 1],
    [1, 1],
    [0, 1],
]


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

        form.addRow("q [kN/m²]", self.q_input)
        form.addRow("lc [m]", self.lc_input)
        form.addRow("d [m]", self.d_input)
        form.addRow("E [kPa]", self.e_input)
        form.addRow("nu [-]", self.nu_input)
        form.addRow("Počet bodů liniových grafů", self.n_pts_input)

        self.polygon_input = QTextEdit()
        self.polygon_input.setPlaceholderText("[[x1, y1], [x2, y2], ...]")
        self.polygon_input.setPlainText(json.dumps(DEFAULT_POLYGON, indent=2, ensure_ascii=False))

        self.ulozeni_input = QTextEdit()
        self.ulozeni_input.setPlaceholderText("[[w1, phi1], [w2, phi2], ...]")
        self.ulozeni_input.setPlainText(json.dumps(DEFAULT_ULOZENI, indent=2, ensure_ascii=False))

        layout.addLayout(form)
        layout.addWidget(QLabel("Polygon (JSON seznam bodů [x, y]):"))
        layout.addWidget(self.polygon_input)
        layout.addWidget(QLabel("Okrajové podmínky (JSON seznam [w, φn] pro každou hranu):"))
        layout.addWidget(self.ulozeni_input)

        self.status = QLabel("Nastavte hodnoty a spusťte výpočet.")

        run_button = QPushButton("Spustit výpočet")
        run_button.clicked.connect(self.run_solver)

        layout.addWidget(run_button)
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

    def _validate_json_fields(self):
        try:
            polygon = json.loads(self.polygon_input.toPlainText())
            ulozeni = json.loads(self.ulozeni_input.toPlainText())
        except json.JSONDecodeError as err:
            QMessageBox.critical(self, "Neplatný JSON", f"Chyba JSON: {err}")
            return None

        if not isinstance(polygon, list) or len(polygon) < 3:
            QMessageBox.critical(self, "Neplatný polygon", "Polygon musí být seznam alespoň 3 bodů.")
            return None

        if not all(isinstance(p, list) and len(p) == 2 for p in polygon):
            QMessageBox.critical(self, "Neplatný polygon", "Každý bod polygonu musí být [x, y].")
            return None

        if not isinstance(ulozeni, list) or len(ulozeni) != len(polygon):
            QMessageBox.critical(
                self,
                "Neplatné uložení",
                "`ulozeni` musí mít stejný počet řádků jako polygon hran.",
            )
            return None

        if not all(isinstance(u, list) and len(u) == 2 for u in ulozeni):
            QMessageBox.critical(self, "Neplatné uložení", "Každý řádek `ulozeni` musí být [w, φn].")
            return None

        return polygon, ulozeni

    def run_solver(self):
        validated = self._validate_json_fields()
        if validated is None:
            return

        polygon, ulozeni = validated
        env = os.environ.copy()
        env["KIRCHHOFF_Q"] = str(self.q_input.value())
        env["KIRCHHOFF_LC"] = str(self.lc_input.value())
        env["KIRCHHOFF_D"] = str(self.d_input.value())
        env["KIRCHHOFF_E"] = str(self.e_input.value())
        env["KIRCHHOFF_NU"] = str(self.nu_input.value())
        env["KIRCHHOFF_N_QUERY_PTS"] = str(self.n_pts_input.value())
        env["KIRCHHOFF_POLYGON"] = json.dumps(polygon)
        env["KIRCHHOFF_ULOZENI"] = json.dumps(ulozeni)

        script_path = Path(__file__).with_name("kirchhoff_plate.py")

        self.status.setText("Výpočet běží…")
        self.repaint()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            self.status.setText("Hotovo. Výsledek byl spočten, grafy se otevřely v samostatných oknech.")
        else:
            self.status.setText("Výpočet selhal – podrobnosti jsou vypsané v konzoli.")
            print(result.stdout)
            print(result.stderr)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KirchhoffWindow()
    window.resize(640, 760)
    window.show()
    sys.exit(app.exec_())
