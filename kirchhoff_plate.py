## Kirchhoff plate problem

"""
Řešení jednoduchými trojúhelníkovými prvky Morley,
které jsou kvadratické, mají stupně volnosti:
1) uzly ve vrcholu projúhelníku - hodnota průhybu (w),
2) uzly uprostřed stran prvku - derovace ve směru normály k okraji prvku (w_n).

Na jednom prvky je tedy 6 stupňů vlnosti (3 x w + 3 x w_n).

Jedná se o tzv. nekonformní prvky. 

Ohybové momenty jsou dány druhými derivacemi funkce w,
tzn. potom jsou aproximovány na prvku konstantní funkcí.
"""

import numpy as np
from skfem import *
import meshio
import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import draw
from skfem.models.poisson import unit_load
from skfem.helpers import dd, ddot, trace, eye
import gmsh
import sys
import math
import os
import json
from datetime import datetime
from pathlib import Path


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default

def _env_json_array(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return np.array(default)
    parsed = json.loads(value)
    return np.array(parsed)


def _validate_line_params(name, line):
    if line.ndim != 1 or line.shape[0] != 3:
        raise ValueError(f"`{name}` musí mít tvar [start, stop, konst].")


"""
## ZADÁNÍ #######################################################
"""

# Zatížení desky
q = _env_float("KIRCHHOFF_Q", 15)  # zatížení v kN/m^2

# Hranice oblasti - uzavřený polygon
polygon = _env_json_array("KIRCHHOFF_POLYGON", [
    [3.0, 0.0],
    [3.0, 3.0],
    [0.0, 3.0],
    [0.0, 0.2],
    [0.2, 0.2],
    [0.2, 0.0],
])

# Hranice oblasti - okrajové podmínky
# [1, 1] - zabráněno průhybu i natočení - vetknutí
# [1, 0] - zabráněno průhybu, natočení umožněno - kloub
# [0, 0] - umořněn průhyb i natočení - volný okraj desky
# [0, 1] - umožněn průhyb, zabráněno natočení - např. osa symetrie
ulozeni = _env_json_array("KIRCHHOFF_ULOZENI", [
    [0, 1],
    [0, 1],
    [0, 1],
    [1, 1],
    [1, 1],
    [0, 1],
]).astype(bool)  # Převod na booleovské hodnoty

if polygon.ndim != 2 or polygon.shape[1] != 2:
    raise ValueError("Polygon musí být 2D pole tvaru (n, 2).")

if polygon.shape[0] < 3:
    raise ValueError("Polygon musí obsahovat alespoň 3 body.")

if ulozeni.ndim != 2 or ulozeni.shape[1] != 2:
    raise ValueError("`ulozeni` musí mít tvar (n, 2).")

if ulozeni.shape[0] != polygon.shape[0]:
    raise ValueError("Počet řádků `ulozeni` musí odpovídat počtu hran polygonu.")

# délka strany prvku v m
lc = _env_float("KIRCHHOFF_LC", .06)

# Materiál
"""
Pro výpočet ohybových momentů je zásadní POISSONův poměr
Ostatní parametry se projevují jen pro funkce w a dw/de, dw/dy.
Tyto deformace jsou ale pro linární materiál, a proto pro ŽB velmi podhodnocené!!!
"""
d = _env_float("KIRCHHOFF_D", 0.2)  # tloušťka desky v m
E = _env_float("KIRCHHOFF_E", 35e6)  # modul pružnosti v kPa
nu = _env_float("KIRCHHOFF_NU", 0.25)  # poissonův poměr

# Grafy podél linie
# 1) linie rovnoběžná s osou x
line_par_x = _env_json_array("KIRCHHOFF_LINE_PAR_X", [0.2, 3, 0])  # x_start, x_stop, y_konstantní
# 2) linie rovnoběžná s osou y
line_par_y = _env_json_array("KIRCHHOFF_LINE_PAR_Y", [0.2, 3, 0])  # y_start, y_stop, x_konstantní
_validate_line_params("KIRCHHOFF_LINE_PAR_X", line_par_x)
_validate_line_params("KIRCHHOFF_LINE_PAR_Y", line_par_y)
# 3) počet bodů na liniiovém grafu
N_query_pts = _env_int("KIRCHHOFF_N_QUERY_PTS", 100)

report_file = os.getenv("KIRCHHOFF_REPORT_FILE", "kirchhoff_report.pdf")
input_file = os.getenv("KIRCHHOFF_INPUT_FILE", "kirchhoff_input.json")
plots_dir = os.getenv("KIRCHHOFF_PLOTS_DIR", "kirchhoff_plots")


def compute_plate_response():
    """Vytvoří síť, sestaví úlohu a spočte odpověď desky."""

    gmsh.initialize()
    try:
        gmsh.model.add("t2")

        point_ids = []
        for i, (x, y) in enumerate(polygon):
            point_id = gmsh.model.geo.addPoint(x, y, 0, lc, i + 1)
            point_ids.append(point_id)

        line_ids = []
        for i in range(len(point_ids)):
            start_point = point_ids[i]
            end_point = point_ids[(i + 1) % len(point_ids)]
            line_id = gmsh.model.geo.addLine(start_point, end_point)
            line_ids.append(line_id)

        gmsh.model.geo.addCurveLoop(line_ids, 1)
        plane_surface_id = gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()

        for i, line_id in enumerate(line_ids, start=1):
            line_tag = gmsh.model.addPhysicalGroup(1, [line_id])
            gmsh.model.setPhysicalName(1, line_tag, f"boundary_line{i}")

        surface_tag = gmsh.model.addPhysicalGroup(2, [plane_surface_id])
        gmsh.model.setPhysicalName(2, surface_tag, "Surface")

        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)
        gmsh.write("deska.msh")
    finally:
        gmsh.finalize()

    msh = meshio.read("deska.msh")
    points = msh.points[:, :2]
    cells = msh.cells_dict["triangle"]

    m = MeshTri(points.T, cells.T)
    basis = Basis(m, ElementTriMorley())

    def C(T):
        return E / (1 + nu) * (T + nu / (1 - nu) * eye(trace(T), 2))

    @BilinearForm
    def bilinf(u, v, _):
        return d ** 3 / 12.0 * ddot(C(dd(u)), dd(v))

    @LinearForm
    def load(v, _):
        return q * v

    K = bilinf.assemble(basis)
    f = load.assemble(basis)
    return m, basis, K, f


"""
Pro další zpracování jsou důležitá tato pole:

m.p - souřadnice x,y uzlových bodů trojúhelníkové sítě
basis.doflocs - souřadnice bodů x,y s nějakám stupněm volnosti DOF
basis.dofs.nodal_dofs - indexi stupňů volnosti příslušející vrvholů trojúhelníkových prvků,
                tj. místa s přiřazeným stupněm volnosti pro průhyb u prvků ElementTriMorley()
basis.dofs.facet_dofs - indexi stupňů volnosti příslušející středům stran trojúhelníkových prvků,
                tj. u prvků ElementTriMorley() s přiřazeným stupněm volnosti pro natočení ve směru normály 

"""


# Funkce pro rozhodnuti, zda bod leží na úcečce
def is_point_on_segment(point, A, B, tol=1e-9):
    """
    Určí, zda bod point leží na úsečce AB.    
    Parametry:
    point: [x, y] souřadnice bodu, který zkoumáme
    A: [xA, yA] souřadnice bodu A
    B: [xB, yB] souřadnice bodu B
    tol: tolerance pro plovoucí desetinou čárku (výchozí hodnota 1e-9)   
    Výstup:
    True pokud bod leží na úsečce AB, jinak False
    """
    # Vektory pro bod, bod A a bod B
    point = np.array(point)
    A = np.array(A)
    B = np.array(B)
    # Zkontrolovat, zda je bod na přímce procházející body A a B
    AB = B - A
    AP = point - A
    # Převést vektory na 3D (přidáním třetího rozměru s hodnotou 0)
    AB_3d = np.array([AB[0], AB[1], 0])
    AP_3d = np.array([AP[0], AP[1], 0])
    # Zkontrolovat, zda bod leží na přímce AB
    cross_product = np.cross(AB_3d, AP_3d)
    if abs(cross_product[2]) > tol:
        return False  # Bod neleží na přímce
    # Zkontrolovat, zda bod leží mezi body A a B
    dot_product = np.dot(AP, AB)
    if dot_product < 0:
        return False  # Bod leží mimo segment, směrem před A
    squared_length_AB = np.dot(AB, AB)
    if dot_product > squared_length_AB:
        return False  # Bod leží mimo segment, směrem za B
    return True


def is_point_inside_or_on_boundary(point, polygon, tol=1e-9):
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        if is_point_on_segment(point, a, b, tol=tol):
            return True

    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1 + tol) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def validate_query_line_in_polygon(line_params, axis, polygon, n_samples=100):
    line_values = np.linspace(line_params[0], line_params[1], n_samples)
    for value in line_values:
        if axis == "x":
            point = np.array([value, line_params[2]])
        else:
            point = np.array([line_params[2], value])

        if not is_point_inside_or_on_boundary(point, polygon):
            raise ValueError(
                f"Linie pro {axis}-graf není celá uvnitř oblasti. Problematický bod: {point.tolist()}"
            )

def solve_plate_system(m, basis, K, f):
    # Ruční přiřazení okrajových podmínek
    # Kontrola umístění bodů s DOF na hranici s okrajovou podmínkou
    conden = np.zeros(basis.doflocs[0].shape)
    # 1) Okrajové podmínky pro pruhyb
    # ozn. stupnu volnosti ve basis.dofs.nodal_dofs
    for i in range(basis.dofs.nodal_dofs.shape[1]):
        point = [basis.doflocs[0][basis.dofs.nodal_dofs[0][i]],
                 basis.doflocs[1][basis.dofs.nodal_dofs[0][i]]]
        for j in range(polygon.shape[0]):
            if ulozeni[j][0]:
                if j == (polygon.shape[0] - 1):
                    A = [polygon[j][0], polygon[j][1]]
                    B = [polygon[0][0], polygon[0][1]]
                else:
                    A = [polygon[j][0], polygon[j][1]]
                    B = [polygon[j + 1][0], polygon[j + 1][1]]
                if is_point_on_segment(point, A, B, tol=1e-9):
                    conden[basis.dofs.nodal_dofs[0][i]] = 1

    # 2) Okrajové podmínky pro natoceni
    # ozn. stupnu volnosti ve basis.dofs.facet_dofs
    for i in range(basis.dofs.facet_dofs.shape[1]):
        point = [basis.doflocs[0][basis.dofs.facet_dofs[0][i]],
                 basis.doflocs[1][basis.dofs.facet_dofs[0][i]]]
        for j in range(polygon.shape[0]):
            if ulozeni[j][1]:
                if j == (polygon.shape[0] - 1):
                    A = [polygon[j][0], polygon[j][1]]
                    B = [polygon[0][0], polygon[0][1]]
                else:
                    A = [polygon[j][0], polygon[j][1]]
                    B = [polygon[j + 1][0], polygon[j + 1][1]]
                if is_point_on_segment(point, A, B, tol=1e-9):
                    conden[basis.dofs.facet_dofs[0][i]] = 1

    conden = conden.astype(int)
    D = np.where(conden == 1)[0]

    # řešení soustavy rovnic se staticky kondenzovanou maticí K a vektorem f
    w_raw = solve(*condense(K, f, D=D))

    # Jednotková desková tuhost D=1: přepočet průhybu násobkem deskové tuhosti.
    Dpl = E * d ** 3 / (12 * (1 - nu ** 2))
    w = w_raw * Dpl

    # Momentové veličiny se počítají z původního řešení w_raw.
    interp_grad = basis.interpolate(w_raw).grad
    phi_x = interp_grad[0]
    phi_y = interp_grad[1]
    int_points_loc, _ = basis.quadrature
    int_points_gl = basis.mapping.F(int_points_loc)

    phi_xx = np.zeros_like(phi_x)
    phi_yy = np.zeros_like(phi_x)
    phi_xy = np.zeros_like(phi_x)
    phi_yx = np.zeros_like(phi_x)
    for i in range(int_points_gl.shape[1]):
        A = np.array([[int_points_gl[0][i][0], int_points_gl[1][i][0], 1],
                      [int_points_gl[0][i][1], int_points_gl[1][i][1], 1],
                      [int_points_gl[0][i][2], int_points_gl[1][i][2], 1]])
        b_phi_x = np.array([[phi_x[i][0]], [phi_x[i][1]], [phi_x[i][2]]])
        b_phi_y = np.array([[phi_y[i][0]], [phi_y[i][1]], [phi_y[i][2]]])

        solution_phi_x = np.linalg.solve(A, b_phi_x)
        solution_phi_y = np.linalg.solve(A, b_phi_y)

        phi_xx[i, :] = solution_phi_x[0]
        phi_xy[i, :] = solution_phi_x[1]
        phi_yx[i, :] = solution_phi_y[0]
        phi_yy[i, :] = solution_phi_y[1]

    M_x = -Dpl * (phi_xx + nu * phi_yy)
    M_y = -Dpl * (phi_yy + nu * phi_xx)
    M_xy = -Dpl * (1 - nu) * 0.5 * (phi_xy + phi_yx)

    M_x_dim_lower = M_x + np.abs(M_xy)
    M_x_dim_upper = M_x - np.abs(M_xy)
    M_y_dim_lower = M_y + np.abs(M_xy)
    M_y_dim_upper = M_y - np.abs(M_xy)

    basis_p0 = basis.with_element(ElementTriP0())
    mx = basis_p0.project(M_x)
    my = basis_p0.project(M_y)
    mxy = basis_p0.project(M_xy)
    mx_dim_lower = basis_p0.project(M_x_dim_lower)
    my_dim_lower = basis_p0.project(M_y_dim_lower)
    mx_dim_upper = basis_p0.project(M_x_dim_upper)
    my_dim_upper = basis_p0.project(M_y_dim_upper)

    query_pts_x = np.vstack([
        np.linspace(line_par_x[0], line_par_x[1], N_query_pts),
        line_par_x[2] * np.ones(N_query_pts),
    ])
    p0_probes_x = basis_p0.probes(query_pts_x)

    query_pts_y = np.vstack([
        line_par_y[2] * np.ones(N_query_pts),
        np.linspace(line_par_y[0], line_par_y[1], N_query_pts),
    ])
    p0_probes_y = basis_p0.probes(query_pts_y)

    return {
        "D": D,
        "w": w,
        "basis_p0": basis_p0,
        "mx": mx,
        "my": my,
        "mxy": mxy,
        "mx_dim_lower": mx_dim_lower,
        "my_dim_lower": my_dim_lower,
        "mx_dim_upper": mx_dim_upper,
        "my_dim_upper": my_dim_upper,
        "query_pts_x": query_pts_x,
        "p0_probes_x": p0_probes_x,
        "query_pts_y": query_pts_y,
        "p0_probes_y": p0_probes_y,
    }


def visualize_probe_x(query_pts_x, p0_probes_x, mx, mx_dim_lower, mx_dim_upper, line_par_x):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots()
    ax.plot(query_pts_x[0], p0_probes_x @ mx_dim_lower, color='blue', label='lower')
    ax.plot(query_pts_x[0], p0_probes_x @ mx_dim_upper, color='red', label='upper')
    ax.plot(query_pts_x[0], p0_probes_x @ mx, color='gray', linestyle=':', label='m_x')
    ax.set_title(f'Moment $M_{{x, dim}}(x, y={line_par_x[2]:.2f})$ [kNm]')
    ax.set_xlabel('x [m]')
    ax.invert_yaxis()  #záporný moment nahoru
    ax.grid(True)
    plt.legend() #zobraz legentu
    return fig


def visualize_probe_y(query_pts_y, p0_probes_y, my, my_dim_lower, my_dim_upper, line_par_y):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots()
    ax.plot(query_pts_y[1], p0_probes_y @ my_dim_lower, color='blue', label='lower')
    ax.plot(query_pts_y[1], p0_probes_y @ my_dim_upper, color='red', label='upper')
    ax.plot(query_pts_y[1], p0_probes_y @ my, color='gray', linestyle=':', label='m_y')
    ax.set_title(f'Moment $M_{{y, dim}}(x={line_par_y[2]:.2f}, y)$ [kNm]')
    ax.set_xlabel('y [m]')
    ax.invert_yaxis()  #záporný moment nahoru
    ax.grid(True)
    plt.legend() #zobraz legentu
    return fig


def visualize_w(m,basis,w):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots()
    # Draw the mesh
    draw(m, ax=ax)
    # Zobrazení funkce w
    plot(basis, w, ax=ax, shading='gouraud', colorbar=True)
    ax.set_title('Tvar deformace pro jednotkovou deskovou tuhost (D = 1 kN/m)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    
    ax.set_aspect('equal')  # Nastavení stejného měřítka pro osy
    return fig


def visualize_moments(m, basis_p0, mx, my, mxy):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Vytvoříme 3 okna vedle sebe

    # Zobrazení momentu mx
    ax1 = draw(m, ax=axes[0])
    plot(basis_p0, mx, ax=ax1, shading='gouraud', colorbar=True)
    ax1.set_title('Moment $M_x$ [kNm]')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_aspect('equal')  # Nastavení stejného měřítka pro osy

    # Zobrazení momentu my
    ax2 = draw(m, ax=axes[1])
    plot(basis_p0, my, ax=ax2, shading='gouraud', colorbar=True)
    ax2.set_title('Moment $M_y$ [kNm]')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_aspect('equal')  # Nastavení stejného měřítka pro osy

    # Zobrazení momentu mxy
    ax3 = draw(m, ax=axes[2])
    plot(basis_p0, mxy, ax=ax3, shading='gouraud', colorbar=True)
    ax3.set_title('Moment $M_{xy}$ [kNm]')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_aspect('equal')  # Nastavení stejného měřítka pro osy

    plt.tight_layout()  # Pro lepší rozložení grafů
    return fig


def visualize_dim_moments_x(m, basis_p0, mx_dim_lower, mx_dim_upper):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Vytvoříme 2 okna vedle sebe

    # Zobrazení momentu mx
    ax1 = draw(m, ax=axes[0])
    plot(basis_p0, mx_dim_lower, ax=ax1, shading='gouraud', colorbar=True)
    ax1.set_title('Moment $M_{x, dim, lower}$ [kNm]')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_aspect('equal')  # Nastavení stejného měřítka pro osy

    # Zobrazení momentu my
    ax2 = draw(m, ax=axes[1])
    plot(basis_p0, mx_dim_upper, ax=ax2, shading='gouraud', colorbar=True)
    ax2.set_title('Moment $M_{x, dim, upper}$ [kNm]')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_aspect('equal')  # Nastavení stejného měřítka pro osy

    plt.tight_layout()  # Pro lepší rozložení grafů
    return fig


def visualize_dim_moments_y(m, basis_p0, my_dim_lower, my_dim_upper):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Vytvoříme 2 okna vedle sebe

    # Zobrazení momentu my
    ax1 = draw(m, ax=axes[0])
    plot(basis_p0, my_dim_lower, ax=ax1, shading='gouraud', colorbar=True)
    ax1.set_title('Moment $M_{y, dim, lower}$ [kNm]')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_aspect('equal')  # Nastavení stejného měřítka pro osy

    # Zobrazení momentu my
    ax2 = draw(m, ax=axes[1])
    plot(basis_p0, my_dim_upper, ax=ax2, shading='gouraud', colorbar=True)
    ax2.set_title('Moment $M_{y, dim, upper}$ [kNm]')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_aspect('equal')  # Nastavení stejného měřítka pro osy


    plt.tight_layout()  # Pro lepší rozložení grafů
    return fig


def visualize_mesh(m, D, basis, line_par_x, line_par_y):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots()
    # Draw the mesh
    draw(m, ax=ax)

    # 0) Vyznačení okrajových podmínek po jednotlivých hranách polygonu
    edge_styles = {
        (0, 0): {"color": "#6c757d", "linestyle": "-", "label": "Volný okraj [0,0]"},
        (1, 0): {"color": "#1f77b4", "linestyle": "-", "label": "Kloub [1,0]"},
        (0, 1): {"color": "#2ca02c", "linestyle": "-", "label": "Osa symetrie [0,1]"},
        (1, 1): {"color": "#d62728", "linestyle": "-", "label": "Vetknutí [1,1]"},
    }
    for i in range(polygon.shape[0]):
        point_a = polygon[i]
        point_b = polygon[(i + 1) % polygon.shape[0]]
        boundary_key = tuple(int(value) for value in ulozeni[i])
        style = edge_styles.get(
            boundary_key,
            {"color": "black", "linestyle": "--", "label": f"Neznámá podmínka {list(boundary_key)}"},
        )
        ax.plot(
            [point_a[0], point_b[0]],
            [point_a[1], point_b[1]],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            label=style["label"],
            zorder=3,
        )

    # Vyznačit body se DOF, které se staticky kondenzují:
    # 1) předepsaní u=0 pro kondenzaci
    valid_indices = np.intersect1d(D, basis.dofs.nodal_dofs)
    x_coords = basis.doflocs[0][valid_indices]  # Souřadnice x pro indexy v D a současně v basis.dofs.nodal_dofs
    y_coords = basis.doflocs[1][valid_indices]  # Souřadnice y pro indexy v D a současně v basis.dofs.nodal_dofs
    ax.scatter(x_coords, y_coords, color='blue', label='$w=0$')
    # 2) předepsaní u_n=0 pro kondenzaci
    valid_indices = np.intersect1d(D, basis.dofs.facet_dofs)
    x_coords = basis.doflocs[0][valid_indices]  # Souřadnice x pro indexy v D a současně v basis.dofs.facet_dofs
    y_coords = basis.doflocs[1][valid_indices]  # Souřadnice y pro indexy v D a současně v basis.dofs.facet_dofs
    ax.scatter(x_coords, y_coords, color='red', label=r'$\phi_n=0$') #"r" je důležité, aby zpětné lomítko nebylo interpretováno jako escape sekvence

    # 3) Vyznačení linií pro liniové grafy momentů
    ax.plot(
        [line_par_x[0], line_par_x[1]],
        [line_par_x[2], line_par_x[2]],
        color='green',
        linewidth=1.8,
        label='Linie grafu $M_x$',
    )
    ax.plot(
        [line_par_y[2], line_par_y[2]],
        [line_par_y[0], line_par_y[1]],
        color='green',
        linewidth=1.8,
        linestyle='--',
        label='Linie grafu $M_y$',
    )

    # Display the plot
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal', adjustable='box') #stejné měřítko os
    handles, labels = ax.get_legend_handles_labels()
    unique_entries = dict(zip(labels, handles))
    ax.legend(unique_entries.values(), unique_entries.keys())
    return fig


def save_input_assignment(input_path, input_data):
    Path(input_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'w', encoding='utf-8') as file:
        json.dump(input_data, file, ensure_ascii=False, indent=2)


def save_report_pdf(pdf_path, input_data, figures):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    Path(pdf_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        fig_text, ax_text = plt.subplots(figsize=(8.27, 11.69))
        ax_text.axis('off')
        input_json = json.dumps(input_data, ensure_ascii=False, indent=2)
        ax_text.text(0.02, 0.98, f"Kirchhoff plate report\nVygenerováno: {datetime.now().isoformat(timespec='seconds')}\n\nZadání:\n{input_json}",
                     va='top', ha='left', family='monospace', fontsize=9)
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)

        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')


def save_plot_images(output_dir, figures):
    os.makedirs(output_dir, exist_ok=True)
    for i, fig in enumerate(figures, start=1):
        fig.savefig(os.path.join(output_dir, f"plot_{i:02d}.png"), dpi=200, bbox_inches='tight')


def _build_edges_text(polygon_points, ulozeni_flags):
    lines = []
    for (x_coord, y_coord), (w_flag, phi_n_flag) in zip(polygon_points, ulozeni_flags):
        lines.append(f"{x_coord},{y_coord},{int(w_flag)},{int(phi_n_flag)}")
    return "\n".join(lines)


def _field_extrema_with_location(field_values, dof_locations):
    values = np.asarray(field_values).ravel()
    x_coords = np.asarray(dof_locations[0]).ravel()
    y_coords = np.asarray(dof_locations[1]).ravel()

    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))

    return {
        "min": {
            "value": float(values[min_idx]),
            "x": float(x_coords[min_idx]),
            "y": float(y_coords[min_idx]),
        },
        "max": {
            "value": float(values[max_idx]),
            "x": float(x_coords[max_idx]),
            "y": float(y_coords[max_idx]),
        },
    }


def _build_field_extrema_data(basis_p0, mx, my, mxy, mx_dim_lower, my_dim_lower, mx_dim_upper, my_dim_upper):
    return {
        "mx": _field_extrema_with_location(mx, basis_p0.doflocs),
        "my": _field_extrema_with_location(my, basis_p0.doflocs),
        "mxy": _field_extrema_with_location(mxy, basis_p0.doflocs),
        "mx_dim_lower": _field_extrema_with_location(mx_dim_lower, basis_p0.doflocs),
        "my_dim_lower": _field_extrema_with_location(my_dim_lower, basis_p0.doflocs),
        "mx_dim_upper": _field_extrema_with_location(mx_dim_upper, basis_p0.doflocs),
        "my_dim_upper": _field_extrema_with_location(my_dim_upper, basis_p0.doflocs),
    }


def _build_saved_input_data(input_path, basis_p0, mx, my, mxy, mx_dim_lower, my_dim_lower, mx_dim_upper, my_dim_upper):
    project_name = os.path.basename(os.path.dirname(os.path.abspath(input_path)))
    return {
        "project_name": project_name,
        "q": q,
        "lc": lc,
        "d": d,
        "E": E,
        "nu": nu,
        "n_query_pts": N_query_pts,
        "edges_text": _build_edges_text(polygon.tolist(), ulozeni.astype(int).tolist()),
        "line_par_x": line_par_x.tolist(),
        "line_par_y": line_par_y.tolist(),
        "field_extrema": _build_field_extrema_data(
            basis_p0, mx, my, mxy, mx_dim_lower, my_dim_lower, mx_dim_upper, my_dim_upper
        ),
    }


def main():
    m, basis, K, f = compute_plate_response()

    # Včasné ověření, že linie grafů leží uvnitř oblasti.
    validate_query_line_in_polygon(line_par_x, axis="x", polygon=polygon)
    validate_query_line_in_polygon(line_par_y, axis="y", polygon=polygon)

    preview = solve_plate_system(m, basis, K, f)
    fig_mesh = visualize_mesh(m, preview["D"], basis, line_par_x, line_par_y)

    figures = [
        fig_mesh,
        visualize_w(m, basis, preview["w"]),
        visualize_moments(m, preview["basis_p0"], preview["mx"], preview["my"], preview["mxy"]),
        visualize_dim_moments_x(m, preview["basis_p0"], preview["mx_dim_lower"], preview["mx_dim_upper"]),
        visualize_dim_moments_y(m, preview["basis_p0"], preview["my_dim_lower"], preview["my_dim_upper"]),
        visualize_probe_x(preview["query_pts_x"], preview["p0_probes_x"], preview["mx"], preview["mx_dim_lower"], preview["mx_dim_upper"], line_par_x),
        visualize_probe_y(preview["query_pts_y"], preview["p0_probes_y"], preview["my"], preview["my_dim_lower"], preview["my_dim_upper"], line_par_y),
    ]

    input_data = _build_saved_input_data(
        input_file,
        preview["basis_p0"],
        preview["mx"],
        preview["my"],
        preview["mxy"],
        preview["mx_dim_lower"],
        preview["my_dim_lower"],
        preview["mx_dim_upper"],
        preview["my_dim_upper"],
    )

    save_input_assignment(input_file, input_data)
    save_plot_images(plots_dir, figures)
    save_report_pdf(report_file, input_data, figures)
    print(f"Uloženo zadání: {input_file}")
    print(f"Uloženy obrázky: {plots_dir}")
    print(f"Uložen report: {report_file}")

    plt.show() #zobrazí všechny grafy najednou


def preview_mesh_main():
    m, basis, _, _ = compute_plate_response()
    visualize_mesh(m, np.array([], dtype=int), basis, line_par_x, line_par_y)
    plt.show()
    validate_query_line_in_polygon(line_par_x, axis="x", polygon=polygon)
    validate_query_line_in_polygon(line_par_y, axis="y", polygon=polygon)
    
    
    


if __name__ == "__main__":
    try:
        if os.getenv("KIRCHHOFF_PREVIEW_MESH") == "1":
            preview_mesh_main()
        else:
            main()
    except ValueError as exc:
        print(f"Chyba vstupu nebo nestability výpočtu: {exc}")
        sys.exit(1)
