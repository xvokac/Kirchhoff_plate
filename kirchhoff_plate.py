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


"""
## GENEROVÁNÍ SÍTĚ V GMSH #################################################################
"""

gmsh.initialize()

#New model
gmsh.model.add("t2")

#Nodes
# Přidání bodů z polygonu
point_ids = []
for i, (x, y) in enumerate(polygon):
    point_id = gmsh.model.geo.addPoint(x, y, 0, lc, i+1)
    point_ids.append(point_id)

# lines
# Přidání úseček mezi body
line_ids = []
for i in range(len(point_ids)):
    start_point = point_ids[i]
    end_point = point_ids[(i+1) % len(point_ids)]  # Cykluje zpět k prvnímu bodu
    line_id = gmsh.model.geo.addLine(start_point, end_point)
    line_ids.append(line_id)


#CurveLoop
gmsh.model.geo.addCurveLoop(line_ids, 1)

#PlaneSurface
plane_surface_id = gmsh.model.geo.addPlaneSurface([1], 1)

#synchronize
gmsh.model.geo.synchronize()
 
# Přiřazení tagů pro linie do fyzikálního seskupení v cyklu
#(pouze s dimenzí 1 pro linie)
for i, line_id in enumerate(line_ids, start=1):
    line_tag = gmsh.model.addPhysicalGroup(1, [line_id])
    gmsh.model.setPhysicalName(1, line_tag, f"boundary_line{i}")

# Physical group for the surface (2D)
surface_tag = gmsh.model.addPhysicalGroup(2, [plane_surface_id])
gmsh.model.setPhysicalName(2, surface_tag, "Surface")

#generate
gmsh.model.mesh.generate(2)

# Nastavení formátu MSH na verzi 2
gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)

#Write mesh
gmsh.write("deska.msh")

""" pro vizualizaci sítě v gmsh je možné pouřít následující kód - není nutné,
dále se vizualizuje včetně stupňů volnosti určených ke sttické kondenzaci.

#Visualize
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
    
"""

gmsh.finalize()

"""
## ZPRACOVÁNÍ SCIKIT-FEM ######################################################################
"""

# Načtení Gmsh sítě z .msh souboru
msh = meshio.read("deska.msh")

# Převod na formát scikit-fem
points = msh.points[:, :2]  # Only use the 2D points (x, y)
cells = msh.cells_dict["triangle"]

# Vytvoření MeshTri sítě
m = MeshTri(points.T, cells.T)
basis = Basis(m, ElementTriMorley())


# Elasticita desky
def C(T):
    return E / (1 + nu) * (T + nu / (1 - nu) * eye(trace(T), 2))

# Bilineární forma
@BilinearForm
def bilinf(u, v, _):
    return d ** 3 / 12.0 * ddot(C(dd(u)), dd(v))

# Zatížení
@LinearForm
def load(v, _):
    return q * v


# Sestavení soustavy
K = bilinf.assemble(basis)
f = load.assemble(basis)


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

# Ruční přiřazení okrajových podmínek
# Kontrola umístění bodů s DOF na hranici s okrajovou podmínkou
conden = np.zeros(basis.doflocs[0].shape)
# 1) Okrajové podmínky pro pruhyb
# ozn. stupnu volnosti ve basis.dofs.nodal_dofs
for i in range(basis.dofs.nodal_dofs.shape[1]): 
    point = [basis.doflocs[0][basis.dofs.nodal_dofs[0][i]],
             basis.doflocs[1][basis.dofs.nodal_dofs[0][i]] ] #bod s DOF nodal
    for j in range(polygon.shape[0]):     #pro kazdou hranici
        if ulozeni[j][0]:    #je li zabraneno pruhybu
            if j==(polygon.shape[0]-1): #vyber koncove bodu strany polynomu
                A=[polygon[j][0], polygon[j][1]]
                B=[polygon[0][0], polygon[0][1]]
            else:
                A=[polygon[j][0], polygon[j][1]]
                B=[polygon[j+1][0], polygon[j+1][1]]
            if is_point_on_segment(point, A, B, tol=1e-9):
                conden[basis.dofs.nodal_dofs[0][i]] = 1
# 2) Okrajové podmínky pro natoceni
# ozn. stupnu volnosti ve basis.dofs.facet_dofs
for i in range(basis.dofs.facet_dofs.shape[1]): 
    point = [basis.doflocs[0][basis.dofs.facet_dofs[0][i]],
             basis.doflocs[1][basis.dofs.facet_dofs[0][i]] ] #bod s DOF nodal
    for j in range(polygon.shape[0]):     #pro kazdou hranici            
        if ulozeni[j][1]:    #je li zabraneno natoceni
            if j==(polygon.shape[0]-1): #vyber koncove bodu strany polynomu
                A=[polygon[j][0], polygon[j][1]]
                B=[polygon[0][0], polygon[0][1]]
            else:
                A=[polygon[j][0], polygon[j][1]]
                B=[polygon[j+1][0], polygon[j+1][1]]
            if is_point_on_segment(point, A, B, tol=1e-9):
                conden[basis.dofs.facet_dofs[0][i]] = 1
##Převedení na integer
conden = conden.astype(int) #hodnota je 1 pro staticky kondenzovane pozice v K a f
#D - jaké stupně volnosti (indexi) se budou staticky kondenzovat
D = np.where(conden == 1)[0]

# řešení soustavy rovnic se staticky kondenzovanou maticí K a vektorem f
w = solve(*condense(K, f, D=D))

# Výpočet odvozených parametrů pro výpočet dimenzačních momentů
# Výpočet gradientu (natočení průřezu)
interp_grad = basis.interpolate(w).grad #určí gradenty v integračních bodech
# interp_grad[0] je ∂w/∂x, interp_grad[1] je ∂w/∂y - po prvku je to lineární funkce souřadnic x a y
phi_x = interp_grad[0]  # Natočení ve směru osy x 
phi_y = interp_grad[1]  # Natočení ve směru osy y
# Souřadnice integracnich bodu
int_points_loc, int_weights  = basis.quadrature #lokální sořadnice referenčního prvku a jeho váhy
int_points_gl = basis.mapping.F(int_points_loc) #mapování na síť prvků - globální souřadnice
# Výpočet grad phi_x a grad phi_y...
"""
Jsou dány phi_x a phi_y v integračních bodech. Na prvku jsou to lineární funce.
Pokud chci gradient těchto funkcí, najdu ze 3 bodů řešením lineární sosutavy předpis
lineární funkce phi_x(x,y) = x * phi_xx + y * phi_xy + 1 * konst1 (dosadit  pro 3 body z 6 možných)
a rovněž        phi_y(x,y) = x * phi_yx + y * phi_yy + 1 * konst2 (dosadit  pro 3 body)
Řešením těchto dvou soustav rovnic jsou koeficienty phi_xx, phi_yy, phi_xy phi_yx, které jsou na prvku konstantní.
Tyto hodnoty lze potom předepsat na integrační body prvky jako další pole a dále s němi pracovat
pro stanoveni ohybovych momentu.
"""
phi_xx = np.zeros_like(phi_x) #nulove matice
phi_yy = np.zeros_like(phi_x)
phi_xy = np.zeros_like(phi_x)
phi_yx = np.zeros_like(phi_x)
for i in range(int_points_gl.shape[1]):
    A = np.array([[int_points_gl[0][i][0], int_points_gl[1][i][0], 1],
                 [int_points_gl[0][i][1], int_points_gl[1][i][1], 1],
                 [int_points_gl[0][i][2], int_points_gl[1][i][2], 1],
                 ])# 3 body (souřadnice x, y) na jednom prvku do soustavy rovnic
    b_phi_x = np.array([ [ phi_x[i][0] ],
                        [ phi_x[i][1] ],
                        [ phi_x[i][2] ],
                        ])# hodnoty phi_x v integračních bodech - prava strana rovnice
    b_phi_y = np.array([ [ phi_y[i][0] ],
                        [ phi_y[i][1] ],
                        [ phi_y[i][2] ],
                        ])# hodnoty phi_y v integračních bodech - prava strana rovnice
    # Řešení soustav pro phi_x a phi_y
    solution_phi_x = np.linalg.solve(A, b_phi_x)  # [a_x, b_x, c_x]
    solution_phi_y = np.linalg.solve(A, b_phi_y)  # [a_y, b_y, c_y]
    # Gradienty:
    phi_xx[i,:] = solution_phi_x[0]  # a_x
    phi_xy[i,:]  = solution_phi_x[1]  # b_x
    phi_yx[i,:]  = solution_phi_y[0]  # a_y
    phi_yy[i,:]  = solution_phi_y[1]  # b_y


# ohybove momenty
Dpl = E * d ** 3/(12 * (1 - nu ** 2)) # Deskovou tuhost D tady značím Dpl (proměnná D ve scikit-fem jsou stupně volnosti ke statické kondenzaci)
M_x = -Dpl * (phi_xx + nu * phi_yy)  
M_y = -Dpl * (phi_yy + nu * phi_xx)  
M_xy = -Dpl * (1-nu) * 0.5*(phi_xy + phi_yx)  

# dimenzační momenty
M_x_dim_lower = M_x + np.abs(M_xy)
M_x_dim_upper = M_x - np.abs(M_xy)
M_y_dim_lower = M_y + np.abs(M_xy)
M_y_dim_upper = M_y - np.abs(M_xy)

# Zavedeni prvků P0 - momenty na prvcích - konstanty po prvku
basis_p0 = basis.with_element(ElementTriP0())
mx = basis_p0.project(M_x)
my = basis_p0.project(M_y)
mxy = basis_p0.project(M_xy)
mx_dim_lower = basis_p0.project(M_x_dim_lower)
my_dim_lower = basis_p0.project(M_y_dim_lower)
mx_dim_upper = basis_p0.project(M_x_dim_upper)
my_dim_upper = basis_p0.project(M_y_dim_upper)

# Příprava pro zobrazení grafů podél linií
validate_query_line_in_polygon(line_par_x, axis="x", polygon=polygon)
validate_query_line_in_polygon(line_par_y, axis="y", polygon=polygon)

query_pts_x = np.vstack([
    np.linspace(line_par_x[0],line_par_x[1],N_query_pts),  # x[0] coordinate values - proměnné x
    line_par_x[2]*np.ones(N_query_pts),  # x[1] coordinate values - konstantní y
])
p0_probes_x = basis_p0.probes(query_pts_x)

query_pts_y = np.vstack([
    line_par_y[2]*np.ones(N_query_pts),  # x[0] coordinate values - konstantní x
    np.linspace(line_par_y[0],line_par_y[1],N_query_pts),  # x[1] coordinate values - proměnné y
])
p0_probes_y = basis_p0.probes(query_pts_y)


def visualize_probe_x(query_pts_x, p0_probes_x,mx_dim_lower, mx_dim_upper, line_par_x):
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


def visualize_probe_y(query_pts_y, p0_probes_y,my_dim_lower, my_dim_upper, line_par_y):
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
    ax.set_title('Shape of deflection $w(x,y)$')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    
    ax.set_aspect('equal')  # Nastavení stejného měřítka pro osy
    return fig


def visualize_moments(mx, my, mxy):
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


def visualize_dim_moments_x(mx_dim_lower, mx_dim_upper):
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


def visualize_dim_moments_y(my_dim_lower, my_dim_upper):
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


def visualize_mesh(m, D, basis):
    from skfem.visuals.matplotlib import draw, plot
    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots()
    # Draw the mesh
    draw(m, ax=ax)

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
    # Display the plot
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal', adjustable='box') #stejné měřítko os
    plt.legend() #zobraz legentu
    return fig


def save_input_assignment(input_path, input_data):
    with open(input_path, 'w', encoding='utf-8') as file:
        json.dump(input_data, file, ensure_ascii=False, indent=2)


def save_report_pdf(pdf_path, input_data, figures):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    figures = [
        visualize_mesh(m, D, basis),
        visualize_w(m,basis,w),
        visualize_moments(mx,my,mxy),
        visualize_dim_moments_x(mx_dim_lower, mx_dim_upper),
        visualize_dim_moments_y(my_dim_lower, my_dim_upper),
        visualize_probe_x(query_pts_x, p0_probes_x,mx_dim_lower, mx_dim_upper, line_par_x),
        visualize_probe_y(query_pts_y, p0_probes_y,my_dim_lower, my_dim_upper, line_par_y),
    ]

    input_data = {
        "q": q,
        "polygon": polygon.tolist(),
        "ulozeni": ulozeni.astype(int).tolist(),
        "lc": lc,
        "d": d,
        "E": E,
        "nu": nu,
        "line_par_x": line_par_x.tolist(),
        "line_par_y": line_par_y.tolist(),
        "N_query_pts": N_query_pts,
    }

    save_input_assignment(input_file, input_data)
    save_plot_images(plots_dir, figures)
    save_report_pdf(report_file, input_data, figures)
    print(f"Uloženo zadání: {input_file}")
    print(f"Uloženy obrázky: {plots_dir}")
    print(f"Uložen report: {report_file}")

    plt.show() #zobrazí všechny grafy najednou
    
