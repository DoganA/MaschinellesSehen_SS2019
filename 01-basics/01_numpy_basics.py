import numpy as np
# A1. Numpy and Linear Algebra

print("a) (*) Erzeugen Sie einen Vektoramit Nullen der Länge 10 (10 Elemente) und setzen den Wertdes 5. Elementes auf eine 1.")
vec_a = np.zeros(10)
vec_a[4] = 1
print(vec_a)
#############################################################################################################
print()
print("b) (*) Erzeugen Sie einen Vektor mit Ganzahl-Werten von 10 bis 49 (geht in einer Zeile).")
vec_b = np.arange(10, 50)
print(vec_b)

#############################################################################################################
print()
print("f) (*) Drehen Sie die Werte des Vektors aus b) oder a) um (geht in einer Zeile).")
print(vec_a[::-1]);
print(vec_b[::-1]);

#############################################################################################################
print()
print("g) (*) Summieren Sie alle Werte in einem Array (geht in einer Zeile).")

vec_g = np.arange(1, 6)
print('sum of', vec_g, 'is', vec_g.sum())

#############################################################################################################
print()
print("h) (*) Erzeugen Sie eine 4x4 Matrix mit den Werte 0 (links oben) bis 15 (rechts unten) (geht ineiner Zeile).")
print(np.arange(0, 16).reshape(4, 4))

#############################################################################################################
print()
print("i) (*) Erzeugen Sie eine 5x3 Matrix mit Zufallswerteintegers zwischen 0-100 (geht in einer Zeile).")
mat_i = np.random.randint(0, 100, size=(5, 3))
print('Zufallsmatrix: mat_i\n')
print(mat_i)

#############################################################################################################
print()
print("j) (*) Multiplizieren Sie eine 4x3 Matrix mit einer 3x2 Matrix (geht zwar in einer Zeile, aber benutzenSie lieber Hilfsvariablen und drei Zeilen).")

mat4x3 = np.arange(0, 12).reshape((4, 3))
mat3x2 = np.arange(0, 6).reshape((3, 2))
print('4x3 Matrix:\n', mat4x3)
print('\n3x2 Matrix:\n', mat3x2)
print('\n Ergebnis der Multiplikation: ', np.dot(mat4x3, mat3x2))

#############################################################################################################
print()
print("k) (*) Erzeugen Sie eine 5x5 Matrix und geben Sie jeweils die geraden und die ungeraden Zeile aus(geht jeweils in einer Zeile).")

mat_k = np.arange(0, 25).reshape((5, 5))
print(mat_k)
print('\nGerade Zeilen\n')
print(mat_k[0::2])
print('\nUngerade Zeilen\n')
print(mat_k[1::2])

#############################################################################################################
print()
print(
    "l) (**) Erzeuge eine 5x5 Matrix mit Zufallswerteintegers zwischen 0-100 und finde deren Maximumund Minimum und normalisieren Sie die Werte (sodass alle Werte zwischen 0 und 1 liegen - einWert wird 1 (max) sein und einer 0 (min)).")

mat_l = np.random.randint(0, 100, size=(5, 5))
print('Zufallsmatrix:\n')
print(mat_l)
print('\nmin:', mat_l.min(), 'max:', mat_l.max())
print("\nNormalisiert:\n")
mat_l_norm = (mat_l - mat_l.min()) / mat_l.max()
print(mat_l_norm)
print('\nmin:', mat_l_norm.min(), 'max:', mat_l_norm.max())

#############################################################################################################
print()
print(
    "n) (**) Erzeugen Sie eine Matrix M der Größe 4x3 und einen Vektor v mit Länge 3. MultiplizierenSie jeden Spalteneintrag aus mit der kompletten Spalte aus M. Nutzen Sie dafür Broadcasting.")

M = np.arange(0, 12).reshape(4, 3)
v = np.arange(2, 5)
print(M, '\n------\n')
print(v, '\n------\n')
print(M * v, '\n------\n');

#############################################################################################################
print()
print(
    "o) (***) Erzeugen Sie einen Zufallsmatrix der Größe 6x2, die Sie als Kartesische Koordinaten inter-pretieren können ([[x0, y0],[x1, y1],[x2, y2]]). Konvertieren Sie diese in Polarkoordinaten.")


def cart2pol(x, y):
    # https://www.w3resource.com/python-exercises/numpy/python-numpy-random-exercise-14.php
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)  # arctan2 automatically chooses the right quadrant
    return np.array([r, theta])


mat_o_cart = np.random.randint(0, 255, size=(6, 2))
mat_o_pol = np.empty_like(mat_o_cart, dtype=np.float32)

for n, (x, y) in enumerate(mat_o_cart[:]):
    mat_o_pol[n] = cart2pol(x, y)

print(mat_o_cart)
print()
print(mat_o_pol)

#############################################################################################################
print()
print(
    "p) (***) Implementieren Sie zwei Funktionen, die das Skalarprodukt und die Vektorlängefür Vek-toren beliebiger Längeberechnen. Nutzen Sie dabei NICHT die gegebenen Funktionen vonNumPy. Testen Sie Ihre Funktionen mit den gegebenen Vektoren:")

# TODO: ist es relevant dass das Spaltenvektoren sind?

v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([-1, 9, 5, 3, 1])


def dotproduct(vec1, vec2):
    product = 0.
    if vec1.shape != vec2.shape:
        return None
    for x1, x2 in zip(vec1, vec2):
        product += x1 * x2
    return product


def magnitude(vec):
    return np.sqrt(dotproduct(vec, vec))


print('Skalarprodukt von', v1, v2, '=', dotproduct(v1, v2))
print('Länge von', v1, magnitude(v1))
print('Länge von', v2, magnitude(v2))

#############################################################################################################
print()
print(
    "q) (***) Berechnen Sie(vT0v1)M v0unter der Nutzung von NumPy Operationen.Achten Sie dar-auf, dass hierv0, v1Spaltenvektoren gegeben sind.vT0ist also ein Zeilenvektor.")

# Should result in [3,9,15,2]T

M_m = np.asmatrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 2, 2]]))
v0_m = np.array([[1], [1], [0]])
v1_m = np.array([[-1], [2], [5]])

# (np.dot(v0_m.T, v1_m)) leads to a single element two-dimensional array.
#  Broadcasting does not help there. It needs to get casted to a scalar because mp.dot or other matrix
#  multiplication featues do not recognize an single element matrix as a scalar.
print(np.asscalar(np.dot(v0_m.T, v1_m)) * M_m * v0_m)