import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def encontrar_mas_similares(lista1: List[Tuple[int, int]], lista2: List[Tuple[int, int]]) -> List[int]:
    indices_similares = []

    for par1 in lista1:
        distancias = [euclidean_distance(par1, par2) for par2 in lista2]
        indice_mas_cercano = np.argmin(distancias).item()
        indices_similares.append(indice_mas_cercano)

    return indices_similares


number = [[45, 38], [88, 39], [128, 39], [168, 39], [209, 38], [272, 38], [310, 38], [351, 38], [396, 39], [434, 42],
          [46, 96], [87, 96], [129, 97], [170, 98], [211, 96], [269, 99], [309, 98], [352, 97], [396, 99], [434, 99],
          [47, 152], [87, 151], [131, 153], [170, 153], [210, 153], [273, 154], [312, 154], [350, 154], [393, 154],
          [433, 152]]


list_option = {"v1": [
    [846, 148], [846, 448], [978, 146], [982, 446], [1152, 146], [1152, 446], [1326, 146], [1326, 448], [1328, 848],
    [1542, 142], [1548, 448], [1548, 848], [1718, 146], [1718, 448], [1720, 848]
], "v2": [
    [848, 198], [848, 500], [982, 198], [982, 500], [1110, 196], [1112, 498], [1112, 900], [1244, 196], [1246, 498],
    [1372, 198], [1374, 498], [1544, 194], [1544, 496], [1546, 896], [1722, 194], [1722, 496], [1722, 898], [1898, 196],
    [1898, 498], [1898, 898], [2026, 194], [2028, 496]
]}

qty_ans_by_row = {"v1": [2, 2, 2, 3, 3, 3], "v2": [2, 2, 3, 2, 2, 3, 3, 3, 2]}
list_columns = {"v1": ["file", "create_v1", "create_v2", "Graduation_plans_v1", "Interesting_scince_v1",
                       "perception_science_v1", "work_science_v1", "number_control"],
                "v2": ["file", "Create_v2", "Destroy_v2", "Language", "Parents_Graduates", "Graduation_plans_v2",
                       "Interesting_scince_v2", "perception_science_v2", "work_science_v2", "before seen",
                       "number_control"]}

df1 = pd.DataFrame(columns=list_columns["v1"])
df2 = pd.DataFrame(columns=list_columns["v2"])

size_rect = [45, 45]
image_master = {"image_gray": cv2.imread("pre0.jpg", cv2.IMREAD_GRAYSCALE), "version": "v2"}
_, image_master["image_bn"] = cv2.threshold(image_master["image_gray"], 200, 255, cv2.THRESH_BINARY_INV)
image_master["hist"] = cv2.calcHist([image_master["image_bn"]], [0], None, [256], [0, 245])

path = "."
files = os.listdir(path)
# files = ['page0.jpg']
for file in files:
    if file.endswith(".jpg") and not file.startswith("num"):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img2 = img.copy()
        _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        hist = cv2.calcHist([img], [0], None, [256], [0, 245])
        chi_cuadrado = round(cv2.compareHist(image_master["hist"], hist, cv2.HISTCMP_CHISQR), 2)
        list2col = [file]
        c = 0
        v = image_master["version"] if chi_cuadrado < 500 else ("v1" if image_master["version"] == "v2" else "v2")
        for row in qty_ans_by_row[v]:
            list_area = []
            for coord in list_option[v][c:c + row]:
                print(coord)
                h = size_rect[0]
                w = size_rect[1]
                y = coord[0] - w/2
                x = coord[1] - h/2
                roi = img[int(y):int(y + h), int(x):int(x + w)]
                list_area.append(np.sum(roi))
                # cv2.imshow("roi", roi)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            c += row

            indice_max = list_area.index(max(list_area))
            print(file, list_area, indice_max)
            # print(file, indice_max)
            list2col.append(indice_max)
        # control number ----------------------------------------------------------------
        x, y, h, w = 100, 390, 200, 550
        img2 = img2[int(y):int(y + h), int(x):int(x + w)]
        img2 = cv2.GaussianBlur(img2, (7, 7), 0)
        _, thresh = cv2.threshold(img2, 250, 255, cv2.THRESH_BINARY_INV)
        # Ajuste de la imagen
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y_coords, x_coords = np.where(thresh == 255)
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            img2 = img2[y_min:y_max + 1, x_min:x_max + 1]
        # Obtencion del centroide de cada respuesta
        _, thresh_answer = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh_answer = cv2.dilate(thresh_answer, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(thresh_answer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append([cX, cY])
        # Ordenar centroides por coordenada Y y luego X
        centroids = sorted(centroids, key=lambda q: (q[1], q[0]))
        number_control = encontrar_mas_similares(centroids, number)
        number_control = [x + 1 for x in number_control]
        number_control_string = ''.join(str(i)[-1] for i in number_control)
        list2col.append(number_control_string.zfill(3))
        if v == "v2":
            df2.loc[len(df2)] = list2col
        else:
            df1.loc[len(df1)] = list2col
        # ------------------------------------
        # cv2.imshow("Deteccion de Circulos", cv2.resize(img2, None, fx=0.5, fy=0.5))
        # cv2.imshow("roi", roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
print(df1.to_string())
print("*" * 40)
print(df2.to_string())
df1.to_csv("salida1.csv")
df2.to_csv("salida2.csv")
