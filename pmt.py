import xml.dom.minidom as dom
import numpy as np


def pmt_pos():
    pos = []
    for i in range(30):
        pos.append([])
    tree = dom.parse("PMT_Position.xml")
    vols = tree.documentElement
    vols = vols.getElementsByTagName("physvol")
    for vol in vols:
        name = vol.getAttribute("name")
        index = int(name[4:])
        position = vol.getElementsByTagName("position")[0]
        x = float(position.getAttribute("x"))
        y = float(position.getAttribute("y"))
        z = float(position.getAttribute("z"))
        pos[index] = [x, y, z]
    return np.array(pos)


if __name__ == "__main__":
    print(pmt_pos())
