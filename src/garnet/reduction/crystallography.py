from mantid.geometry import PointGroupFactory, SpaceGroupFactory

import gemmi

point_groups = PointGroupFactory.getAllPointGroupSymbols()
space_groups = SpaceGroupFactory.getAllSpaceGroupSymbols()

point_crystal = {}
point_lattice = {}
point_laue = {}

for point_group in point_groups:
    pg = PointGroupFactory.createPointGroup(point_group)
    pg_name = pg.getHMSymbol().replace(" ", "")
    point_crystal[pg_name] = pg.getCrystalSystem().name
    point_lattice[pg_name] = pg.getLatticeSystem().name
    point_laue[pg_name] = pg.getLauePointGroupSymbol()

space_point = {}
space_number = {}

for space_group in space_groups:
    sg = SpaceGroupFactory.createSpaceGroup(space_group)
    sg_name = sg.getHMSymbol().replace(" ", "")
    space_point[sg_name] = sg.getPointGroup().getHMSymbol().replace(" ", "")
    space_number[sg_name] = sg.getNumber()

mantid_to_gemmi = {}
gemmi_misses = []
mantid_misses = []

keys = list(space_point.keys())
for sg in gemmi.spacegroup_table():
    name = sg.short_name()

    if sg.ext == "H":
        name = name.replace("H", "R")
    elif sg.ext == "R":
        name = name + ":r"
    elif sg.ext == "2":
        name = name + ":2"
    key = name in keys

    # Acta Cryst. (1992). A48, 727-732
    if not key:
        if sg.number in [39, 41, 64, 67, 68]:
            if name.startswith("A"):
                name = name[:1] + "e" + name[1 + 1 :]
            elif name.startswith("B"):
                name = name[:2] + "e" + name[2 + 1 :]
            elif name.startswith("C"):
                name = name[:3] + "e" + name[3 + 1 :]
    key = name in keys
    if not key:
        if sg.number in range(3, 15 + 1):
            name = name[:1] + "1" + name[1:] + "1"
    key = name in keys

    if key:
        mantid_to_gemmi[name] = sg.hm
    else:
        gemmi_misses.append([name, sg.ext, sg.number])

for key in keys:
    if mantid_to_gemmi.get(key) is None:
        mantid_misses.append(key)

for key in mantid_misses:
    print(key)

space_groups = {sg.replace(" ", ""): sg for sg in space_groups}
point_groups = {pg.replace(" ", ""): pg for pg in point_groups}
