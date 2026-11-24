"""What worlds are made of: everyday things, furniture and surface materials (sizes in metres, masses in kg).

Everyday things are real objects from Google Scanned Objects (Downs et al. 2022, arXiv 2204.11918, CC-BY 4.0), in
the MuJoCo conversion kevinzakka/mujoco_scanned_objects (MIT; decomposed into convex hulls). sim/worlds/assets.py
fetches them on demand into models/objects/ (not in git, ~2 MB each); without them each thing is built as a box or
cylinder of the same size, so worlds, tests and CI run without the download. Sizes (full width, depth, height) were
measured from the meshes (sim/worlds/assets.py measure); masses are typical retail weights.

Furniture is boxes (a sofa is a seat, a back and two arms) in typical sizes; materials are a colour, a procedural
pattern (sim/worlds/textures.py) and a friction coefficient from published ranges (wood on rubber ~0.6-0.8,
laminate/tile ~0.4-0.6, carpet/rug ~0.8-1.2: Engineering Toolbox friction tables; Pinch's feet are TPU).
"""

# name: (GSO model, (width, depth, height), mass, where, fallback shape, rgba)
# where: "desk" things live on desks and tables, "floor" things on the floor, "any" both
THINGS = {
    "mug":          ("ACE_Coffee_Mug_Kristen_16_oz_cup", (0.125, 0.089, 0.134), 0.35, "desk", "cylinder", (0.9, 0.9, 0.85, 1)),
    "mug_blue":     ("Cole_Hardware_Mug_Classic_Blue", (0.163, 0.123, 0.094), 0.33, "desk", "cylinder", (0.2, 0.35, 0.7, 1)),
    "mouse":        ("Razer_Abyssus_Ambidextrous_Gaming_Mouse", (0.100, 0.144, 0.036), 0.08, "desk", "box", (0.1, 0.1, 0.1, 1)),
    "keyboard":     ("Kanex_MultiSync_Wireless_Keyboard", (0.442, 0.124, 0.023), 0.45, "desk", "box", (0.85, 0.85, 0.85, 1)),
    "mouse_mat":    ("Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat", (0.271, 0.218, 0.009), 0.1, "desk", "box", (0.12, 0.12, 0.12, 1)),
    "pencil_case":  ("Big_Dot_Aqua_Pencil_Case", (0.207, 0.087, 0.094), 0.12, "desk", "box", (0.3, 0.75, 0.8, 1)),
    "crayons":      ("Crayola_Crayons_24_count", (0.074, 0.029, 0.116), 0.15, "desk", "box", (0.95, 0.8, 0.1, 1)),
    "tape":         ("3M_Vinyl_Tape_Green_1_x_36_yd", (0.109, 0.110, 0.026), 0.08, "desk", "cylinder", (0.2, 0.7, 0.3, 1)),
    "tape_purple":  ("Shurtape_Tape_Purple_CP28", (0.117, 0.117, 0.050), 0.2, "desk", "cylinder", (0.5, 0.2, 0.6, 1)),
    "clock":        ("Crosley_Alarm_Clock_Vintage_Metal", (0.109, 0.053, 0.161), 0.3, "desk", "cylinder", (0.75, 0.2, 0.15, 1)),
    "orchid_pot":   ("Down_To_Earth_Orchid_Pot_Ceramic_Red", (0.129, 0.130, 0.119), 0.6, "desk", "cylinder", (0.75, 0.15, 0.1, 1)),
    "plant_pot":    ("Ecoforms_Plant_Pot_GP9AAvocado", (0.093, 0.093, 0.086), 0.3, "desk", "cylinder", (0.5, 0.6, 0.2, 1)),
    "bowl":         ("Cole_Hardware_Bowl_Scirocco_YellowBlue", (0.111, 0.113, 0.062), 0.4, "desk", "cylinder", (0.9, 0.8, 0.3, 1)),
    "book":         ("Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book",
                     (0.139, 0.050, 0.210), 0.45, "any", "box", (0.85, 0.85, 0.75, 1)),
    "game_box":     ("Animal_Crossing_New_Leaf_Nintendo_3DS_Game", (0.137, 0.125, 0.015), 0.06, "desk", "box", (0.3, 0.6, 0.3, 1)),
    "bookend":      ("Victor_Reversible_Bookend", (0.233, 0.106, 0.235), 0.25, "desk", "box", (0.2, 0.2, 0.2, 1)),
    "mints":        ("Orbit_Bubblemint_Mini_Bottle_6_ct", (0.208, 0.070, 0.118), 0.06, "desk", "cylinder", (0.4, 0.8, 0.9, 1)),
    "tape_measure": ("SNAIL_MEASURING_TAPE", (0.062, 0.192, 0.065), 0.1, "desk", "box", (0.9, 0.7, 0.2, 1)),
    "dino":         ("Dino_3", (0.288, 0.068, 0.126), 0.1, "any", "box", (0.3, 0.6, 0.3, 1)),
    "triceratops":  ("Great_Dinos_Triceratops_Toy", (0.068, 0.046, 0.079), 0.12, "any", "box", (0.6, 0.5, 0.3, 1)),
    "elephant":     ("Elephant", (0.189, 0.208, 0.269), 0.12, "any", "box", (0.6, 0.6, 0.65, 1)),
    "dog_toy":      ("Dog", (0.230, 0.185, 0.198), 0.12, "any", "box", (0.6, 0.45, 0.3, 1)),
    "stacking_bear": ("STACKING_BEAR", (0.106, 0.100, 0.176), 0.2, "any", "cylinder", (0.9, 0.5, 0.2, 1)),
    "lego_figure":  ("Lego_Friends_Mia", (0.089, 0.198, 0.074), 0.01, "any", "box", (0.9, 0.6, 0.7, 1)),
    "jenga":        ("2_of_Jenga_Classic_Game", (0.079, 0.084, 0.292), 0.5, "desk", "box", (0.85, 0.75, 0.55, 1)),
    "ramekin":      ("BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup", (0.087, 0.087, 0.043), 0.15, "desk", "cylinder", (0.95, 0.95, 0.95, 1)),
    "shoe":         ("Reebok_VERSA_TRAIN", (0.105, 0.294, 0.117), 0.35, "floor", "box", (0.2, 0.2, 0.25, 1)),
    "sandal":       ("NAPA_VALLEY_NAVAJO_SANDAL", (0.124, 0.230, 0.080), 0.25, "floor", "box", (0.5, 0.35, 0.2, 1)),
    "backpack":     ("Jansport_School_Backpack_Blue_Streak", (0.354, 0.366, 0.436), 0.6, "floor", "box", (0.15, 0.25, 0.6, 1)),
    "teddy":        ("Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige", (0.269, 0.369, 0.120), 0.3, "floor", "box", (0.85, 0.75, 0.6, 1)),
    "basket":       ("Target_Basket_Medium", (0.267, 0.265, 0.214), 0.5, "floor", "box", (0.7, 0.55, 0.35, 1)),
    "fabric_cube":  ("Closetmaid_Premium_Fabric_Cube_Red", (0.317, 0.271, 0.287), 0.4, "floor", "box", (0.7, 0.15, 0.15, 1)),
    "fire_truck":   ("FIRE_TRUCK", (0.068, 0.123, 0.102), 0.3, "floor", "box", (0.85, 0.1, 0.1, 1)),
    "toy_train":    ("CAR_CARRIER_TRAIN", (0.037, 0.221, 0.048), 0.3, "floor", "box", (0.2, 0.5, 0.85, 1)),
    "xylophone":    ("OVAL_XYLOPHONE", (0.204, 0.281, 0.071), 0.4, "floor", "box", (0.9, 0.5, 0.2, 1)),
    "shape_clock":  ("Melissa_Doug_Shape_Sorting_Clock", (0.305, 0.301, 0.059), 0.4, "floor", "box", (0.3, 0.6, 0.9, 1)),
    "towel":        ("Cole_Hardware_Dishtowel_Blue", (0.169, 0.233, 0.024), 0.08, "floor", "box", (0.3, 0.45, 0.8, 1)),
    "duplo_box":    ("LEGO_Duplo_Build_and_Play_Box_4629", (0.439, 0.241, 0.217), 0.8, "floor", "box", (0.9, 0.85, 0.2, 1)),
}

# room furniture (boxes): name: (width, depth, height) ranges as ((lo, hi), ...), against a wall?
# the smallest surface Pinch is put on (long, short side, m): the v2.1 stance puts its feet 17.4 cm from the body
# centre and the side cliff beams see the desk 21-23 cm out, so standing in the middle of the short side needs
# ~0.5 m; 0.55 leaves it a few cm to turn in. Real desks are 50-80 cm deep (the owner's 60).
SURFACE_MIN = (0.8, 0.55)

FURNITURE = {
    "desk":      (((0.8, 1.6), (0.55, 0.8), (0.70, 0.78)), True),     # >= SURFACE_MIN: room for Pinch's stance
    "table":     (((0.8, 1.8), (0.6, 1.0), (0.70, 0.76)), False),
    "coffee":    (((0.8, 1.2), (0.55, 0.7), (0.38, 0.46)), False),
    "sofa":      (((1.6, 2.2), (0.8, 0.95), (0.80, 0.90)), True),
    "bed":       (((1.4, 1.9), (1.9, 2.1), (0.45, 0.60)), True),
    "wardrobe":  (((0.8, 1.6), (0.55, 0.62), (1.8, 2.1)), True),
    "shelf":     (((0.6, 1.2), (0.28, 0.4), (0.8, 1.9)), True),
    "chair":     (((0.42, 0.5), (0.42, 0.5), (0.85, 0.95)), False),
    "tv_stand":  (((1.0, 1.8), (0.35, 0.45), (0.40, 0.55)), True),
    "counter":   (((1.2, 2.4), (0.58, 0.62), (0.88, 0.92)), True),
    "nightstand": (((0.4, 0.5), (0.35, 0.45), (0.5, 0.6)), True),
}
ROOM_FURNITURE = {        # what each kind of room tends to have (the first ones first)
    "study":   ["desk", "chair", "shelf", "shelf"],
    "living":  ["sofa", "coffee", "tv_stand", "shelf"],
    "bedroom": ["bed", "wardrobe", "nightstand", "desk"],
    "kitchen": ["counter", "table", "chair", "chair"],
}

# surfaces: name: (pattern, base colours to pick from, friction range for TPU feet)
SURFACES = {
    "oak":       ("wood", [(0.66, 0.49, 0.32), (0.75, 0.58, 0.40), (0.55, 0.38, 0.24)], (0.6, 0.8)),
    "walnut":    ("wood", [(0.36, 0.24, 0.16), (0.42, 0.28, 0.18)], (0.6, 0.8)),
    "white_lam": ("noise", [(0.92, 0.92, 0.90), (0.85, 0.86, 0.88)], (0.45, 0.65)),
    "grey_lam":  ("noise", [(0.55, 0.56, 0.58), (0.35, 0.36, 0.38)], (0.45, 0.65)),
    "glass":     ("flat", [(0.75, 0.82, 0.85)], (0.35, 0.5)),
}
FLOORS = {
    "wood_floor": ("planks", [(0.62, 0.45, 0.30), (0.72, 0.55, 0.38), (0.48, 0.34, 0.22)], (0.5, 0.7)),
    "tile":       ("tiles", [(0.85, 0.84, 0.80), (0.70, 0.70, 0.72), (0.80, 0.72, 0.62)], (0.4, 0.6)),
    "carpet":     ("noise", [(0.45, 0.42, 0.40), (0.30, 0.35, 0.45), (0.60, 0.55, 0.48)], (0.9, 1.2)),
    "concrete":   ("noise", [(0.62, 0.62, 0.60)], (0.6, 0.8)),
}
RUGS = {
    "stripes": ("stripes", [(0.75, 0.30, 0.25), (0.25, 0.40, 0.65), (0.85, 0.75, 0.45)], (0.9, 1.3)),
    "checker": ("checker", [(0.30, 0.55, 0.45), (0.65, 0.35, 0.55)], (0.9, 1.3)),
    "plain":   ("noise", [(0.80, 0.75, 0.65), (0.35, 0.30, 0.40)], (0.9, 1.3)),
}
WALLS = [(0.92, 0.90, 0.85), (0.85, 0.88, 0.90), (0.95, 0.93, 0.88), (0.80, 0.85, 0.78), (0.90, 0.82, 0.75)]
SKIN = [(0.98, 0.84, 0.72), (0.90, 0.72, 0.58), (0.78, 0.57, 0.42), (0.60, 0.42, 0.30), (0.42, 0.28, 0.20)]
CLOTHES = [(0.2, 0.3, 0.6), (0.7, 0.2, 0.2), (0.2, 0.5, 0.3), (0.9, 0.9, 0.9), (0.15, 0.15, 0.15), (0.8, 0.6, 0.2),
           (0.5, 0.3, 0.6)]
TROUSERS = [(0.15, 0.2, 0.35), (0.2, 0.2, 0.2), (0.45, 0.4, 0.35), (0.3, 0.3, 0.4)]
