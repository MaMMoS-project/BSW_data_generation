import pathlib as pl
from collections import defaultdict
import mammos_entity as me

paths = sorted(pl.Path("data").glob("**/parameters.yaml"))

collection = defaultdict(list)
filepath = []

for file in paths:
    collection_single = me.io.entities_from_file(file)
    for k, v in vars(collection_single).items():
        collection[k].append(v)
    filepath.append(str(file))

for k, v in collection.items():
    collection[k] = me.concat_flat(*v)

me.io.entities_to_file(
    "single_grain_cube_50nm_aligned.csv",
    (
        "Micromagnetic hysteresis simulations of a 50 nm single-grain"
        " cube with randomly sampled Ms, A, and K1. Applied field and"
        " anisotropy axis aligned with the cube edge."
        " Source: https://github.com/MaMMoS-project/BSW_data_generation"
    ),
    **collection,
    filepath=filepath,
)
