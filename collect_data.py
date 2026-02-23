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
    "full.csv",
    "Single grain cube of 50 nm with random intrinsic material parameters. Generated based on the code in https://github.com/MaMMoS-project/BSW_data_generation",
    **collection,
    filepath=filepath
)
