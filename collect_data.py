import pathlib as pl
import mammos_entity as me


paths = sorted(pl.Path("data").glob("**/parameters.yaml"))

collection = {"filepath": []}

for file in paths:
    collection_single = me.io.entities_from_file(file)    
    for k, v in collection_single.items():        
        if k not in collection:
            collection[k] = v
        else:
            collection[k] = me.concat_flat(collection[k], v)
    collection["filepath"].append(str(file))
        
me.io.entities_to_file("full.csv", **collection)