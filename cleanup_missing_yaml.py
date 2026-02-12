import mammos_entity as me
import mammos_units as u
from mammos_mumag.hysteresis import read_result
import mammos_analysis
import os
import glob
import numpy as np
from tqdm import tqdm

u.set_enabled_equivalencies(u.magnetic_flux_field())
D = me.Entity("DemagnetizingFactor", 1 / 3)

def parse_krn_file(krn_path):
    with open(krn_path, 'r') as f:
        values = [float(x) for x in f.readline().split()]
    Ms = me.Ms((values[4] * u.T).to("A/m"))
    A = me.A(values[5], unit="J/m")
    K1 = me.Ku(values[2], unit="J/m3")
    return Ms, A, K1

data_dirs = [d for d in glob.glob("data/*") if os.path.isdir(d)]

for data_dir in tqdm(sorted(data_dirs), desc="Processing directories"):
    yaml_path = os.path.join(data_dir, "parameters.yaml")
    if os.path.exists(yaml_path):
        continue
    
    krn_files = glob.glob(os.path.join(data_dir, "*.krn"))
    if not krn_files:
        continue
    
    try:
        Ms, A, K1 = parse_krn_file(krn_files[0])
        
        try:
            results = read_result(outdir=data_dir, name="cube")
            extrinsic_properties = mammos_analysis.hysteresis.extrinsic_properties(
                H=results.H, M=results.M, demagnetization_coefficient=D.value
            )
            Hc, Mr, BHmax = extrinsic_properties.Hc, extrinsic_properties.Mr, extrinsic_properties.BHmax
        except:
            Hc, Mr, BHmax = me.Hc(np.nan), me.Mr(np.nan), me.BHmax(np.nan)
        
        me.io.entities_to_file(
            yaml_path,
            "Results for a single grain cube of 50 nm with random intrinsic material parameters.",
            Ms=Ms, A=A, K=K1, D=D, Hc=Hc, Mr=Mr, BHmax=BHmax
        )
    except:
        pass
