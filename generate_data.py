import mammos_entity as me
import mammos_units as u
from mammos_mumag.hysteresis import read_result
from mammos_mumag.materials import Materials
from mammos_mumag.mesh import Mesh
from mammos_mumag.parameters import Parameters
from mammos_mumag.simulation import Simulation
import mammos_analysis

import os
import glob
import numpy as np

u.set_enabled_equivalencies(u.magnetic_flux_field())

job = os.getenv("SLURM_ARRAY_JOB_ID") or os.getenv("SLURM_JOB_ID", "local")
task = os.getenv("SLURM_ARRAY_TASK_ID", "0")
proc = os.getenv("SLURM_PROCID") or os.getenv("SLURM_LOCALID") or "0"

current_directory = os.getcwd()
working_path = f"{current_directory}/data/{job}_{task}_{proc}"
os.mkdir(working_path)

# Material parameters
rand_gen = np.random.default_rng()

while True:
    Ms = me.Ms((rand_gen.uniform(0.1, 5) * u.T).to("A/m"))  # 0.1 - 5 T
    A = me.A(rand_gen.uniform(1e-13, 1e-11), unit="J/m")  #  1e-13 - 1e-11 J/m
    K1 = me.Ku(rand_gen.uniform(10e3, 10e6), unit="J/m3")  # 10 - 100000 kA/m

    l_A = np.sqrt(2*A.q/(u.constants.mu0*Ms.q**2)).to(u.nm)
    l_K = np.sqrt(A.q/K1.q).to(u.nm)

    print(f"Exchange length l_A: {l_A}, Domain wall thickness l_K: {l_K}")

    threshold = 1 * u.nm
    if l_A >= threshold and l_K >= threshold:
        break

D = me.Entity("DemagnetizingFactor", 1 / 3)  # Demag factor

mesh = Mesh("cube50_singlegrain_msize1")

mat = Materials(
    domains=[
        {  # cube
            "theta": 0.0,
            "phi": 0.0,
            "Ms": Ms,
            "A": A,
            "K1": K1,
        },
        {},  # non-magnetic material
        {},  # Shell
    ],
)

par = Parameters(
    size=1.0e-9,
    scale=0,
    m_vect=[0, 0, 1],
    h_start=(1 * u.T).to("A/m"),
    h_final=(-10 * u.T).to("A/m"),
    h_step=(-5 * u.mT).to("A/m"),
    h_vect=[0, 0.01745, 0.99984], 
    m_step=(2 * Ms.q).to("A/m"),
    m_final=(-0.5 * Ms.q).to("A/m"),
    tol_fun=1e-10,
    tol_h_mag_factor=1,
    precond_iter=10,
)


sim = Simulation(
    mesh=mesh,
    materials=mat,
    parameters=par,
)

sim.run_loop(outdir=working_path, name="cube")

results = read_result(outdir=working_path, name="cube")

extrinsic_properties = mammos_analysis.hysteresis.extrinsic_properties(
    H=results.H,
    M=results.M,
    demagnetization_coefficient=D.value,
)

me.io.entities_to_file(
    os.path.join(working_path, "parameters.yaml"),
    "Results for a single grain cube of 50 nm with random intrinsic material parameters.",
    Ms=Ms,
    A=A,
    K=K1,
    D=D,
    Hc=extrinsic_properties.Hc,
    Mr=extrinsic_properties.Mr,
    BHmax=extrinsic_properties.BHmax,
)

# Remove all files apart from parameter file, csv, and info file
pattern = os.path.join(working_path, 'cube*')
files_to_remove = glob.glob(pattern)

for file_path in files_to_remove:
    if not file_path.endswith(".csv"):
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")