"""
openmc_worker.py -- runs OpenMC gamma-ray jobs for geometry_class/gamma_openmc.py.

Runs where OpenMC is installed (on Windows: inside WSL). It does not import
PyBorehole: every job is a self-contained JSON description of one tool
station (cells, materials, sources), written by OpenMCGammaGeometry.

    python openmc_worker.py jobs.json results.json

Each result holds the pulse-height spectrum in the crystal (mean and standard
deviation per energy bin), the same for the gross count above the threshold,
and the total photon emission rate of the sources (photons/s). OpenMC scales
fixed-source tallies by the total source strength, and the strengths are
emission rates, so the tallies are already count rates (counts/s).

Cross sections: $OPENMC_CROSS_SECTIONS, or the first library found under
~/openmc-data/*/cross_sections.xml that has neutron data (OpenMC needs
neutron data for every nuclide, even in a photon-only run).
"""
import glob
import json
import os
import sys
import tempfile

import numpy as np
import openmc


def cross_sections() -> str:
    path = os.environ.get("OPENMC_CROSS_SECTIONS")
    if path:
        return path
    for path in sorted(glob.glob(os.path.expanduser("~/openmc-data/*/cross_sections.xml"))):
        with open(path) as fh:
            if 'type="neutron"' in fh.read():
                return path
    raise FileNotFoundError("No OpenMC library with neutron data under ~/openmc-data; "
                            "set OPENMC_CROSS_SECTIONS")


def build_model(job):
    mats = {}
    for key, m in job["materials"].items():
        mat = openmc.Material(name=key)
        mat.set_density("g/cm3", m["density"])
        for el, frac in m["elements"].items():
            mat.add_element(el, frac, "wo")
        mats[key] = mat

    rw, hw = job["world"]                       # cm; vacuum beyond
    surfaces = {}

    def zplane(z):
        key = ("z", round(z, 6))
        if key not in surfaces:
            bc = "vacuum" if abs(abs(z) - hw) < 1e-9 else "transmission"
            surfaces[key] = openmc.ZPlane(z0=z, boundary_type=bc)
        return surfaces[key]

    def zcyl(r):
        key = ("r", round(r, 6))
        if key not in surfaces:
            bc = "vacuum" if abs(r - rw) < 1e-9 else "transmission"
            surfaces[key] = openmc.ZCylinder(r=r, boundary_type=bc)
        return surfaces[key]

    def tube(rin, rout, zlo, zhi):
        region = +zplane(zlo) & -zplane(zhi) & -zcyl(rout)
        return region & +zcyl(rin) if rin > 0 else region

    cells = [openmc.Cell(fill=mats[m], region=tube(*c)) for *c, m in job["cells"]]
    rc, zlo, zhi = job["crystal"]
    crystal = openmc.Cell(name="crystal", fill=mats["nai"], region=tube(0.0, rc, zlo, zhi))
    geometry = openmc.Geometry(cells + [crystal])

    sources, total = [], 0.0
    for rin, rout, slo, shi, strength, energies, probs in job["sources"]:
        space = openmc.stats.CylindricalIndependent(
            openmc.stats.PowerLaw(rin, rout, 1), openmc.stats.Uniform(0.0, 2 * np.pi),
            openmc.stats.Uniform(slo, shi))
        sources.append(openmc.IndependentSource(
            space=space, angle=openmc.stats.Isotropic(),
            energy=openmc.stats.Discrete(energies, probs), strength=strength, particle="photon"))
        total += strength

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.photon_transport = True
    settings.electron_treatment = "ttb"
    settings.batches = job["batches"]
    settings.particles = max(1, job["particles"] // job["batches"])
    settings.seed = job["seed"]
    settings.source = sources
    settings.output = {"tallies": False}

    cell_filter = openmc.CellFilter([crystal])
    spec = openmc.Tally(name="spectrum")
    spec.filters = [cell_filter, openmc.EnergyFilter(job["edges"])]
    spec.scores = ["pulse-height"]
    gross = openmc.Tally(name="gross")
    gross.filters = [cell_filter, openmc.EnergyFilter([job["threshold"], job["edges"][-1]])]
    gross.scores = ["pulse-height"]
    model = openmc.Model(geometry, openmc.Materials(list(mats.values())), settings,
                         openmc.Tallies([spec, gross]))
    return model, total


def run_job(job):
    model, total = build_model(job)
    with tempfile.TemporaryDirectory(prefix="openmc_gr_") as tmp:
        sp = model.run(cwd=tmp, threads=job["threads"], output=False,
                       openmc_exec=os.path.join(sys.prefix, "bin", "openmc"))
        with openmc.StatePoint(sp) as f:
            s, g = f.get_tally(name="spectrum"), f.get_tally(name="gross")
            return {"depth": job["depth"], "group": job["group"], "total_strength": total,
                    "mean": s.mean.ravel().tolist(), "std": s.std_dev.ravel().tolist(),
                    "gross_mean": float(g.mean.ravel()[0]), "gross_std": float(g.std_dev.ravel()[0])}


def main(jobs_file, results_file):
    openmc.config["cross_sections"] = cross_sections()
    with open(jobs_file) as fh:
        jobs = json.load(fh)
    results = []
    for job in jobs:
        results.append(run_job(job))
        with open(results_file, "w") as fh:   # written after every job: partial results survive
            json.dump(results, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
