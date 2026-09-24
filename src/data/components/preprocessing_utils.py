"""CSV-to-graph preprocessing used by the crystal training datamodule."""

from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


CRYSTAL_NN = local_env.CrystalNN(
    distance_cutoffs=None,
    x_diff_weight=-1,
    porous_adjustment=False,
)


def build_crystal(crystal_str: str, niggli: bool = True, primitive: bool = False):
    """Parse a CIF and return a canonical pymatgen Structure."""
    crystal = Structure.from_str(crystal_str, fmt="cif")
    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()
    return Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )


def build_crystal_graph(crystal: Structure, graph_method: str = "crystalnn"):
    """Convert a pymatgen Structure to the arrays consumed by CrystalDataset."""
    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, CRYSTAL_NN)
        except Exception:
            fallback = local_env.CrystalNN(
                distance_cutoffs=None,
                x_diff_weight=-1,
                porous_adjustment=False,
                search_cutoff=10,
            )
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, fallback)
    elif graph_method == "none":
        crystal_graph = None
    else:
        raise ValueError(f"Unsupported graph method: {graph_method}")

    edge_indices = []
    to_jimages = []
    if crystal_graph is not None:
        for left, right, image in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.extend(([right, left], [left, right]))
            to_jimages.extend((image, tuple(-value for value in image)))

    lengths = np.asarray(crystal.lattice.abc)
    angles = np.asarray(crystal.lattice.angles)
    atom_types = np.asarray(crystal.atomic_numbers)
    return {
        "atom_types": atom_types,
        "frac_coords": np.asarray(crystal.frac_coords),
        "cell": np.asarray(crystal.lattice.matrix),
        "lattices": np.concatenate((lengths, angles)),
        "lengths": lengths,
        "angles": angles,
        "edge_indices": np.asarray(edge_indices),
        "to_jimages": np.asarray(to_jimages),
        "num_atoms": int(atom_types.shape[0]),
    }


def process_one(
    row,
    niggli: bool,
    primitive: bool,
    graph_method: str,
    prop_list,
    use_space_group: bool = False,
    tol: float = 0.01,
):
    crystal_str = row["cif"]
    crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)
    result = {
        "mp_id": row.get("material_id", str(row.name)),
        "cif": crystal_str,
        "graph_arrays": build_crystal_graph(crystal, graph_method),
        "spacegroup": (
            int(SpacegroupAnalyzer(crystal, symprec=tol).get_space_group_number())
            if use_space_group
            else 1
        ),
    }
    result.update({key: row[key] for key in prop_list if key in row.index})
    return result


def preprocess(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    """Preprocess a CSV split while preserving row order."""
    frame = pd.read_csv(input_file)
    worker = partial(
        process_one,
        niggli=niggli,
        primitive=primitive,
        graph_method=graph_method,
        prop_list=prop_list,
        use_space_group=use_space_group,
        tol=tol,
    )
    rows = [frame.iloc[index] for index in range(len(frame))]
    if int(num_workers) <= 1:
        return [worker(row) for row in rows]
    return Parallel(n_jobs=int(num_workers), prefer="processes")(
        delayed(worker)(row) for row in rows
    )
