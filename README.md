# 3CERATOPS

**3D Coordination Environment Reconstruction from Abstract TOPology to Structure**

3CERATOPS converts [T-REX](https://doi.org/10.1039/D6DD00129G) strings into
three-dimensional structures of transition-metal complexes. Starting from an
abstract, geometry-aware description of a coordination complex, it reconstructs
the metal coordination sphere, embeds the ligands, samples multiple conformers,
and relaxes the resulting structures with xTB or UMA.

The installable distribution and command-line program are named `3ceratops`.
The Python import package is `trex_geo`.

> [!IMPORTANT]
> 3CERATOPS is research software under development. Generated structures should be
> inspected, validated, and optimized at the intended electronic-structure
> level before scientific interpretation.

## What 3CERATOPS does

- Parses mononuclear T-REX strings through the `trex-notation` package.
- Predicts metal-ligand distances and ligand-metal-ligand angles with the
  bundled ligand field graph neural network (LF-GNN) ensemble, or uses explicit or idealized internal coordinates.
- Reconstructs coordination environments by Gram-matrix distance geometry.
- Embeds complete ligand structures with constrained RDKit conformer generation.
- Samples multiple random seeds before energy-based conformer selection.
- Optimizes structures with constrained GFN-FF followed by optional GFN2-xTB
  refinement, or with a tiered UMA workflow.
- Handles molecular charge, spin, chelating ligands, and haptic coordination.
- Provides both a command-line interface and a Python API.

## Installation

### Recommended: Conda plus pip

When Conda is available, install `xtb-python` and ASE from conda-forge first.
This avoids asking pip to build the native xTB library and is the recommended
installation route on macOS, Apple silicon, Linux workstations, and HPC systems.
Python 3.10 is recommended for the broadest xTB compatibility.

```bash
git clone https://github.com/KevlishviliGroup/3ceratops.git
cd 3ceratops

conda create -n 3ceratops python=3.10
conda activate 3ceratops
conda install -c conda-forge xtb-python ase

python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[ml]"
```

Do not include the `xtb` extra in the pip command after installing xTB through
Conda. The Conda installation provides the same `xtb` Python import used by
3CERATOPS.

If LF-GNN prediction is not needed, the final command can instead be:

```bash
python -m pip install .
```

The structure generator will then use idealized internal coordinates whenever
explicit values are not supplied.

### Plain virtual environment

A pip-only installation is more restricted because xTB may need to be compiled
from source. Use Python 3.10 or earlier for this route and expect to need a
compatible native compiler toolchain. On Apple silicon, use the Conda procedure
above instead.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[xtb,ml]"
```

### Optional UMA support

```bash
python -m pip install ".[ml,uma]"
```

UMA additionally requires access to the UMA model files and a configured
Hugging Face token. The selected UMA model is downloaded on first use.


### Python imports

```python
import trex_geo
from trex_geo import trex_to_xyz, trex_to_xyz_file
```

## Verify the installation

Run the environment check from outside the source directory:

```bash
cd /tmp
3ceratops doctor
```

`doctor` reports the core, xTB, ML, and UMA stacks separately and confirms that
the bundled LF-GNN ensemble is available. Missing optional dependencies do not
make the command fail; missing core requirements do.

The LF-GNN checkpoints are installed under `LF_GNN/ensemble_gram` and are found
automatically. A complete installation should report ten checkpoints. Because
the ensemble is large, both the source checkout and installed package require
substantial disk space.


## Command-line quick start

T-REX input can be supplied from a file or as a quoted inline string. A file is
usually safer because T-REX strings contain shell metacharacters.

The following example generates low-spin tris(ethylenediamine)cobalt(III):

```bash
printf '%s\n' \
  'Co{+3,S=0} | L=[ SMILES:NCCN, SMILES:NCCN, SMILES:NCCN ] | MAP:{ (1:1, 2:4), (1:4, 3:1), (2:1, 3:4) } | G:O' \
  > co_en3.trex

3ceratops generate \
  --trex-file co_en3.trex \
  --output results/co_en3.xyz \
  --optimizer xtb \
  --internals predicted \
  --n-seeds 5 \
  --top-k 3 \
  --work-dir results/co_en3_work \
  -vv
```

The xTB workflow performs constrained GFN-FF optimization for all successful
seeds, retains the `--top_k` lowest-energy candidates, and refines those candidates
with GFN2-xTB by default. Disable the final refinement with `--no-refine-gfn2`.

To bypass LF-GNN prediction and use idealized internal coordinates:

```bash
3ceratops generate \
  --trex-file co_en3.trex \
  --output results/co_en3_idealized.xyz \
  --optimizer xtb \
  --internals idealized \
  --n-seeds 1 \
  --top-k 1
```

A quoted inline string is also accepted:

```bash
3ceratops generate \
  --trex 'Co{+3,S=0} | L=[ SMILES:NCCN, SMILES:NCCN, SMILES:NCCN ] | MAP:{ (1:1, 2:4), (1:4, 3:1), (2:1, 3:4) } | G:O' \
  --output co_en3.xyz \
  --optimizer xtb \
  --internals predicted \
  --n-seeds 5 \
  --top-k 3
```

### UMA generation

```bash
3ceratops generate \
  --trex-file co_en3.trex \
  --output results/co_en3_uma.xyz \
  --optimizer uma \
  --internals predicted \
  --device cuda \
  --n-seeds 20 \
  --top-k 5 \
  --refine-medium
```

The UMA pipeline screens all seeds with constrained `uma-s-1p1`, optionally
refines the retained candidates with constrained `uma-m-1p1`, and can perform a
final unconstrained medium-model relaxation with `--refine-unconstrained`.

Run `3ceratops generate --help` for all generation options.

## Python API

```python
from trex_geo import trex_to_xyz

trex_str = (
    "Co{+3,S=0} | "
    "L=[ SMILES:NCCN, SMILES:NCCN, SMILES:NCCN ] | "
    "MAP:{ (1:1, 2:4), (1:4, 3:1), (2:1, 3:4) } | "
    "G:O"
)

result = trex_to_xyz(
    trex_str,
    n_seeds=5,
    top_k=3,
    refine_gfn2=True,
)

if not result.success:
    raise RuntimeError(result.error)

print(result.mol.GetNumAtoms())
print(result.energy)              # Hartree
print(result.internals_source)    # provided, predicted, or idealized
print(result.diagnostics)
```

Write the best structure directly to an XYZ file:

```python
from trex_geo import trex_to_xyz_file, trex_to_xyz_uma_file

trex_to_xyz_file(trex_str, "xtb_structure.xyz")
trex_to_xyz_uma_file(trex_str, "uma_structure.xyz")
```

`TrexToXyzResult.energy` and XYZ comment-line energies are reported in Hartree
for both pipelines.

## Internal-coordinate sources

3CERATOPS uses internal coordinates in this order:

1. explicit `internals` supplied by the caller;
2. LF-GNN prediction, unless `use_idealized=True`;
3. idealized values from `trex.geometry`.

The bundled ensemble is used automatically. To use another ensemble, pass a
directory containing `seed_*/best.pt` checkpoints:

```bash
3ceratops generate \
  --trex-file complex.trex \
  --output complex.xyz \
  --model-dir /path/to/ensemble_gram
```

The Python equivalent is `model_path="/path/to/ensemble_gram"`. A flat
directory of `.pt` checkpoint files is also accepted.

If prediction dependencies, checkpoints, or inference are unavailable, the
generation functions log the problem and fall back to idealized
internals. Check `result.internals_source` or the XYZ comment line to confirm
which source was used.

Explicit internals use distances in angstrom and angle cosines:

```python
internals = {
    "site_r": [2.30, 2.30, 2.05, 2.05],
    "trans_cos": [-0.99, -0.99],
    "cis_cos": [0.01, 0.01, -0.01, -0.01],
    "cis_pair_indices": [(0, 2), (0, 3), (1, 2), (1, 3)],
}

pt_trex = (
    "Pt{+2}|"
    "L=[SMILES:[Cl-],SMILES:[Cl-],SMILES:N,SMILES:N]|"
    "MAP:{(1:1,2:1),(3:1,4:1)}|"
    "G:SP"
)

result = trex_to_xyz(pt_trex, internals=internals)
```

## Generation workflow

For each T-REX input, 3CERATOPS:

1. parses the metal, charge, spin, ligands, donor atoms, topology, and geometry;
2. obtains explicit, predicted, or idealized internal coordinates;
3. reconstructs the metal-centered coordination environment;
4. embeds the complete complex with multiple RDKit random seeds;
5. performs constrained low-cost optimization on every successful seed;
6. ranks the candidates by energy and refines the lowest-energy subset; and
7. returns or writes the best final structure.

The current pipeline ranks candidates without T-REX round-trip
validation because reconstruction can be slow for select structures.
For dataset generation, validate structures separately:

```python
from trex_geo.multiseed_trex_validate import validate_trex_roundtrip

is_valid, reconstructed = validate_trex_roundtrip(
    mol=result.mol,
    original_trex=trex_str,
    charge=+3,
    geometry="octahedral",
)
```

## Haptic sites

The default `haptic_scale=0.35` contracts haptic ghost-site distances only
during constrained embedding. This is an empirical scale that worked best for 
reconstruction, not a physical metal-centroid distance. The unscaled 
metal-centroid distance is used to construct haptic constraints during 
xTB or UMA relaxation.

## Supported geometry mappings

`TrexFull.get_geo_type()` can return inferred lowercase codes or explicit
geometry flags. The current mapping is:

| Geometry | Inferred code | Explicit code | CN |
|---|---:|---:|---:|
| Octahedral | `O` | `O` | 6 |
| Square planar | `sqpl` | `SP` | 4 |
| Square pyramidal | `sqpy` | `SQP` | 5 |
| Tetrahedral | `thd` | `T` | 4 |
| Trigonal bipyramidal | `trbp` | `TBP` | 5 |
| Pentagonal bipyramidal | `pntbp` | - | 7 |
| Linear | `ln` | `L` | 2 |
| Bent | `bnt` | - | 2 |
| Trigonal planar | `trpl` | `TP` | 3 |
| Trigonal pyramidal | - | `TPY` | 3 |
| T-shaped | `tsh` | - | 3 |
| See-saw | `ssw` | - | 4 |

## tmQMg ligand index

The packaged ligand catalog stores observed canonical donor sites, haptic sites,
intraligand trans pairs, binding-mode counts, and subset-based hemilability evidence.
Trigonal-prismatic complexes are intentionally excluded.

Input SMILES are canonicalized with the same T-REX/RDKit path used to build the
catalog. The catalog records the RDKit version and lookup fails on a version mismatch,
preventing silent changes to atom numbering.

```python
from trex_geo.assemble import BindingModeInput, LigandIndex

index = LigandIndex()
record = index.lookup("NCCN")

# A missing ligand can be added to the user's override index. Site indices below
# are 1-based positions in the supplied (not yet canonicalized) SMILES.
record = index.resolve(
    "PCCN",
    if_missing=BindingModeInput(
        sites=((1,), (4,)),
        trans_pairs=(),
        label="cis_bidentate",
    ),
)
```

The packaged catalog is never modified at runtime. User additions are written
atomically to `~/.3ceratops/ligands.json`, or to the path set by
`THREECERATOPS_LIGAND_OVERRIDES`.

To rebuild the packaged catalog reproducibly:

```bash
python scripts/build_tmqmg_ligand_index.py \
  Final_tmqmG_clean.csv trex_geo/data/tmqmg_ligand_index.json
```

## Stoichiometry assembly

`assemble` first resolves each ligand against the catalog, then enumerates ligand
counts and binding-mode multisets that exactly fill each requested coordination
geometry. The default uses the most frequently observed binding mode; request
`all_observed` to include lower-denticity or alternative modes.

```python
from trex_geo.assemble import AssemblySpace, LigandSpec, enumerate_stoichiometries

space = AssemblySpace(
    metal="Pd",
    oxidation_state=2,
    spin=0,
    ligands=(
        LigandSpec("NCCN", max_count=2),
        LigandSpec("[Cl-]", max_count=4),
    ),
    geometries=("square_planar", "octahedral"),
)

candidates = enumerate_stoichiometries(space, max_candidates=500)
```

This stage returns structured stoichiometric candidates. It deliberately does
not enumerate MAP isomers or generate coordinates; those are subsequent assembly
stages. Trigonal-prismatic geometry is unsupported.

## Dependencies

| Capability | Packages and resources |
|---|---|
| Core | NumPy, RDKit, PuLP, SciPy, and `trex-notation` |
| xTB optimization | ASE and `xtb-python` |
| LF-GNN internals | PyTorch, PyTorch Geometric, and the bundled checkpoints |
| UMA optimization | ASE, PyTorch, `fairchem-core`, UMA access, and a Hugging Face token |
| Development | pytest, coverage, Black, isort, and mypy |


## Examples and tests

Four standalone Python examples are included:

1. `examples/01_basic_xtb.py`
2. `examples/02_explicit_internals.py`
3. `examples/03_multiseed_predicted.py`
4. `examples/04_uma_write.py`

For development, first install xTB and ASE with Conda, then install the package
in editable mode:

```bash
conda install -c conda-forge xtb-python ase
python -m pip install -e ".[ml,dev]"
pytest
```

## Troubleshooting

### xTB fails during pip installation

Use the recommended Conda installation:

```bash
conda install -c conda-forge xtb-python ase
python -m pip install ".[ml]"
```

Do not include `xtb` in the pip extras after the Conda installation.

### `doctor` cannot find the model ensemble

The default installation includes `LF_GNN/ensemble_gram/seed_*/best.pt`. If a
custom or incomplete installation omits the checkpoints, provide the ensemble
directory explicitly:

```bash
3ceratops doctor --model-dir /path/to/LF_GNN/ensemble_gram
```

### Predicted generation reports `source=idealized`

Prediction was unavailable and 3CERATOPS used its documented fallback. Run
`3ceratops doctor`, confirm that PyTorch and PyTorch Geometric are installed,
and confirm that the model ensemble contains the expected checkpoints.


## Assemble: enumerate complex topologies

`trex_geo.assemble` enumerates exact coordination-number stoichiometries 
of supplied ligands, all compatible T-REX MAP assignments, and then 
canonicalizes and deduplicates the resulting full T-REX strings. 
Some ligands can be inferred from common names by Ligand2SMILES installation; 
all T-REX strings are assembled using SMILES.

```python
from trex_geo.assemble import (
    AssemblyConstraints,
    AssemblySpace,
    LigandSpec,
    SiteRef,
    assemble_trex,
)

space = AssemblySpace(
    metal="Pd",
    oxidation_state=2,
    spin=0,
    ligands=(
        LigandSpec("NCCN", max_count=1, min_count=1, identifier="en"),
        LigandSpec("[Cl-]", max_count=2, min_count=2, identifier="chloride"),
    ),
    geometries=("square_planar",),
)

constraints = AssemblyConstraints(
    required_trans=((
        SiteRef("en", copy=0, atoms=(1,)),
        SiteRef("chloride", copy=0, atoms=(1,)),
    ),),
)

structures = assemble_trex(space, constraints=constraints)
for structure in structures:
    print(structure.trex)
```

Generate optimized 3D structures for every unique topology:

```python
from trex_geo.assemble import generate_assembled

batch = generate_assembled(
    structures,
    "assembled_pd",
    optimizer="xtb",
    generator_kwargs={
        "model_path": "/path/to/LF_GNN/ensemble_gram",
        "n_seeds": 10,
        "top_k": 3,
        "refine_gfn2": True,
    },
)
print(batch.succeeded, batch.failed, batch.manifest_path)
```

Each canonical T-REX string receives a readable T-REX-derived candidate
directory plus a short stable hash. `manifest.json` records the topology, 
geometry, stoichiometry, binding modes, optimizer, energy, internal-coordinate 
source, diagnostics, and any failure. It is updated atomically after every 
candidate. Rerunning skips successful candidates whose XYZ files remain and 
retries failed or missing candidates; use `overwrite=True` to regenerate
everything. One failed candidate does not stop the rest of the batch.

`SiteRef.atoms` uses the same 1-based, ligand-local canonical-SMILES atom
indices as T-REX `MapSite`. A multi-atom tuple is one haptic site. `copy` is
zero-based within a `LigandSpec`; instantiated T-REX ligand positions remain
1-based. Required-cis pairs are forbidden from appearing in a T-REX MAP pair.
An intraligand required-trans pair must already belong to the selected indexed
binding mode, so constraints cannot silently change cis/trans, fac/mer, or
tetradentate ligand geometry.

Trigonal-prismatic assembly is deliberately unsupported.

### Assemble from the CLI

The natural command runs enumeration and batch 3D generation:

```bash
3ceratops assemble examples/assemble.yaml
```

To inspect the canonical topology space first:

```bash
3ceratops assemble examples/assemble.yaml --enumerate-only
```

An example YAML file is provided in examplesJSON uses the same keys as YAML. 
Each ligand supplies either `smiles` or an exact Ligand2SMILES `name`; 
both are converted to canonical SMILES before index lookup. Constraint 
atoms are 1-based canonical-SMILES atom indices, matching
T-REX `MapSite`, while `copy` is zero-based within the named ligand spec.

The CLI always writes `assembled.trex`, containing one canonical full T-REX
string per line. A complete generation run also creates:

```text
assembled_pd/
  assembled.trex
  manifest.json
  Pd_+2_LIGANDS_...__6d723a98/
    structure.xyz
    topology.trex
    generation.json
    work/
```

The candidate prefix is derived from the canonical T-REX string and sanitized
for filesystem use. The short hash prevents collisions caused by sanitization
or filename-length truncation. The complete canonical string is retained in
`topology.trex`, the XYZ comment, and both manifests.

## Limitations

- The package is research software and is not guaranteed to produce a
  chemically appropriate structure for every valid T-REX input.
- The LF-GNN ensemble predicts internal coordinates for chemistry represented
  in its training domain; out-of-domain predictions require additional care.
- The default generation workflow does not perform T-REX round-trip validation.
- For dataset generation, validate the returned structure separately.
- xTB and UMA energies are useful for screening but do not replace
  higher-level electronic-structure calculations.
- Generated structures should be inspected and optimized at the intended DFT
  level before scientific interpretation.

## Citation

If you use 3CERATOPS, cite the software release associated with the version
you ran. A provisional software citation is:

```bibtex
@software{kevlishvili_3ceratops_2026,
  author = {Kevlishvili, Ilia},
  title = {3CERATOPS: 3D Coordination Environment Reconstruction from
           Abstract TOPology to Structure},
  year = {2026},
  url = {https://github.com/KevlishviliGroup/3ceratops}
}
```

Please also cite the T-REX paper:

> I. Kevlishvili and D. Dorabawila, "Taming T-REX: a canonical language for
> geometry-aware generative design of transition-metal complexes," *Digital
> Discovery* **2026**, *5*, 2845. <https://doi.org/10.1039/D6DD00129G>

## License and acknowledgments

3CERATOPS is distributed under the MIT License. See `LICENSE`.

- Bond-order assignment was adapted from MetalloGen under its BSD 3-Clause
  License.
- Coordination-sphere placement uses Gram-matrix distance reconstruction.

Questions and bug reports can be submitted through the
[KevlishviliGroup/3ceratops issue tracker](https://github.com/KevlishviliGroup/3ceratops/issues).
