"""
3-CERATOPS (`trex_geo`): T-REX to 3D structure generation

Convert T-REX canonical strings to 3D molecular structures.

Example usage:
    from trex_geo import trex_to_xyz, trex_to_xyz_file
    
    # Generate 3D structure from T-REX string
    result = trex_to_xyz(trex_str)
    
    if result.success:
        mol = result.mol  # RDKit Mol with 3D conformer
        print(f"Energy: {result.energy}")
    
    # Or write directly to file
    trex_to_xyz_file(trex_str, "output.xyz")
"""

__version__ = "0.1.0"

from .trex_to_xyz import (
    trex_to_xyz,
    trex_to_xyz_file,
    TrexToXyzResult,
    get_structure_info,
    validate_internals,
    compute_idealized_internals,
)

from .trex_embed import (
    embed_trex_metallogen,
    EmbedResult,
)

from .coord_reconstruction import (
    reconstruct_coordination_sphere,
)

from .multiseed_trex_validate import (
    generate_and_validate,
    generate_with_seeds,
    MultiSeedResult,
    SeedResult,
)

from .xtb_optimize import (
    optimize_with_xtb,
    HapticInfo,
)

from .trex_to_xyz_uma import (
    trex_to_xyz_uma,
    trex_to_xyz_uma_file,
)

from .uma_optimize import (
    optimize_with_uma,
    UMAResult,
)

__all__ = [
    # Main API
    "trex_to_xyz",
    "trex_to_xyz_file",
    "TrexToXyzResult",
    # Structure utilities
    "get_structure_info",
    "validate_internals",
    "compute_idealized_internals",
    # Lower-level embedding
    "embed_trex_metallogen",
    "EmbedResult",
    # Coordinate reconstruction
    "reconstruct_coordination_sphere",
    # Multi-seed pipeline
    "generate_and_validate",
    "generate_with_seeds",
    "MultiSeedResult",
    "SeedResult",
    # xTB optimization
    "optimize_with_xtb",
    "HapticInfo",
    # UMA pipeline
    "trex_to_xyz_uma",
    "trex_to_xyz_uma_file",
    "optimize_with_uma",
    "UMAResult",
]
