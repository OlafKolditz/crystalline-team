# Simple liquid flow modell on cuboid domain (without Python BC)

- flow from left to right
- time dependent source term

run with

```
ogs -p lower_permeability.xml -p FunctionDependentSourceTerm.xml left_right_gw_flow_source_term_base.prj -m meshes/cuboid_1024x1024x256_hex_64x64x16 -o results_cuboid_1024x1024x256_hex_64x64x16_lower-permeability-obstacles-lenses_source_terms
```

# Adding Python BC

## Required files

- scripts/SimplePythonBC.xml
- xml patch file SimplePythonBC.xml
- patch file
  - adds path to the Python boundary condition script
  - substituts the existing boundary condition with the Python-type boundary
    condition
- Documentation: https://www.opengeosys.org/6.5.8/docs/userguide/features/python_bc/

```
~/w/o/build/release/bin/ogs -p lower_permeability.xml -p SimplePythonBC.xml left_right_gw_flow_source_term_base.prj -m meshes/cuboid_1024x1024x256_hex_64x64x16 --write-prj -o results
                                                      ^^^ patch the project
```

