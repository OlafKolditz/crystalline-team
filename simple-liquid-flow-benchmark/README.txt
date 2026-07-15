# Simple liquid flow modell on cuboid domain

- flow from left to right
- time dependent source term

run with
`
ogs -p lower_permeability.xml -p FunctionDependentSourceTerm.xml left_right_gw_flow_source_term_base.prj -m meshes/cuboid_1024x1024x256_hex_64x64x16 -o results_cuboid_1024x1024x256_hex_64x64x16_lower-permeability-obstacles-lenses_source_terms
