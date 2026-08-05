try:
    import ogs.callbacks as OpenGeoSys
except ModuleNotFoundError:
    import OpenGeoSys

# Dirichlet BCs
class LeftBoundaryCondition(OpenGeoSys.BoundaryCondition):
    def getDirichletBCValue(self, _t, coords, _node_id, _primary_vars):
        x, y, z = coords
        value = (300-z)*1000*9.81
        return (True, value)

# instantiate BC objects referenced in OpenGeoSys' prj file
left_boundary_dirichlet = LeftBoundaryCondition()
