#!/usr/bin/env python

import numpy as np
from olympus.emulators.emulator import Emulator
from olympus.surfaces.surface import AbstractSurface
from olympus.objects import Object
from olympus.utils import generate_id
from olympus.campaigns import ParameterSpace
from olympus.campaigns import Observations


class Campaign(Object):
    """ stores information about a single optimization run

    This class logs all information about a single optimization run,

    """

    ATT_ACCEPTS = {
        "type": "string",
        "default": "param_vector",
        "valid": ["param_vector", "array", "dict"],
    }
    ATT_GOAL = {
        "type": "string",
        "default": "minimize",
        "valid": ["minimize", "maximize"],
    }
    ATT_EMULATOR_TYPE = {
        "type": "string",
        "default": "n/a",
        "valid": ["numeric", "analytic"],
    }
    ATT_DATASET_KIND = {"type": "string", "default": "n/a"}
    ATT_MEASUREMENT_NAME = {"type": "string", "default": "n/a"}
    ATT_MODEL_KIND = {"type": "string", "default": "n/a"}
    ATT_SURFACE_KIND = {"type": "string", "default": "n/a"}
    ATT_ID = {"type": "string", "default": generate_id}
    ATT_OBSERVATIONS = {"type": "Observations", "default": Observations}
    ATT_PARAM_SPACE = {"type": "ParameterSpace", "default": ParameterSpace}
    ATT_PLANNER_KIND = {"type": "string", "default": "n/a"}

    def __repr__(self):
        repr_ = f"<Campaign (dataset={self.dataset_kind}, model={self.model_kind}, planner={self.planner_kind}, num_iter={len(self.params)})>"
        return repr_

    @property
    def params(self, *args, **kwargs):
        return self.observations.get_params(*args, **kwargs)

    @property
    def values(self, *args, **kwargs):
        return self.observations.get_values(*args, **kwargs)

    @property
    def best_values(self):
        vals = self.observations.get_values()
        if len(vals) == 0:
            return vals
        best_vals = [vals[0]]
        for val in vals[1:]:
            if self.goal == "minimize":
                best_vals.append(np.minimum(val, best_vals[-1]))
            elif self.goal == "maximize":
                best_vals.append(np.maximum(val, best_vals[-1]))
        best_vals = np.array(best_vals)
        return best_vals

    def add_observation(self, param, value):
        self.observations.add_observation(param, value)

    def set_planner_specs(self, planner):
        self.set_planner_kind(planner.kind)
        self.set_goal(planner.goal)

    def set_param_space(self, param_space):
        self.param_space = param_space
        self.observations.set_param_space(param_space)

    def set_value_space(self, value_space):
        self.value_space = value_space
        self.observations.set_value_space(value_space)

    def set_emulator_specs(self, emulator):
        """Store info about the Emulator (or Surface) into the campaign object.

        Args:
            emulator (object): an Emulator or Surface instance.

        """
        self.set_accepts("array")
        if isinstance(emulator, Emulator):
            self.set_emulator_type("numeric")
            self.set_dataset_kind(emulator.dataset.kind)
            self.set_measurement_name(emulator.dataset.measurement_name)
            self.set_model_kind(emulator.model.kind)
        elif isinstance(emulator, AbstractSurface):
            self.set_emulator_type("analytic")
            self.set_surface_kind(emulator.kind)

        self.set_param_space(emulator.param_space)
        self.set_value_space(emulator.value_space)

    # def set_surface_specs(self, surface):
    # 	self.set_accepts('array')
    # 	self.set_surface_name(surface.name)
    # self.set_goal(surface.goal)
    # 	self.set_param_space(surface.get_param_space())

    def to_dict(self):
        return Object.to_dict(self, exclude=["func"])


# ===============================================================================
