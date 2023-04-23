#!/usr/bin/env python

import numpy as np

import olympus
from olympus import Logger
from olympus.campaigns.observations import Observations
from olympus.campaigns.param_space import ParameterSpace
from olympus.datasets import Dataset
from olympus.scalarizers.scalarizer import Scalarizer
from olympus.emulators.emulator import Emulator
from olympus.objects import Object, ParameterContinuous
from olympus.surfaces.surface import AbstractSurface
from olympus.utils import generate_id
from olympus.utils.data_transformer import cube_to_simpl, simpl_to_cube


class Campaign(Object):
    """stores information about a single optimization run

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
    ATT_VALUE_SPACE = {"type": "ParameterSpace", "default": ParameterSpace}
    ATT_PLANNER_KIND = {"type": "string", "default": "n/a"}

    ATT_SCALARIZER = {"type": "Scalarizer", "default": "n/a"}

    ATT_IS_MOO = {"type": "bool", "default": False}
    ATT_SCALARIZED_OBSERVATIONS = {
        "type": "Observations",
        "default": Observations,
    }

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
    def scalarized_values(self, *args, **kwargs):
        return self.scalarized_observations.get_values(*args, **kwargs)

    @property
    def best_values(self):
        """returns an array of the best objective function values at each
        iteration of the campaign
        """
        # check to see if we have a moo problem
        if self.is_moo:
            # multiobjective optimization, the minimum scalarized merit
            # value always corresponds to the best parameters (irrespective
            # of the individual optimization goals)
            vals = self.observations.get_values()
            scal_vals = self.scalarized_observations.get_values()
            if len(scal_vals) == 0:
                return vals
            best_vals = [vals[0]]
            best_merit = scal_vals[0]
            for idx, (scal_val, val) in enumerate(
                zip(scal_vals[1:], vals[1:])
            ):
                to_add = np.argmin(
                    [scal_val, best_merit]
                )  # 0 means add current measurement, 1 means re-append previous best
                best_vals.append([val, best_vals[-1]][to_add])
            best_vals = np.array(best_vals)
            return best_vals

        else:
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

    def add_and_scalarize(self, param, value, scalarizer):
        # successively add observation, then scalarize the entire history
        # of objective measurements
        setattr(self, 'scalarizer', scalarizer)

        self.observations.add_observation(param, value)
        self.scalarized_observations.add_observation(
            param, 1.0
        )  # dummy objective value

        # compute the scalarized merits from the objective values
        values = self.observations.get_values()  # (# obs, # objs)
        merits = scalarizer.scalarize(values)

        # update scalarized_observations
        self.reset_merit_history(merits)

    def observations_to_simpl(self):
        """convert parameters for the current observations from cube
        to simplex
        """
        # do not need to make transformation if we have no observations yet,
        # leave the params attribute as None
        if self.observations.params is not None:
            cube_params = self.observations.get_params()
            simpl_params = cube_to_simpl(cube_params)
            self.observations.params = simpl_params
            self.scalarized_observations.params = simpl_params
        else:
            message = f"No observations found. Skipping requested cube to simplex transformation on parameters."
            Logger.log(message, "WARNING")

    def observations_to_cube(self):
        """convert parameters for the current observations from simplex
        to cube
        """
        # do not need to make transformation if we have no observations yet,
        # leave the params attribute as None
        if self.observations.params is not None:
            simpl_params = self.observations.get_params()
            cube_params = simpl_to_cube(simpl_params)
            self.observations.params = cube_params
            self.scalarized_observations.params = cube_params
        else:
            message = "No observations found. Skipping requested simplex to cube transformation on parameters"

    def observations_to_int(self):
        """ convert ordinal targets in string representation to their integer indices
        """
        # quick check if we have the proper value space for this method
        if not np.all([v.type=='ordinal' for v in self.value_space]):
            message = '"observations_to_int" method reserved for use with ordinal objectives only'
            Logger.log(mesage, 'FATAL')
        if self.observations.values is not None:
            str_values = self.observations.get_values()
            int_values = []
            for str_value in str_values:
                int_value = []
                for value_ix, value in enumerate(self.value_space):
                    int_value.append(value['options'].index(str_value[value_ix]))
                int_values.append(int_value)
            self.observations.values = np.array(int_values)
        else:
            pass

    def observations_to_str(self):
        """ convert ordinal targets in integer representation to their string descriptions
        """
        if not np.all([v.type=='ordinal' for v in self.value_space]):
            message = '"observations_to_str" method reserved for use with ordinal objectives only'
            Logger.log(mesage, 'FATAL')
        if self.observations.values is not None:
            int_values = self.observations.get_values()
            str_values = []
            for int_value in int_values:
                str_value = []
                for value_ix, value in enumerate(self.value_space):
                    str_value.append(value['options'][int_value[value_ix]])
                str_values.append(str_value)
            self.observations.values = np.array(str_values)
        else:
            pass



    def reset_merit_history(self, merits):
        """updates the scalarized_observation history with a 1d list or
        array of merits values
        """
        if not len(merits) == len(self.observations.get_values()):
            message = "Length of provided merits does not match the number of current observations"
            Logger.log(message, "FATAL")
        dim_merits = len(np.array(merits).shape)
        if not dim_merits == 1:
            message = f"Merits must be a 1D list or array. You provided a {dim_merits}D array."
            Logger.log(message, "FATAL")
        # TODO: should we check here if the merits are bewteen 0 and 1? nessecary??
        self.scalarized_observations.values = merits

    def set_planner_specs(self, planner):
        self.set_planner_kind(planner.kind)
        self.set_goal(planner.goal)

    def set_param_space(self, param_space):
        self.param_space = param_space
        self.observations.set_param_space(param_space)
        # FOR MOO
        self.scalarized_observations.set_param_space(param_space)

    def set_value_space(self, value_space):
        self.value_space = value_space
        self.observations.set_value_space(value_space)
        if len(self.value_space) > 1:
            self.is_moo = True
        # for moo --> make merit objective
        self.scalarized_observations.set_value_space(
            ParameterSpace().add(
                ParameterContinuous(
                    name="merit",
                    low=0.0,
                    high=1.0,
                )
            )
        )

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
        elif isinstance(emulator, Dataset):
            # emualtor is a dataset
            self.set_emulator_type("numeric")
            self.set_dataset_kind(emulator.kind)
            self.set_measurement_name(emulator.measurement_name)
            # TODO: this is a messy hack
            if isinstance(emulator.goal, str):
                self.set_goal(emulator.goal)
            else:
                pass
            # self.set_model_kind(emulator.model.kind)
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
