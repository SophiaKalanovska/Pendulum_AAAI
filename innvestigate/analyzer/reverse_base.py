import random

import numpy as np
import six

import tensorflow as tf
import tensorflow.keras.backend as K
from innvestigate import layers as ilayers, utils as iutils
import innvestigate.utils.keras as kutils
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.utils.keras import graph as kgraph

class ReverseAnalyzerBase(AnalyzerNetworkBase):
    """Convenience class for analyzers that revert the model's structure.

    This class contains many helper functions around the graph
    reverse function :func:`innvestigate.utils.keras.graph.reverse_model`.

    The deriving classes should specify how the graph should be reverted
    by implementing the following functions:

    * :func:`_reverse_mapping(layer)` given a layer this function
      returns a reverse mapping for the layer as specified in
      :func:`innvestigate.utils.keras.graph.reverse_model` or None.

      This function can be implemented, but it is encouraged to
      implement a default mapping and add additional changes with
      the function :func:`_add_conditional_reverse_mapping` (see below).

      The default behavior is finding a conditional mapping (see below),
      if none is found, :func:`_default_reverse_mapping` is applied.
    * :func:`_default_reverse_mapping` defines the default
      reverse mapping.
    * :func:`_head_mapping` defines how the outputs of the model
      should be instantiated before the are passed to the reversed
      network.

    Furthermore other parameters of the function
    :func:`innvestigate.utils.keras.graph.reverse_model` can
    be changed by setting the according parameters of the
    init function:

    :param reverse_verbose: Print information on the reverse process.
    :param reverse_clip_values: Clip the values that are passed along
      the reverted network. Expects tuple (min, max).
    :param reverse_project_bottleneck_layers: Project the value range
      of bottleneck tensors in the reverse network into another range.
    :param reverse_check_min_max_values: Print the min/max values
      observed in each tensor along the reverse network whenever
      :func:`analyze` is called.
    :param reverse_check_finite: Check if values passed along the
      reverse network are finite.
    :param reverse_keep_tensors: Keeps the tensors created in the
      backward pass and stores them in the attribute
      :attr:`_reversed_tensors`.
    :param reverse_reapply_on_copied_layers: See
      :func:`innvestigate.utils.keras.graph.reverse_model`.
    """

    def __init__(self,
                 model,
                 reverse_verbose=False,
                 reverse_clip_values=False,
                 reverse_project_bottleneck_layers=False,
                 reverse_check_min_max_values=False,
                 reverse_check_finite=False,
                 reverse_keep_tensors=True,
                 reverse_reapply_on_copied_layers=False,
                 **kwargs):
        self._reverse_verbose = reverse_verbose
        self._reverse_clip_values = reverse_clip_values
        self._reverse_project_bottleneck_layers = (
            reverse_project_bottleneck_layers)
        self._reverse_check_min_max_values = reverse_check_min_max_values
        self._reverse_check_finite = reverse_check_finite
        self._reverse_keep_tensors = reverse_keep_tensors
        self._reverse_reapply_on_copied_layers = (
            reverse_reapply_on_copied_layers)
        super(ReverseAnalyzerBase, self).__init__(model, **kwargs)

    def _gradient_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state, forward = False):
        if forward:
            if reverse_state["last"]:
                # percent = [tf.expand_dims(percent[0], 0)]
                # R_prime = [tf.keras.layers.Multiply()([a, b])
                #            for a, b in zip(reversed_Ys, percent)]
                # X_prime = [tf.keras.layers.Multiply()([a, b])
                #            for a, b in zip(Xs, reverse_state["percent"])]
                R_prime = tf.expand_dims(reverse_state["relevance_prime"][0][0][reverse_state["the_label_index"][0]], 0)
                return R_prime
            else:
                self._layer_wo_act = kgraph.copy_layer_wo_activation(reverse_state["layer"],
                                                                     name_template="reversed_kernel_%s" + str(
                                                                         random.randint(0, 10000000)))
                Zs = kutils.apply(self._layer_wo_act, Xs)

                X_prime = [tf.keras.layers.Multiply()([a, b])
                           for a, b in zip(Xs, reverse_state["percent"])]

                Zs_prime = kutils.apply(reverse_state['layer'], X_prime)

                percent = [tf.math.divide_no_nan(n, d)
                           for n, d in zip(Zs_prime, Zs)]

                R_prime = [tf.keras.layers.Multiply()([a, b])
                           for a, b in zip(reversed_Ys, percent)]
                return R_prime, percent
        else:
            mask = [x not in reverse_state["stop_mapping_at_tensors"] for x in Xs]
            return ilayers.GradientWRT(len(Xs), mask=mask)(Xs + Ys + reversed_Ys)

    def _reverse_mapping(self, layer):
        """
        This function should return a reverse mapping for the passed layer.

        If this function returns None, :func:`_default_reverse_mapping`
        is applied.

        :param layer: The layer for which a mapping should be returned.
        :return: The mapping can be of the following forms:
          * A function of form (A) f(Xs, Ys, reversed_Ys, reverse_state)
            that maps reversed_Ys to reversed_Xs (which should contain
            tensors of the same shape and type).
          * A function of form f(B) f(layer, reverse_state) that returns
            a function of form (A).
          * A :class:`ReverseMappingBase` subclass.
        """
        if layer in self._special_helper_layers:
            # Special layers added by AnalyzerNetworkBase
            # that should not be exposed to user.
            return self._gradient_reverse_mapping

        return self._apply_conditional_reverse_mappings(layer)

    def _add_conditional_reverse_mapping(
            self, condition, mapping, priority=-1, name=None):
        """
        This function should return a reverse mapping for the passed layer.

        If this function returns None, :func:`_default_reverse_mapping`
        is applied.

        :param condition: Condition when this mapping should be applied.
          Form: f(layer) -> bool
        :param mapping: The mapping can be of the following forms:
          * A function of form (A) f(Xs, Ys, reversed_Ys, reverse_state)
            that maps reversed_Ys to reversed_Xs (which should contain
            tensors of the same shape and type).
          * A function of form f(B) f(layer, reverse_state) that returns
            a function of form (A).
          * A :class:`ReverseMappingBase` subclass.
        :param priority: The higher the earlier the condition gets
          evaluated.
        :param name: An identifying name.
        """
        if getattr(self, "_reverse_mapping_applied", False):
            raise Exception("Cannot add conditional mapping "
                            "after first application.")

        if not hasattr(self, "_conditional_reverse_mappings"):
            self._conditional_reverse_mappings = {}

        if priority not in self._conditional_reverse_mappings:
            self._conditional_reverse_mappings[priority] = []

        tmp = {"condition": condition, "mapping": mapping, "name": name}
        self._conditional_reverse_mappings[priority].append(tmp)

    def _apply_conditional_reverse_mappings(self, layer):
        mappings = getattr(self, "_conditional_reverse_mappings", {})
        self._reverse_mapping_applied = True

        # Search for mapping. First consider ones with highest priority,
        # inside priority in order of adding.
        sorted_keys = sorted(mappings.keys())[::-1]
        for key in sorted_keys:
            for mapping in mappings[key]:
                if mapping["condition"](layer):
                    return mapping["mapping"]

        return None

    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        """
        Fallback function to map reversed_Ys to reversed_Xs
        (which should contain tensors of the same shape and type).
        """
        return self._gradient_reverse_mapping(
            Xs, Ys, reversed_Ys, reverse_state)

    def _head_mapping(self, X):
        """
        Map output tensors to new values before passing
        them into the reverted network.
        """
        return X

    def _postprocess_analysis(self, X):
        return X

    def _reverse_model(self,
                       model,
                       stop_analysis_at_tensors=[],
                       return_all_reversed_tensors=True):
        return kgraph.reverse_model(
            model,
            reverse_mappings=self._reverse_mapping,
            default_reverse_mapping=self._default_reverse_mapping,
            head_mapping=self._head_mapping,
            stop_mapping_at_tensors=stop_analysis_at_tensors,
            verbose=self._reverse_verbose,
            clip_all_reversed_tensors=self._reverse_clip_values,
            project_bottleneck_tensors=self._reverse_project_bottleneck_layers,
            return_all_reversed_tensors=return_all_reversed_tensors)
    def _forward_model(self,
                       model,
                       the_label_index,
                       stop_analysis_at_tensors=[],
                       return_all_reversed_tensors=False,
                       mask=[]):
        return kgraph.forward_model(
            model,
            the_label_index,
            reverse_mappings=self._reverse_mapping,
            default_reverse_mapping=self._default_reverse_mapping,
            head_mapping=self._head_mapping,
            stop_mapping_at_tensors=stop_analysis_at_tensors,
            verbose=self._reverse_verbose,
            clip_all_reversed_tensors=self._reverse_clip_values,
            project_bottleneck_tensors=self._reverse_project_bottleneck_layers,
            return_all_reversed_tensors=return_all_reversed_tensors,
            mask=mask)

    def _create_analysis(self, model, stop_analysis_at_tensors=[]):
        return_all_reversed_tensors = (
                self._reverse_check_min_max_values or
                self._reverse_check_finite or
                self._reverse_keep_tensors
        )
        ret = self._reverse_model(
            model,
            stop_analysis_at_tensors=stop_analysis_at_tensors,
            return_all_reversed_tensors=return_all_reversed_tensors)

        if return_all_reversed_tensors:
            ret = (self._postprocess_analysis(ret[0]), ret[1])
            # self._reversed_tensors = ret[1]
            # ret = ret[0]
        else:
            ret = self._postprocess_analysis(ret)
        #
        # return_all_reversed_tensors = False
        if return_all_reversed_tensors:
            debug_tensors = []
            self._debug_tensors_indices = {}

            values_grad = list(six.itervalues(ret[1]))
            mapping = {i: v["id"] for i, v in enumerate(values_grad)}
            self.tensors = [v["final_tensor"] for v in values_grad]
            self._reverse_tensors_mapping = mapping
            # tensors = ret[1]

            if self._reverse_check_min_max_values:
                tmp = [ilayers.Min(None)(x) for x in self.tensors]
                self._debug_tensors_indices["min"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp))
                debug_tensors += tmp

                tmp = [ilayers.Max(None)(x) for x in self.tensors]
                self._debug_tensors_indices["max"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp))
                debug_tensors += tmp

            if self._reverse_check_finite:
                tmp = iutils.to_list(ilayers.FiniteCheck()(self.tensors))
                self._debug_tensors_indices["finite"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp))
                debug_tensors += tmp

            if self._reverse_keep_tensors:
                self._debug_tensors_indices["keep"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(self.tensors))
                debug_tensors += self.tensors

            ret = (ret[0], debug_tensors)
        return ret

    def _handle_debug_output(self, debug_values):

        if self._reverse_check_min_max_values:
            indices = self._debug_tensors_indices["min"]
            tmp = debug_values[indices[0]:indices[1]]
            tmp = sorted([(self._reverse_tensors_mapping[i], v)
                          for i, v in enumerate(tmp)])
            print("Minimum values in tensors: "
                  "((NodeID, TensorID), Value) - {}".format(tmp))

            indices = self._debug_tensors_indices["max"]
            tmp = debug_values[indices[0]:indices[1]]
            tmp = sorted([(self._reverse_tensors_mapping[i], v)
                          for i, v in enumerate(tmp)])
            print("Maximum values in tensors: "
                  "((NodeID, TensorID), Value) - {}".format(tmp))

        if self._reverse_check_finite:
            indices = self._debug_tensors_indices["finite"]
            tmp = debug_values[indices[0]:indices[1]]
            nfinite_tensors = np.flatnonzero(np.asarray(tmp) > 0)

            if len(nfinite_tensors) > 0:
                nfinite_tensors = sorted([self._reverse_tensors_mapping[i]
                                          for i in nfinite_tensors])
                print("Not finite values found in following nodes: "
                      "(NodeID, TensorID) - {}".format(nfinite_tensors))

        if self._reverse_keep_tensors:
            indices = self._debug_tensors_indices["keep"]
            tmp = debug_values[indices[0]:indices[1]]
            tmp = sorted([(self._reverse_tensors_mapping[i], v)
                          for i, v in enumerate(tmp)])
            self._reversed_tensors = tmp

    def _handle_propagation(self,
                       model,
                       relevance_clusters,
                       stop_analysis_at_tensors=[],
                       return_all_reversed_tensors=True):
        return kgraph.forward_model(
            model,
            relevance_clusters,
            reverse_mappings=self._reverse_mapping,
            default_reverse_mapping=self._default_reverse_mapping,
            head_mapping=self._head_mapping,
            stop_mapping_at_tensors=stop_analysis_at_tensors,
            verbose=self._reverse_verbose,
            clip_all_reversed_tensors=self._reverse_clip_values,
            project_bottleneck_tensors=self._reverse_project_bottleneck_layers,
            return_all_reversed_tensors=return_all_reversed_tensors)

    def _pendulum(self, model, the_label_index, stop_analysis_at_tensors=[], mask=[], return_all_reversed_tensors=True):

        return_all_reversed_tensors = (
                self._reverse_check_min_max_values or
                self._reverse_check_finite or
                self._reverse_keep_tensors
        )

        ret = self._forward_model(
            model,
            the_label_index,
            stop_analysis_at_tensors=[],
            return_all_reversed_tensors=True,
            mask=mask
        )

        if return_all_reversed_tensors:
            ret = (self._postprocess_analysis(ret[0]), ret[1])
        else:
            ret = self._postprocess_analysis(ret)

        if return_all_reversed_tensors:
            debug_tensors = []
            self._debug_tensors_indices = {}

            values_grad = list(six.itervalues(ret[1]))
            mapping = {i: v["id"] for i, v in enumerate(values_grad)}
            self.tensors = [v["final_tensor"] for v in values_grad]
            self._reverse_tensors_mapping = mapping
            # tensors = ret[1]

            if self._reverse_check_min_max_values:
                tmp = [ilayers.Min(None)(x) for x in self.tensors]
                self._debug_tensors_indices["min"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp))
                debug_tensors += tmp

                tmp = [ilayers.Max(None)(x) for x in self.tensors]
                self._debug_tensors_indices["max"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp))
                debug_tensors += tmp

            if self._reverse_check_finite:
                tmp = iutils.to_list(ilayers.FiniteCheck()(self.tensors))
                self._debug_tensors_indices["finite"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(tmp))
                debug_tensors += tmp

            if self._reverse_keep_tensors:
                self._debug_tensors_indices["keep"] = (
                    len(debug_tensors),
                    len(debug_tensors) + len(self.tensors))
                debug_tensors += self.tensors

            ret = (ret[0], debug_tensors)
        return ret

    def _get_state(self):
        state = super(ReverseAnalyzerBase, self)._get_state()
        state.update({"reverse_verbose": self._reverse_verbose})
        state.update({"reverse_clip_values": self._reverse_clip_values})
        state.update({"reverse_project_bottleneck_layers":
                      self._reverse_project_bottleneck_layers})
        state.update({"reverse_check_min_max_values":
                      self._reverse_check_min_max_values})
        state.update({"reverse_check_finite": self._reverse_check_finite})
        state.update({"reverse_keep_tensors": self._reverse_keep_tensors})
        state.update({"reverse_reapply_on_copied_layers":
                      self._reverse_reapply_on_copied_layers})
        return state

    @classmethod
    def _state_to_kwargs(clazz, state):
        reverse_verbose = state.pop("reverse_verbose")
        reverse_clip_values = state.pop("reverse_clip_values")
        reverse_project_bottleneck_layers = (
            state.pop("reverse_project_bottleneck_layers"))
        reverse_check_min_max_values = (
            state.pop("reverse_check_min_max_values"))
        reverse_check_finite = state.pop("reverse_check_finite")
        reverse_keep_tensors = state.pop("reverse_keep_tensors")
        reverse_reapply_on_copied_layers = (
            state.pop("reverse_reapply_on_copied_layers"))
        kwargs = super(ReverseAnalyzerBase, clazz)._state_to_kwargs(state)
        kwargs.update({"reverse_verbose": reverse_verbose,
                       "reverse_clip_values": reverse_clip_values,
                       "reverse_project_bottleneck_layers":
                       reverse_project_bottleneck_layers,
                       "reverse_check_min_max_values":
                       reverse_check_min_max_values,
                       "reverse_check_finite": reverse_check_finite,
                       "reverse_keep_tensors": reverse_keep_tensors,
                       "reverse_reapply_on_copied_layers":
                       reverse_reapply_on_copied_layers})
        return kwargs