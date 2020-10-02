from __future__ import print_function, division, absolute_import
import contextlib
from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions

from numba.core import dispatcher, utils, typing, cpu
from .target import DPPLTargetContext, DPPLTypingContext, DPPLCPUTargetContext
from numba.core.compiler_lock import global_compiler_lock



class DPPLTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    #typingctx = DPPLTypingContext()
    #targetctx = DPPLTargetContext(typingctx)

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPLTargetContext(self.typing_context)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return DPPLTypingContext()

    @property
    def target_context(self):
        """
        The target context for DPPL targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for DPPL targets.
        """
        return self._toplevel_typing_context

# The global DPPL target
dppl_target = DPPLTarget()



class _DPPLNestedContext(object):
    _typing_context = None
    _target_context = None

    @contextlib.contextmanager
    def nested(self, typing_context, target_context):
        old_nested = self._typing_context, self._target_context
        try:
            self._typing_context = typing_context
            self._target_context = target_context
            yield
        finally:
            self._typing_context, self._target_context = old_nested


class DPPLCPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    _nested = _DPPLNestedContext()

    @utils.cached_property
    @global_compiler_lock
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPLCPUTargetContext(self.typing_context)

    @utils.cached_property
    @global_compiler_lock
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return typing.Context()

    @property
    @global_compiler_lock
    def target_context(self):
        """
        The target context for CPU targets.
        """
        nested = self._nested._target_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_target_context

    @property
    @global_compiler_lock
    def typing_context(self):
        """
        The typing context for CPU targets.
        """
        nested = self._nested._typing_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_typing_context

    @global_compiler_lock
    def nested_context(self, typing_context, target_context):
        """
        A context manager temporarily replacing the contexts with the
        given ones, for the current thread of execution.
        """
        return self._nested.nested(typing_context, target_context)


# The global CPU target
dppl_cpu_target = DPPLCPUTarget()


