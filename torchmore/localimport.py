#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import inspect


class LocalImport(object):

    def __init__(self, names):
        if not isinstance(names, dict):
            names = vars(names)
        self.names = names

    def __enter__(self):
        self.frame = inspect.currentframe()
        bindings = self.frame.f_back.f_globals
        self.old_bindings = {k: bindings.get(
            k, None) for k in list(self.names.keys())}
        bindings.update(self.names)

    def __exit__(self, some_type, value, traceback):
        del some_type, value, traceback
        bindings = self.frame.f_back.f_globals
        bindings.update(self.old_bindings)
        extras = [k for k, v in list(self.old_bindings.items()) if v is None]
        for k in extras:
            del bindings[k]
        del self.frame
