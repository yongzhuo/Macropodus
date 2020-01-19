# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/16 22:34
# @author  : Mo
# @function: dump of keras, error, no use.


from tensorflow.python.keras.models import save_model, load_model, Model
import tempfile
import types


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
