.. Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience in Cooperation with Max Planck Society

.. SPDX-License-Identifier: BSD-3-Clause

Best Practices
==============
Using ACME inside a Python script requires some precautions. Please follow this
guide in case you're experiencing strange ``RuntimeError`` or recursion exceptions.

.. _mainblock:

Use a ``__main__`` Block
------------------------
First, please ensure
your script contains a ``main`` module block that defines a central entrypoint for
the Python interpreter. That means instead of just lining up code blocks one after another in a file,
encapsulate those parts of the code, that are supposed to be executed only once
inside a ``__main__`` block. Thus, if your script looks like this

``script_nomain.py``

.. code-block:: python

    # Do something, then otherthing

    import package1
    import package2

    def do_this(...):
        ...
        return something

    def do_that(...):
        ...
        return otherthing

    x = ...
    y = ...

    z = do_this(...x,y,...)
    ...
    w = do_that(...,z,...)

re-structure it using a ``__main__`` block to clearly delineate which parts are
intended to be executed *just once*:

``script_withmain.py``

.. code-block:: python

    # Do something, then otherthing

    import package1
    import package2

    def do_this(...):
        ...
        return something

    def do_that(...):
        ...
        return otherthing

    if __name__ == "__main__":
        x = ...
        y = ...

        z = do_this(...x,y,...)
        ...
        w = do_that(...,z,...)

If ``script_withmain.py`` is called directly (e.g., via ``run script_withmain.py``
in iPython) the interpreter uses the main module block as its entry
point of code execution.


.. _isolation:

Isolate Your Function
---------------------
To avoid any recursive importing errors or accidental spawning of workers by
workers, it is further strongly recommended to capsulate your function `func` in
a dedicated file separate from the :class:`~acme.ParallelMap` call. Thus, instead of using a
single script

``my_script.py``

.. code-block:: python

    # My processing script

    import numpy as np
    from acme import ParallelMap

    def func(x, y, z=3):
        return np.dot(x, y) + z

    if __name__ == "__main__":
        y = np.array([[4, 1], [2, 2]])
        x1 = np.arange(4).reshape(2,2)
        x2 = np.arange(4, 8).reshape(2,2)
        with ParallelMap(func, [x1, x2], y) as pmap:
            results = pmap.compute()

split up the definition of `func` and its ACME parallelization:

``my_func.py``:

.. code-block:: python

    # My processing function

    import numpy as np

    def func(x, y, z=3):
        return np.dot(x, y) + z

``acme_script.py``:

.. code-block:: python

    # My ACME script for func

    import numpy as np
    from acme import ParallelMap
    from my_func import func

    if __name__ == "__main__":
        y = np.array([[4, 1], [2, 2]])
        x1 = np.arange(4).reshape(2,2)
        x2 = np.arange(4, 8).reshape(2,2)
        with ParallelMap(func, [x1, x2], y) as pmap:
            results = pmap.compute()

Then simply launching ``acme_script.py`` via iPython does the trick:

.. code-block:: python

    >>> run acme_script.py

**Note** Just like any regular Python module, ``my_func.py`` permits to define
several distinct functions. This means, if, e.g., `func` requires additional helper
routines, they can all be migrated to ``my_func.py``. For instance:

``my_func.py``:

.. code-block:: python

    # My processing function

    import numpy as np

    def func(x, y, z=3):
        help_here(...)
        help_there(...)
        return np.dot(x, y) + z

    def help_here(...):
        ...

    def help_there(...):
        ...

