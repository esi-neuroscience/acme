.. Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
.. in Cooperation with Max Planck Society

.. SPDX-License-Identifier: CC-BY-NC-SA-1.0

:orphan:

Functional Connectivity Tutorial
--------------------------------

.. note::
    These examples were run on the ESI HPC cluster. This is why we use
    :func:`~acme.esi_cluster_setup` to set up a parallel computing client.
    They are perfectly reproducible on any other cluster or local machine
    by instead using :func:`~acme.slurm_cluster_setup` or :func:`~acme.local_cluster_setup`
    respectively.

The following Python code demonstrates how to use ACME to concurrently compute the
`functional connectivity <https://en.wikipedia.org/wiki/Resting_state_fMRI#Functional>`_
in fMRI datasets of multiple subjects.

We start by downloading a publicly available fMRI dataset comprising
`BOLD <https://en.wikipedia.org/wiki/Blood-oxygen-level-dependent_imaging>`_
signals from 33 adults using the
`Nilearn <https://nilearn.github.io/stable/index.html>`_ package:

.. code-block:: python

    from nilearn import datasets

    data = datasets.fetch_development_fmri(age_group="adult")

The brain regions of interest (ROI) are taken from
`Power et al. 2011 <https://doi.org/10.1016/j.neuron.2011.09.006>`_ comprising
an atlas of 264 areas.

.. code-block:: python

    from nilearn import plotting

    atlas = datasets.fetch_coords_power_2011(legacy_format=False)

Next, we extract ROI-averaged time-series from the downloaded fMRI data.
To do this, we first use the
`MNI <https://en.wikipedia.org/wiki/Talairach_coordinates#Montreal_Neurological_Institute_(MNI)_templates>`_
coordinates of every ROI in the atlas as seeds for masking the fMRI data:

.. code-block:: python

    from nilearn.maskers import NiftiSpheresMasker

    atlasCoords = np.vstack((atlas.rois['x'],
                             atlas.rois['y'],
                             atlas.rois['z'])).T
    masker = NiftiSpheresMasker(seeds=atlasCoords, smoothing_fwhm=6,
                                radius=5., detrend=True,
                                standardize=True,
                                low_pass=0.1, high_pass=0.01, t_r=2)

Next, we pick a random subject (Number 0) and perform the signal extraction
removing nuisance regressors shipped with the fMRI dataset

.. code-block:: python

    timeseries = masker.fit_transform(data.func[0],
                                      confounds=data.confounds[0])

The extracted BOLD time-courses are contained in a `(#time-points x #ROIs) = (168 x 264)` array.
We choose partial correlation as statistical dependence metric for
computing the functional connectivity across all 264 regions in our atlas.
Specifically, we compute the sparse inverse covariance matrix of the BOLD
data with a cross-validated choice of the `$\ell$1` penalty using
scikit-learn's :class:`~sklearn.covariance.GraphicalLassoCV`

.. code-block:: python

    from sklearn.covariance import GraphicalLassoCV

    estimator = GraphicalLassoCV()
    estimator.fit(timeseries)

To visualize the computed inter-areal connectivity of Subject #0, we use
Nilearns :func:`~nilearn.plotting.plot_connectome`

.. code-block:: python

    from nilearn import plotting

    plotting.plot_connectome(con, atlasCoords, title=f"Subject #{subIdx}",
                             edge_threshold="95%", node_size=20,
                             colorbar=True, edge_vmin=-1, edge_vmax=1,
                             figure=fig)

.. image:: _static/subject0.png
    :width: 600
    :align: center
    :alt: Subject 0 FC

Now to parallelize...
