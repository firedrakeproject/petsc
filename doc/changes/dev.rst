====================
Changes: Development
====================

..
   STYLE GUIDELINES:
   * Capitalize sentences
   * Use imperative, e.g., Add, Improve, Change, etc.
   * Don't use a period (.) at the end of entries
   * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

.. rubric:: General:

.. rubric:: Configure/Build:

.. rubric:: Sys:

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

- The ``IS`` passed to ``VecISAXPY()``, ``VecISCopy()``. ``VecISSet()``, and ``VecISShift()`` must have the same communicator of the vectors used
- Make ``VecLock`` API active in optimized mode
- ``VecNestSetSubVec()`` and ``VecNestSetSubVecs()`` now take references to input vectors rather than creating duplicates

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add support in ``PCFieldSplitSetFields()`` including with ``-pc_fieldsplit_%d_fields fields`` for ``MATNEST``,  making it possible to
  utilize multiple levels of ``PCFIELDSPLIT`` with ``MATNEST`` from the command line
- Add ``PCCompositeSpecialSetAlphaMat()`` API to use a matrix other than the identity in
  preconditioners based on an alternating direction iteration, e.g., setting :math:`M` for
  :math:`P = (A + alpha M) M^{-1} (alpha M + B)`

- Change the option database keys for coarsening for ``PCGAMG`` to use the prefix ``-pc_gamg_``, for example ``-pc_gamg_mat_coarsen_type``

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add Rosenbrock-W methods from :cite:`rang2015improved` with :math:`B_{PR}` stability: ``TSROSWR34PRW``, ``TSROSWR3PRL2``, ``TSROSWRODASPR``, and ``TSROSWRODASPR2``

.. rubric:: TAO:

.. rubric:: DM/DA:

- Add ``DMGetSparseLocalize()`` and ``DMSetSparseLocalize()``
- Add ``DMGeomModelRegister()``, ``DMGeomModelRegisterAll()``, ``DMGeomModelRegisterDestroy()``, ``DMSnapToGeomModel()``, ``DMSetSnapToGeomModel()`` to support registering geometric models
- Add ``DMGetOutputSequenceLength()``

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMLabelGetValueBounds()``
- Add ``DMPlexOrientLabel()``
- Add an argument to ``DMPlexLabelCohesiveComplete()`` in order to change behavior at surface boundary
- Remove ``DMPlexSnapToGeomModel()``
- Add refinement argument to ``DMPlexCreateHexCylinderMesh()``
- Now ``DMPlexComputeBdIntegral()`` takes one function per field
- Add ``DMPlexCreateEdgeNumbering()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:

- Add ``PETSC_NULL_ENUM`` to be used instead of ``PETSC_NULL_INTEGER`` when a pointer to an ``enum`` is expected in a PETSc function call
- Add ``PETSC_NULL_INTEGER_ARRAY``, ``PETSC_NULL_SCALAR_ARRAY``, and ``PETSC_NULL_REAL_ARRAY`` for use instead of
  ``PETSC_NULL_INTEGER``, ``PETSC_NULL_SCALAR``,  and ``PETSC_NULL_REAL`` when an array is expected in a PETSc function call
- Add automatically generated interface definitions for most PETSc functions to detect illegal usage at compile time
- Add ``PetscObjectIsNull()`` for users to check if a PETSc object is ``NULL``
- Change the PETSc Fortran API so that non-array values, ``v``, passed to PETSc routines expecting arrays must be cast with ``[v]`` in the calling sequence
