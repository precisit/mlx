.. _precision:

Numerical Precision
===================

By default, MLX may run ``float32`` matrix-multiplication family
operations (matmul, quantized matmul, grouped matmul, convolution and
attention) at reduced precision on hardware with dedicated
matrix-multiplication units. Inputs and outputs stay ``float32``, but
results can differ from a full-precision reference by several orders of
magnitude more than ``float32`` rounding alone would explain.

To keep these operations in full ``float32``, set
:envvar:`MLX_ENABLE_TF32` to ``0`` when launching the process:

.. code-block:: shell

  MLX_ENABLE_TF32=0 python my_script.py

Which operations take the reduced-precision path, and how large the
difference is, depends on the backend and the hardware.

8-bit Quantized Matmul
----------------------

Quantized matmuls dequantize the weights to the type of the input and
multiply in floating point. On hardware with dedicated
matrix-multiplication units MLX can instead round the weights and the input
to 8 bits and multiply those, which is faster and less precise. This is off
by default.

For affine quantized weights set :envvar:`MLX_QMM_INT8`. The weights and the
input are rounded to 8-bit integers with one scale per row. With ``1`` the
scale of the input is refined for every block of 64 elements, with ``4`` for
every four blocks, and ``2`` uses the row scale alone:

.. code-block:: shell

  MLX_QMM_INT8=1 python my_script.py

For ``mxfp4`` and ``mxfp8`` weights set :envvar:`MLX_QMM_FP8`. The weights
and the input are multiplied as 8-bit floating point numbers. This needs
Metal 4.1, a build of MLX that targets macOS 27 or later:

.. code-block:: shell

  MLX_QMM_FP8=1 python my_script.py

Expect a relative difference of one to a few percent from the floating point
result. Only products of 64 rows or more with transposed weights take these
paths. Both variables are read each time a quantized matmul is built.
