import time
cimport numpy as np

np.import_array()

# https://stackoverflow.com/questions/42125084/accessing-opencv-cuda-functions-from-python-no-pycuda
def cudaWarpPerspectiveWrapper(np.ndarray[np.uint8_t, ndim=2] _src,
                               np.ndarray[np.float32_t, ndim=2] _M,
                               _size_tuple,
                               int _flags):
    # Create GPU/device InputArray for src
    cdef Mat src_mat
    cdef GpuMat src_gpu
    pyopencv_to(<PyObject*> _src, src_mat)
    src_gpu.upload(src_mat)

    # Create CPU/host InputArray for M
    cdef Mat M_mat = Mat()
    pyopencv_to(<PyObject*> _M, M_mat)

    # Create Size object from size tuple
    # Note that size/shape in Python is handled in row-major-order -- therefore, width is [1] and height is [0]
    cdef Size size = Size(<int> _size_tuple[1], <int> _size_tuple[0])

    # Create empty GPU/device OutputArray for dst
    cdef GpuMat dst_gpu = GpuMat()
    warpPerspective(src_gpu, dst_gpu, M_mat, size, _flags)

    # Get result of dst
    cdef Mat dst_host
    dst_gpu.download(dst_host)
    cdef np.ndarray out = <np.ndarray> pyopencv_from(dst_host)
    return out
