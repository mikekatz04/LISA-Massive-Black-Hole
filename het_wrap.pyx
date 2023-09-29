import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "Declarations.h":
    cdef cppclass HetLikeWrapper "HetLikeWrap":
        HetLikeWrapper(double *init_params, double Tobs_, double dt_, bool is_noise_free, int seg_, int rep_, double *premove, int NS);
        void dealloc();
        void udpate_heterodyne(double *params) except+
        double get_ll(double *params) except+


cdef class pyHetLikeWrap:
    cdef HetLikeWrapper *f

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float64_t] init_params, Tobs, dt, is_noise_free, seg, rep, np.ndarray[ndim=1, dtype=np.float64_t] premove, NS):
        self.f = new HetLikeWrapper(&init_params[0], Tobs, dt, is_noise_free, seg, rep, &premove[0], NS)

    def get_like(self, np.ndarray[ndim=1, dtype=np.float64_t] params):
        return self.f.get_ll(&params[0])

    def udpate_heterodyne(self, np.ndarray[ndim=1, dtype=np.float64_t] params):
        self.f.udpate_heterodyne(&params[0])

    def __dealloc__(self):
        self.f.dealloc()
        if self.f:
            del self.f
