#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "model.h"
#include "model_methods.h"
#include "truncated_normal.h"

namespace py = pybind11;
using uint = unsigned int;

PYBIND11_MODULE(cpp_modules, m) {
    m.def("truncated_normal_ab_sample", &truncated_normal_ab_sample);
    m.def("generateRandom32bitInt", &generateRandom32bitInt);
    py::class_<Model>(m, "Model")
        .def(py::init<uint, uint, uint, uint>())
        .def("sim", &Model::sim)
        .def("sim_lif", &Model::sim_lif)
        .def("getFR", &Model::getFR)
        .def("setParams", &Model::setParams)
        .def("getState", &Model::getState)
        .def("saveDSPTS", &Model::saveDSPTS)
        .def("saveX", &Model::saveX)
        .def("saveSpts", &Model::saveSpts)
        .def("loadDSPTS", &Model::loadDSPTS)
        .def("loadX", &Model::loadX)
        .def("loadSpts", &Model::loadSpts)
        .def("increment_array", &Model::increment_array)
        .def_readwrite("frozens", &Model::frozens)            // Direct access
        .def_readwrite("F", &Model::F)                        // Direct access
        .def_readwrite("D", &Model::D)                        // Direct access
        .def_readwrite("FF", &Model::FF)                      // Direct access
        .def_readwrite("DD", &Model::DD)                      // Direct access
        .def_readwrite("Jo", &Model::Jo)                      // Direct access
        .def_readwrite("mex", &Model::mex)                    // Direct access
        .def_readwrite("Jmin", &Model::Jmin)                  // Direct access
        .def_readwrite("Jmax", &Model::Jmax)                  // Direct access
        .def_readwrite("STDPon", &Model::STDPon)              // Direct access
        .def_readwrite("homeostatic", &Model::homeostatic)    // Direct access
        .def_readwrite("use_thetas", &Model::use_thetas)      // Direct access
        .def_readwrite("saveflag", &Model::saveflag)          // Direct access
        .def_readwrite("Jinidx", &Model::Jinidx)              // Direct access
        .def_readwrite("Jinidy", &Model::Jinidy)              // Direct access
        .def_readwrite("F", &Model::F)                        // Direct access
        .def_readwrite("D", &Model::D)                        // Direct access
        .def_readwrite("UU", &Model::UU)                      // Direct access
        .def_readwrite("stp_on_I", &Model::stp_on_I)          // Direct access
        .def_readwrite("use_thetas", &Model::use_thetas)      // Direct access
        .def_readwrite("theta", &Model::theta)                // Direct access
        .def_readwrite("hStim", &Model::hStim)                // Direct access
        .def_readwrite("Uexc", &Model::Uexc)                  // Direct access
        .def_readwrite("Uinh", &Model::Uinh)                  // Direct access
        .def_readwrite("t", &Model::t)                        // Direct access
        .def_readwrite("soft_clip_dw", &Model::soft_clip_dw)  // Direct access
        .def_readwrite("dump_dw", &Model::dump_dw)            // Direct access
        .def_readwrite("dump_xy", &Model::dump_xy)            // Direct access
        .def_readwrite("datafolder", &Model::datafolder)      // Direct access

        .def_readwrite("dspts", &Model::dspts)  // Direct access
        .def_readwrite("spts", &Model::spts)    // Direct access
        .def_readwrite("x", &Model::x)          // Direct access

        // return a member of a child object using a func in the parent
        // .def("getUexc", &Model::getUexc, py::return_value_policy::reference_internal)
        // .def("getUinh", &Model::getUinh, py::return_value_policy::reference_internal)
        .def_readwrite("stimIntensity", &Model::stimIntensity);  // Direct access
}