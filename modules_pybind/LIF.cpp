#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>  // For std::pair
#include <vector>

namespace py = pybind11;
using uint = unsigned int;
using namespace std;

static std::map<uint, uint> Map = {{1, 0}, {2, 80}, {3, 160}, {4, 240}, {5, 320}};

static uint seed_counter = 0;
static double EPSILON = 0.001;
static double Jtmin = 0.05;  // range for the mean presynaptic weight
static double Jtmax = 0.15;  // range for the mean presynaptic weight
static double Jmax = 0.95;   // max allowed weight
static double Jmin = 0.01;   // min allowed weight

static double alpha = 50.0;  // hard set
static double JEE = 0.45;    // hard set

class Sphist {
   public:
    vector<double> dvec;
    vector<vector<double>> sphist;
    double last_trimmmed_at = 0.0;
    ofstream ofsr;

    // constructor
    Sphist(uint N) {
        for (uint i = 0; i < N; ++i) {
            sphist.push_back(dvec);
        }
        string fname = "spiketimes";
        ofsr.open(fname);
        ofsr.precision(10);
    }

    void push(uint i, double t) {
#pragma omp critical
        {
            sphist[i].push_back(t);
            ofsr << i << " " << t << endl;
            ;
        }

        // every 100 ms, forget spikes older than 500 ms
        if ((t - last_trimmmed_at) > 100.0) {
            last_trimmmed_at = t;
            sphist[i].erase(sphist[i].begin(),
                            std::lower_bound(sphist[i].begin(), sphist[i].end(), t - 500.0));
        }
    }
};

class Neuron {
   public:
    uint neuronID;
    Sphist* sphist;

    // constants

    double tau;
    double I_E = 0.0;
    double I_I = 0.0;
    double V = -65.0;

    double t = 0.0;
    double* AP;
    double* AMPA;
    double* NMDA;
    double* GABA;
    double* Jo;
    double* F;
    double* D;

    uint NE;
    uint NI;
    uint N;

    double twnd = 500;
    bool HAGA = false;

    double tpp = 15.0;
    double tpd = 120.0;

    double Cp = 0.14;
    double Cd = 0.07;

    double taustf = 250.0;  // !
    double taustd = 350.0;  // NOTE:

    double V_E = 0.0;
    double V_I = -80.0;
    double EL = -65.0;

    double Vth = -55.0;
    double Vr = -70.0;
    double Vspike = 10.0;

    double tau_ampa = 8.0;
    double tau_nmda = 16.0;
    double tau_gaba = 8.0;

    double refractory_period = 2.0;
    double transmission_delay = 0.0;
    double in_refractory = 0.0;

    double h = 0.01;
    double UU = 0.6;

    Neuron(uint i, uint _NE, uint _NI, py::array_t<double> _AMPA, py::array_t<double> _NMDA,
           py::array_t<double> _GABA, py::array_t<double> _AP, py::array_t<double> _Jo,
           py::array_t<double> _F, py::array_t<double> _D, Sphist* _sphist)
        : m_mt(++seed_counter) {
        NE = _NE;
        NI = _NI;
        N = NE + NI;

        auto AMPAbuf = _AMPA.request();
        AMPA = static_cast<double*>(AMPAbuf.ptr);

        auto NMDAbuf = _NMDA.request();
        NMDA = static_cast<double*>(NMDAbuf.ptr);

        auto GABAbuf = _GABA.request();
        GABA = static_cast<double*>(GABAbuf.ptr);

        auto APbuf = _AP.request();
        AP = static_cast<double*>(APbuf.ptr);

        auto Jobuf = _Jo.request();
        Jo = static_cast<double*>(Jobuf.ptr);

        auto Fbuf = _F.request();
        F = static_cast<double*>(Fbuf.ptr);

        auto Dbuf = _D.request();
        D = static_cast<double*>(Dbuf.ptr);

        sphist = _sphist;
        neuronID = i;
        if (neuronID < NE) {
            tau = 10.0;
        } else {
            tau = 20.0;
        }
        // randomize initial voltages
        // V += 9.0 * dice();
        // V -= 9.0 * dice();
    }

    void step(double, bool, bool);
    void normalize_EEweights();

    void push(double t) {
        sphist->push(neuronID, t);
        ;
    }

    void clip_pos(uint i, uint j) {
        if (Jo[i * N + j] < Jmin) Jo[i * N + j] = Jmin;
        if (Jo[i * N + j] > Jmax) Jo[i * N + j] = Jmax;
    }

    void clip_neg(uint i, uint j) {
        if (Jo[i * N + j] > -Jmin) Jo[i * N + j] = -Jmin;
        if (Jo[i * N + j] < -Jmax) Jo[i * N + j] = -Jmax;
    }

   private:
    std::random_device m_randomdevice;
    std::mt19937 m_mt;

    double dice() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(m_mt);
    }

    void symSTDP();
    void asymSTDP();
    double fd(double);

    double log_base(double x, double base) {
        return log(x) / log(base);  // Using natural logarithm (base e)
    }
};

void Neuron::step(double prob_stim, bool plasticity_on, bool symmetric) {
    I_E = 0.0;
    I_I = 0.0;
    uint i = neuronID * N;

    for (uint j = 0; j < N; ++j) {
        if (abs(Jo[i + j]) < EPSILON) {
            continue;
        }
        AP[i + j] -= h;
        if (AP[i + j] < 0.0) {
            AP[i + j] = 10000.0;  // confirm receipt of spike from jth neuron
            if (j < NE) {
                AMPA[i + j] += F[j] * D[j] * Jo[i + j];
                NMDA[i + j] += F[j] * D[j] * Jo[i + j];
            } else {
                GABA[i + j] += F[j] * D[j] * abs(Jo[i + j]);
            }
        }
        // anyway
        if (j < NE) {
            AMPA[i + j] -= (AMPA[i + j] / tau_ampa) * h;
            NMDA[i + j] -= (NMDA[i + j] / tau_nmda) * h;
        } else {
            GABA[i + j] -= (GABA[i + j] / tau_gaba) * h;
        }

        I_E += -(AMPA[i + j] * (V - V_E) + 0.1 * NMDA[i + j] * (V - V_E));
        I_I += GABA[i + j] * (V - V_I);
    }

    // I_E += driving_I * dice();
    if (in_refractory < 0.0) {
        if (neuronID < NE) {
            if (dice() < prob_stim) {
                V = Vth + 1.0;
            }
        }
    }

    if (in_refractory < 0.0) {
        double dV = (-(V - EL) / tau + I_E - I_I) * h;
        V += dV;
    }

    if ((V > Vth) && (in_refractory < 0.0)) {
        V = Vr;
        in_refractory = refractory_period + dice();

        // "send" spikes to all postsynaptic neurons
        for (uint row = 0; row < N; ++row) {
            AP[row * N + neuronID] = transmission_delay;  // + transmission delay
        }

        F[neuronID] += UU * (1 - F[neuronID]);
        D[neuronID] -= D[neuronID] * F[neuronID];
        push(t);
        if (plasticity_on) {
            if (symmetric) {
                symSTDP();
            } else {
                asymSTDP();
            }
            normalize_EEweights();
        }
    }

    // EVERY 10 timesteps STP on no spike // NOTE: !!!
    if (((uint)floor(t / h)) % 10 == 0) {
        F[neuronID] += h * 10.0 * (UU - F[neuronID]) / taustf;
        D[neuronID] += h * 10.0 * (1.0 - D[neuronID]) / taustd;
    }

    t += h;
    in_refractory -= h;
}

double Neuron::fd(double x) { return log(1.0 + alpha / JEE * abs(x)) / log(1.0 + alpha); }

void Neuron::symSTDP() {
    // only EE for now
    uint i = neuronID;
    double dw;
    if (i >= NE) {
        return;
    }
    for (uint ip = 0; ip < NE; ++ip) {
        if (abs(Jo[ip * N + i]) < EPSILON) {
            continue;
        }
        for (auto tt : sphist->sphist[ip]) {
            dw = Cp * exp(-(t - tt) / tpp) - fd(Jo[ip * N + i]) * Cd * exp(-(t - tt) / tpd);
            if (HAGA) {
                dw *= F[i] * D[i];
            }

            if (i < NE) {  // if presynaptic is excitatory
                Jo[ip * N + i] += dw;
                clip_pos(ip, i);
            } else {
                Jo[ip * N + i] -= dw;
                clip_neg(ip, i);
            }
        }
    }

    for (uint j = 0; j < NE; ++j) {
        if (abs(Jo[i * N + j]) < EPSILON) {
            continue;
        }
        for (auto tt : sphist->sphist[j]) {
            dw = Cp * exp(-(t - tt) / tpp) - fd(Jo[i * N + j]) * Cd * exp(-(t - tt) / tpd);
            if (HAGA) {
                dw *= F[j] * D[j];
            }

            if (j < NE) {  // if presynaptic is excitatory
                Jo[i * N + j] += dw;
                clip_pos(i, j);
            } else {
                Jo[i * N + j] -= dw;
                clip_neg(i, j);
            }
        }
    }
}

void Neuron::asymSTDP() {
    // only EE for now
    uint i = neuronID;
    double dw;
    if (i >= NE) {
        return;
    }
    for (uint ip = 0; ip < NE; ++ip) {
        if (abs(Jo[ip * N + i]) < EPSILON) {
            continue;
        }
        for (auto tt : sphist->sphist[ip]) {
            dw = -Cp * exp((tt - t) / tpd);
            if (HAGA) {
                dw *= F[i] * D[i];
            }
            if (i < NE) {  // if presynaptic is excitatory
                Jo[ip * N + i] += dw;
                clip_pos(ip, i);
            } else {
                Jo[ip * N + i] -= dw;
                clip_neg(ip, i);
            }
        }
    }

    for (uint j = 0; j < NE; ++j) {
        if (abs(Jo[i * N + j]) < EPSILON) {
            continue;
        }
        for (auto tt : sphist->sphist[j]) {
            dw = Cp * exp(-(t - tt) / tpp);
            if (HAGA) {
                dw *= F[j] * D[j];
            }
            if (j < NE) {  // if presynaptic is excitatory
                Jo[i * N + j] += dw;
                clip_pos(i, j);
            } else {
                Jo[i * N + j] -= dw;
                clip_neg(i, j);
            }
        }
    }
}

void Neuron::normalize_EEweights() {
    // double target_sum = 6.0;
    // float sum = 0.0;
    // uint i = neuronID * N;
    // if (i >= NE) {
    //     return;
    // }

    // double base = 0.10;

    // for (uint j = 0; j < NE; ++j) {
    //     if (Jo[i + j] > EPSILON) {
    //         Jo[i + j] = log_base(Jo[i + j], base);
    //         clip_pos(i, j);
    //     }
    // }

    // for (uint j = 0; j < NE; ++j) {
    //     if (Jo[i + j] > EPSILON) {
    //         sum += Jo[i + j];
    //     }
    // }
    // double excess_factor = sum / target_sum;

    // for (uint j = 0; j < NE; ++j) {
    //     if (Jo[i + j] > EPSILON) {
    //         Jo[i + j] /= excess_factor;
    //         clip_pos(i, j);
    //     }
    // }
}

class Net {
   public:
    uint NE;
    uint NI;
    uint N;

    Net(uint _NE, uint _NI, py::array_t<double> _AMPA, py::array_t<double> _NMDA,
        py::array_t<double> _GABA, py::array_t<double> _AP, py::array_t<double> _Jo,
        py::array_t<double> _F, py::array_t<double> _D, Sphist* _sphist)
        : m_mt(m_randomdevice()) {
        NE = _NE;
        NI = _NI;
        N = NE + NI;
        for (uint i = 0; i < N; ++i) {
            neurons.push_back(
                // construct an object and remember a unique pointer to it
                std::make_unique<Neuron>(i, NE, NI, _AMPA, _NMDA, _GABA, _AP, _Jo, _F, _D,
                                         _sphist));
        }
    }

    double Jo_lookup(uint i, uint j) {
        return neurons[0]->Jo[i * N + j];
        ;
    }

    vector<double> get_V() {
        vector<double> tmp;
        for (auto& neuron : neurons) {
            tmp.push_back(neuron->V);
        }
        return tmp;
    }

    vector<double> get_IE() {
        vector<double> tmp;
        for (auto& neuron : neurons) {
            tmp.push_back(neuron->I_E);
        }
        return tmp;
    }

    vector<double> get_II() {
        vector<double> tmp;
        for (auto& neuron : neurons) {
            tmp.push_back(neuron->I_I);
        }
        return tmp;
    }

    double get_t(uint i) { return neurons[i]->t; }

    void sim(uint num_steps, double prob_stim, bool plasticity_on, bool symmetric, uint stimID) {
        for (uint i = 0; i < num_steps; ++i) {
            uint _from = Map[stimID];
            uint _to = _from + 80;
#pragma omp parallel for
            for (auto& neuron : neurons) {
                if (stimID != 0) {
                    if ((neuron->neuronID >= _from) && (neuron->neuronID < _to)) {
                        neuron->step(prob_stim * 2.0, plasticity_on, symmetric);
                    } else {
                        neuron->step(prob_stim, plasticity_on, symmetric);
                    }

                } else {
                    neuron->step(prob_stim, plasticity_on, symmetric);
                }
            }
        }

        for (uint i = 0; i < NE; i++) {
            double Jav = 0.0;
            uint c = 0;
            for (uint j = 0; j < NE; j++) {
                if (neurons[i]->Jo[i * N + j] > EPSILON) {
                    Jav += neurons[i]->Jo[i * N + j];
                    c += 1;
                }
            }

            Jav = Jav / ((double)c);  // find mean weight per each postsynaptic neuron

            for (uint j = 0; j < NE; j++) {
                if (neurons[i]->Jo[i * N + j] > EPSILON) {
                    if (Jav > Jtmax)
                        neurons[i]->Jo[i * N + j] -= abs(Jav - Jtmax);  // subtract excess
                    if (Jav < Jtmin) neurons[i]->Jo[i * N + j] += abs(Jav - Jtmin);  // add shortage
                    neurons[i]->clip_pos(i, j);
                }
            }
        }
    }

   private:
    vector<std::unique_ptr<Neuron>> neurons;
    std::random_device m_randomdevice;
    std::mt19937 m_mt;

    double dice() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(m_mt);
    }
};

PYBIND11_MODULE(LIF, m) {
    py::class_<Sphist>(m, "Sphist")
        .def(py::init<uint>())
        .def("push", &Sphist::push)
        .def_readwrite("sphist", &Sphist::sphist);
    py::class_<Neuron>(m, "Neuron")
        .def(py::init<uint, uint, uint, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, Sphist*>())
        .def("push", &Neuron::push)
        .def("step", &Neuron::step)
        .def_readwrite("V", &Neuron::V)
        .def_readwrite("in_refractory", &Neuron::in_refractory)
        .def_readwrite("I_E", &Neuron::I_E)
        .def_readwrite("I_I", &Neuron::I_I)
        .def_readwrite("i", &Neuron::neuronID)
        .def_readwrite("t", &Neuron::t)
        .def("normalize_EEweights", &Neuron::normalize_EEweights);
    py::class_<Net>(m, "Net")
        .def(py::init<uint, uint, py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, Sphist*>())
        .def("get_V", &Net::get_V)
        .def("get_II", &Net::get_II)
        .def("get_IE", &Net::get_IE)
        .def("sim", &Net::sim)
        .def("get_t", &Net::get_t);
}