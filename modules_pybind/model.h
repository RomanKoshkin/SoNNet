
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
// namespace fs = std::experimental::filesystem;

string getexepath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}

// class Model;  // Forward declaration is needed because Child references Parent

// class LimDeque {
//    private:
//     deque<double> fdeque;
//     uint NE;
//     uint N;
//     Model* m;

//    public:
//     vector<deque<double>> Uinh;
//     vector<deque<double>> Uexc;
//     uint maxlen;
//     LimDeque(uint, Model*);
//     void push(uint);
//     double tmp_u;
// };

// class definition
class Model {
   public:
    // ostringstream ossSTP;
    // ofstream ofsSTP;
    double accum;
    double PI = 3.14159265;
    uint nstim;
    uint NE, NI, N, NEo;
    uint SNE, SNI;  // how many neurons get updated per time step
    uint NEa;       // Exact number of excitatory neurons stimulated externally
    uint pmax;

    vector<vector<double>> Jo;
    vector<vector<double>> Ji;

    vector<deque<double>> Uinh;
    vector<deque<double>> Uexc;
    deque<double> fdeque;

    double alpha = 50.0;  // Degree of log-STDP (50.0)
    double JEI = 0.15;    // 0.15 or 0.20

    double pi = 3.14159265;
    double e = 2.71828182;

    double T = 1800 * 1000.0;  // simulation time, ms
    double h = 0.01;           // time step, ms ??????

    // probability of connection
    double cEE = 0.2;  //
    double cIE = 0.2;  //
    double cEI = 0.5;  //
    double cII = 0.5;  //
    vector<vector<bool>> frozens;

    // Synaptic weights
    double JEE = 0.15;      //
    double JEEinit = 0.15;  // ?????????????
    double JIE = 0.15;      //
    double JII = 0.06;      //
    // initial conditions of synaptic weights
    double JEEh = 0.15;  // Standard synaptic weight E-E
    double sigJ = 0.3;   //

    double Jtmax = 0.25;  // J_maxˆtot
    double Jtmin = 0.01;  // J_minˆtot // ??? NOT IN THE PAPER

    // WEIGHT CLIPPING     // ???
    double Jmax = 5.0 * JEE;   // ???
    double Jmin = 0.01 * JEE;  // ????

    // Thresholds of update
    double hE = 1.0;  // Threshold of update of excitatory neurons
    double hI = 1.0;  // Threshold of update of inhibotory neurons

    double IEex = 2.0;   // Amplitude of steady external input to excitatory neurons
    double IIex = 0.5;   // Amplitude of steady external input to inhibitory neurons
    double mex = 0.3;    // mean of external input
    double sigex = 0.1;  // variance of external input

    // Average intervals of update, ms
    double tmE = 5.0;  // t_Eud EXCITATORY
    double tmI = 2.5;  // t_Iud INHIBITORY

    // Short-Term Depression
    double trec = 600.0;  // recovery time constant (tau_sd, p.13 and p.12)
    // double usyn = 0.1;
    double Jepsilon = 0.001;  // BEFORE UPDATING A WEIGHT, WE CHECK IF IT IS GREATER THAN
                              // Jepsilon. If smaller, we consider this connection as
                              // non-existent, and do not update the weight.

    // Time constants of STDP decay
    double tpp = 20.0;    // tau_p
    double tpd = 40.0;    // tau_d
    double twnd = 500.0;  // STDP window lenght, ms

    // Coefficients of STDP
    double Cp = 0.1 * JEE;       // must be 0.01875 (in the paper)
    double Cd = Cp * tpp / tpd;  // must be 0.0075 (in the paper)

    // homeostatic
    // double hsig = 0.001*JEE/sqrt(10.0);
    double hsig = 0.001 * JEE;  // i.e. 0.00015 per time step (10 ms)
    int itauh = 100;            // decay time of homeostatic plasticity, (100s)

    double hsd = 0.1;  // is is the timestep of integration for calculating STP
    double hh = 10.0;  // SOME MYSTERIOUS PARAMETER

    double Ip = 1.0;  // External current applied to randomly chosen excitatory neurons
    double a = 0.20;  // Fraction of neurons to which this external current is applied

    double xEinit = 0.02;  // the probability that an excitatory neurons spikes
                           // at the beginning of the simulation
    double xIinit = 0.01;  // the probability that an inhibitory neurons spikes
                           // at the beginning of the simulation
    double tinit = 100.0;  // period of time after which STDP kicks in

    double DEQUE_LEN_MS = 50.0;

    bool use_thetas = false;
    bool soft_clip_dw = false;
    bool STDPon = true;
    bool symmetric = true;
    bool homeostatic = true;
    double totalInhibW = 0.5;
    // int inhibition_mode = 0;  // whether to use normal inhibition from (I neurons) or tot inhib

    vector<double> dvec;
    vector<double> UU;

    deque<double> spdeque;         // holds a recent history of spikes of one neurons
    vector<deque<double>> sphist;  // holds neuron-specific spike histories

    vector<uint> ivec;
    deque<double> ideque;  // <<<< !!!!!!!!

    vector<deque<double>> dspts;  // <<<< !!!!!!!!
    vector<uint> x;
    set<uint> spts;

    bool saveflag = true;
    double t = 0;
    int tidx = -1;
    // bool trtof = true; // ?????? some flag
    double u;
    uint j;
    vector<uint> smpld;
    set<uint>::iterator it;
    double k1, k2, k3, k4;
    double dw;
    bool dump_dw = false;  // save STDP updates flag
    bool dump_xy = false;  // save STDP updates flag
    // bool Iptof = true;

    vector<vector<uint>> Jinidx; /* list (len=3000) of lists. Eachlist lists indices of excitatory
                                   neurons whose weights are > Jepsilon */
    vector<vector<uint>> Jinidy; /* list (len=3000) of lists. Eachlist lists indices of excitatory
                                   neurons whose weights are > Jepsilon */

    // classes to stream data to files
    ofstream ofsr;
    ofstream dw_fs;
    ofstream xy_fs;
    string datafolder = "data";

    double tauh;  // decay time of homeostatic plasticity, in ms
    double g;     // not used

    // method declarations
    Model(uint, uint, uint, int);  // construction
    void initLIF();                // this is called only if you want a non-default LIF model
    void sim_lif(int);
    double get_bump(int, int);
    double get_abump(int, int);
    double getRecent(uint);
    void setWeights(vector<vector<double>>);
    vector<vector<double>> getWeights();
    double dice();
    double ngn();
    void sim(uint);
    void setParams(py::dict, bool reset_connection_probs = false);
    py::dict getState();
    vector<vector<double>> calc_J(double, double);
    vector<uint> rnd_sample(uint, uint);
    double fd(double, double);

    vector<double> theta;
    vector<double> F;
    vector<double> D;
    double U = 0.6;  // default release probability for HAGA
    double taustf;
    double taustd;
    bool HAGA;

    // LimDeque limdeque;
    double uw = 0.0;
    double alpha_tau = 0.28;
    double heaviside(double);
    double alpha_function_LTP(double);
    double alpha_function_LTD(double);
    double tanh_LTP(double);
    double tanh_LTD(double);
    void homeostatic_plasticity();
    void boundary_condition();

    void STPonSpike(uint);
    void STPonNoSpike();
    void updateMembranePot(uint);
    void checkIfStim(uint);
    void asymSTDP(uint);
    void symSTDP(uint);
    void saveSTP();
    void reinitFD();
    void saveX();
    void saveDSPTS();
    void loadX(string);
    void loadDSPTS(string);
    deque<double> SplitString(string);
    deque<uint> SplitString_int(string);
    void reinit_Jinidx();
    void reinit_Jinidy();
    void saveSpts();
    void loadSpts(string);
    void saveRecentSpikes(uint, double);
    void isBadNumber(double, uint);
    void logFD_inh_exc();
    vector<vector<double>> FF;
    vector<vector<double>> DD;
    void dequeMaintenance(uint);
    double adjustment_factor(double);
    vector<double> getFR();
    void dumpXy(uint, double);
    double delta_stp;
    bool stp_on_I;

    void clip_pos(uint, uint);
    void clip_neg(uint, uint);
    double trunc_norm(double, double, double, double);
    // vector<deque<double>>& getUinh(); // if you want to get Uinh from a parent object
    // vector<deque<double>>& getUexc();

    void increment_array(py::array_t<double>);

    double acc0;
    double acc1;
    double acc2;
    double c0;
    double c1;
    double c2;
    double c;
    double acc;
    vector<uint> hStim;
    vector<double> stimIntensity;
    int cell_id;

    // ostringstream ossa;
    // ofstream ofsa;
    // string fstra;

    // diagnostics:
    //         ostringstream ossb;
    //         ofstream ofsb;
    //         string fstrb;

    // BEGIN: *** LIF constants and ARRAYS ***
    double refractory_period = 2.0;

    // equilibrium potentials:
    double V_E = 0.0;
    double V_I = -80.0;  // equilibrium potential for the inhibitory synapse
    double EL = -65.0;   // leakage potential, mV

    // critical voltages:
    double Vth = -55.0;  // threshold after which an AP is fired,   mV
    double Vr = -70.0;   // reset voltage (after an AP is fired), mV
    double Vspike = 10.0;

    // taus
    double tau_ampa = 8.0;
    double tau_nmda = 16.0;
    double tau_gaba = 8.0;

    // membrane taus
    double TAU_EXCITATORY = 10.0;
    double TAU_INHIBITORY = 20.0;
    double EPSILON = 0.0001;

    vector<int> AP;
    vector<double> neur_type_mask;
    vector<double> tau;
    vector<double> V;
    vector<double> in_refractory;

    vector<vector<double>> ampa;
    vector<vector<double>> nmda;
    vector<vector<double>> gaba;
    vector<double> dV;
    vector<double> I_E;
    vector<double> I_I;
    vector<int> delayed_spike;
    vector<int> neurons_that_spiked_at_this_step;
    double bump;
    double abump;

    double experimental = 1.0;
    // END: *** LIF constants and ARRAYS ***

    // flexible arrays can only be declared at the end of the class !!
    //         double sm[];
    // double* sm;

    string cwd;

   private:
    // init random number generator
    std::random_device m_randomdevice;
    std::mt19937 m_mt;
};

// LimDeque::LimDeque(uint maxlen_, Model* m_) {
//     m = m_;
//     for (uint i = 0; i < m->N; i++) {
//         Uexc.push_back(fdeque);
//         Uinh.push_back(fdeque);
//     }
//     m = m_;
//     maxlen = maxlen_;
// };

// void LimDeque::push(uint i) {
//     Uexc[i].push_back(0.0);
//     if (Uexc[i].size() > maxlen) {
//         Uexc[i].pop_front();
//     }
//     Uinh[i].push_back(0.0);
//     if (Uinh[i].size() > maxlen) {
//         Uinh[i].pop_front();
//     }
//     for (const auto& j : m->spts) {
//         if (j < m->NE) {
//             tmp_u = m->F[j] * m->D[j] * m->Jo[i][j];
//             m->u += tmp_u;
//             Uexc[i].back() += tmp_u;
//         } else {
//             tmp_u = m->Jo[i][j];
//             m->u += tmp_u;
//             Uinh[i].back() += tmp_u;
//         }
//     }
// }

// class construction _definition_. Requires no type specifiction.
Model::Model(uint _NE, uint _NI, uint _NEo, int _cell_id) : m_mt(m_randomdevice()) {
    // : limdeque(10, this), m_mt(m_randomdevice()) {
    cwd = getexepath();
    cout << cwd << endl;
    cout << "Thetas stay constant, change if you need to in `model.h` + in progress on total "
            "inhibition"
         << endl;
    cout << "soft_clip_dw: " << soft_clip_dw << endl;
    cell_id = _cell_id;
    NE = _NE;
    NEo = _NEo;
    NI = _NI;
    N = NE + NI;  //

    // initialize the weight matrix Jo
    Jo = calc_J(JEEinit, JEI);

    for (uint i = 0; i < NE; i++) {
        stimIntensity.push_back(0.0);
    }

    // initialize the STF and STD vectors
    for (uint i = 0; i < N; i++) {
        F.push_back(U);
        D.push_back(1.0);
        UU.push_back(U);
        theta.push_back(0.0);  // firing thresholds for excitatory neurons;
    }

    // how many neurons get updated per time step
    SNE = (uint)floor(NE * h / tmE + 0.001);
    SNI = max((uint)1, (uint)floor(NI * h / tmI + 0.001));

    NEa = (int)floor(NE * a + 0.01);  // Exact number of excitatory neurons stimulated externally
    pmax = NE / NEa;

    srand((uint)time(NULL));

    // ossSpike << "stp" << ".txt";
    // string fname = "data/spike_times_" + std::to_string(cell_id);
    // cout << fname << endl;

    char buf[256];
    if (getcwd(buf, sizeof(buf))) {
        std::cout << "Current working directory: " << buf << std::endl;
    }

    // ofsr.open(fname);
    // ofsr.precision(10);

    // dw_fs.open("data/dw_" + std::to_string(cell_id));
    // dw_fs.precision(10);

    // uncomment this to record presynaptic spikes
    // distrib_FS.open("data/distrib");
    // distrib_FS.precision(10);

    for (uint i = 0; i < N; i++) {
        Jinidx.push_back(ivec);  // shape = (3000, max3000)
        for (uint j = 0; j < N; j++) {
            if (abs(Jo[i][j]) > Jepsilon) {
                Jinidx[i].push_back(j);
            }
        }
    }

    for (uint j = 0; j < N; j++) {
        Jinidy.push_back(ivec);  // shape = (3000, max3000)
        for (uint i = 0; i < N; i++) {
            if (abs(Jo[i][j]) > Jepsilon) {
                Jinidy[j].push_back(i);
            }
        }
    }

    // create a vector size N and fill it with zeros
    // this vector says if a neuron is spiking or not
    for (uint i = 0; i < N; i++) {
        x.push_back(0);
    }

    // we remember in spts the ids of neurons that are spiking at the current
    // step and set neurons with these ids to 1 at the beginning of the
    // simulation, some neurons have to spike, so we initialize some neurons to
    // 1 to make them spike
    for (uint i = 0; i < N; i++) {
        // elements corresponding to excitatory neurons are filled with
        // ones with probability xEinit (0.02)
        if (i < NE && dice() < xEinit) {
            spts.insert(i);
            x[i] = 1;
        }
        // elements corresponding to inhibitory neurons are filled with
        // ones with probability xIinit (0.01)
        if (i >= NE && dice() < xIinit) {
            spts.insert(i);
            x[i] = 1;
        }
    }

    for (uint i = 0; i < N; i++) {
        dspts.push_back(ideque);
    }

    // initialize stimulus
    for (uint i = 0; i < NE; i++) {
        hStim.push_back(0);
    }

    // make a container for neurons' spike histories
    for (uint i = 0; i < NE; i++) {
        sphist.push_back(spdeque);
    }

    for (uint i = 0; i < NE + NI; i++) {
        Uexc.push_back(fdeque);
        Uinh.push_back(fdeque);
    }

    vector<bool> row(N, false);
    for (uint i = 0; i < N; i++) {
        frozens.push_back(row);
    }
    initLIF();
};
