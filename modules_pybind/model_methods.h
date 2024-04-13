#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "truncated_normal.hpp"

using namespace std;
using uint = unsigned int;

// double Model::dice(){
// 	return rand()/(RAND_MAX + 1.0);
// }

void Model::saveRecentSpikes(uint i, double t) {
    sphist[i].push_back(t);
    // remove spikes older than DEQUE_T_LEN
    if ((t - sphist[i].front()) > DEQUE_LEN_MS) {
        sphist[i].pop_front();
    }
}

double Model::dice() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(m_mt);
}

double Model::trunc_norm(double mu, double sigma, double a, double b) {
    int seed = generateRandom32bitInt();
    return truncated_normal_ab_sample(mu, sigma, a, b, seed);
}

double Model::ngn() {
    // sample from a normal distribution based on two uniform distributions
    double u = Model::dice();
    double v = Model::dice();
    return sqrt(-2.0 * log(u)) * cos(2.0 * pi * v);
}

// choose the neuron ids that will be updated at the current time step
vector<uint> Model::rnd_sample(uint ktmp, uint Ntmp) {  // when ktmp << Ntmp
    vector<uint> smpld;
    uint xtmp;
    bool tof;
    while (smpld.size() < (uint)ktmp) {
        xtmp = (uint)floor(Ntmp * Model::dice());
        tof = true;
        // make sure that the sampled id isn't the same as any of the previous
        // ones
        for (uint i = 0; i < smpld.size(); i++) {
            if (xtmp == smpld[i]) {
                tof = false;
            }
        }
        if (tof) smpld.push_back(xtmp);
    }
    return smpld;
}

double Model::fd(double x, double alpha) {
    return log(1.0 + alpha / JEE * abs(x)) / log(1.0 + alpha);
}

vector<double> Model::getFR() {
    vector<double> fr(N, 0.0);
    int fr_acc;
    for (uint i = 0; i < N; i++) {
        fr_acc = 0;
        for (const double& tt : dspts[i]) {
            if (tt > (t - twnd / 10.0)) {
                fr_acc += 1;
            }
        }
        fr[i] = (float)(fr_acc) * (1000.0 / twnd);
    }
    return fr;
}

void Model::STPonSpike(uint i) {
    F[i] += UU[i] * (1 - F[i]);  // U = 0.6
    D[i] -= D[i] * F[i];

    // remove it from the set of spiking neurons
    it = spts.find(i);
    if (it != spts.end()) {
        spts.erase(it++);
    }
    // and turn it OFF
    x[i] = 0;
}

void Model::logFD_inh_exc() {
    FF.push_back(F);
    DD.push_back(D);
}

void Model::STPonNoSpike() {
    for (uint i = 0; i < N; i++) {
        F[i] += hsd * (UU[i] - F[i]) / taustf;  // @@ don't forget about hsd!!!
        D[i] += hsd * (1.0 - D[i]) / taustd;
    }
}

void Model::saveDSPTS() {
    ofstream ofsDSPTS;
    // ofsDSPTS.open("DSPTS_" + std::to_string(cell_id) + "_" +
    // std::to_string(t));
    ofsDSPTS.open(datafolder + "/DSPTS_" + std::to_string(cell_id));
    ofsDSPTS.precision(10);

    for (uint i = 0; i < N; i++) {
        ofsDSPTS << i;
        for (uint sidx = 0; sidx < dspts[i].size(); sidx++) {
            ofsDSPTS << " " << dspts[i][sidx];
        }
        ofsDSPTS << endl;
    }
}

void Model::saveX() {
    ofstream ofsX;
    // ofsX.open("X_" + std::to_string(cell_id) + "_" + std::to_string(t));
    ofsX.open(datafolder + "/X_" + std::to_string(cell_id));
    ofsX.precision(10);
    ofsX << x[0];
    for (uint i = 1; i < N; i++) {
        ofsX << " " << x[i];
    }
    ofsX << endl;
}

void Model::loadDSPTS(string tt) {
    dspts.clear();
    deque<double> iideque;
    // ifstream file("DSPTS_" + std::to_string(cell_id) + "_" + tt);
    ifstream file(datafolder + "/DSPTS_" + std::to_string(cell_id));

    if (!file.is_open()) {
        cout << "DSPTS file not found." << endl;
        throw "DSPTS file not found.";
    }
    string line;
    while (getline(file, line)) {
        iideque = SplitString(line.c_str());
        iideque.pop_front();
        dspts.push_back(iideque);
    }
    file.close();
    cout << "DSPTS loaded" << endl;
}

void Model::loadX(string tt) {
    x.clear();
    deque<double> iideque;
    ifstream file(datafolder + "/X_" + std::to_string(cell_id));
    if (!file.is_open()) {
        cout << "X file not found." << endl;
        throw "X file not found.";
    }
    string line;
    while (getline(file, line)) {
        iideque = SplitString(line.c_str());
        for (uint i = 0; i < N; i++) {
            x.push_back(iideque[i]);
        }
    }
    file.close();
    cout << "X loaded" << endl;
}

deque<double> Model::SplitString(string line) {
    deque<double> iideque;
    string temp = "";
    for (uint i = 0; i < line.length(); ++i) {
        if (line[i] == ' ') {
            iideque.push_back(stod(temp));
            temp = "";
        } else {
            temp.push_back(line[i]);
        }
    }
    iideque.push_back(stod(temp));
    return iideque;
}

deque<uint> Model::SplitString_int(string line) {
    deque<uint> iideque;
    string temp = "";
    for (uint i = 0; i < line.length(); ++i) {
        if (line[i] == ' ') {
            iideque.push_back(stoi(temp));
            temp = "";
        } else {
            temp.push_back(line[i]);
        }
    }
    return iideque;
}

double Model::getRecent(uint i) {
    int J = sphist[i].size();  // J is the number if presynaptic spikes
    // we have an array of pointers
    double acc = 0.0;  // zero accumulator
    for (int j = 0; j < J; j++) {
        // the weight of the spike will be the (exponentially) lower the older it is
        double expw = exp(0.08 * ((sphist[i][j]) - (t)));
        acc += expw;
    }
    return acc;
}

void Model::checkIfStim(uint i) {
    if (hStim[i] == 1) {
        if (dice() < stimIntensity[i]) {
            u += Ip;
        }
    }
}

double Model::heaviside(double x) { return double(x > 0); }

double Model::alpha_function_LTP(double wgt) {
    return 2.5 * exp(-wgt / alpha_tau) * (wgt / alpha_tau);
}

double Model::alpha_function_LTD(double wgt) {
    return 2.5 * exp(-(Jmax - wgt) / alpha_tau) * (Jmax - wgt) / alpha_tau;
}

double Model::tanh_LTP(double wgt) { return -tanh(30 * (wgt - Jmax)); }

double Model::tanh_LTD(double wgt) { return tanh(30 * wgt); }

double Model::adjustment_factor(double wgt) {
    if (wgt > 0.0) {
        return 0.5 * (cos(wgt / Jmax * PI) + 1.0);
    } else {
        return -0.5 * (cos(wgt / Jmax * PI)) + 0.5;
    }
}

void Model::asymSTDP(uint i) {
    /* First (LTD), we treat the chosen neuron as PREsynaptic and loop over all
      the POSTSYNAPTIC excitatory neurons that THE CHOSEN NEURON synapses on.
      Since we're at time t (and this is the latest time), the spikes recorded on
      those "POSTsynaptic" neurons will have an earlier timing than the spike
      recorded on the currently chosen neuron (that we treat as PREsynaptic). This
      indicates that the synaptic weight between this chosen neuron (presynaptic)
      and all the other neurons (postsynaptic) will decrease.  */

    for (const auto& ip : Jinidy[i]) {
        if (t > tinit) {
            // dspts is a deque of spiking times on the ith POSTSYNAPTIC neurons
            for (const auto& tt : dspts[ip]) {
                if (frozens[ip][i] == true) {
                    continue;
                }
                dw = -Cd * exp((tt - t) / tpd);
                if (HAGA == 1) {
                    dw *= F[i] * D[i];
                    // isBadNumber(dw, 330);
                }

                // if (t > 100000.0) {
                //     dw = alpha_function_LTD(abs(Jo[ip][i])) * dw;
                // }

                if (soft_clip_dw) {
                    dw *= adjustment_factor(Jo[ip][i]);  // cosine
                    // dw *= tanh_LTD(Jo[ip][i]);
                }

                if (i < NE) {
                    if (dump_dw) {
                        dw_fs << t << " " << dw << " " << i << " " << F[i] << " " << D[i] << " "
                              << endl;
                    }
                    Jo[ip][i] += dw;
                    clip_pos(ip, i);
                } else {
                    if (dump_dw) {
                        dw_fs << t << " " << -dw << " " << i << " " << F[i] << " " << D[i] << " "
                              << endl;
                    }
                    Jo[ip][i] -= dw;
                    clip_neg(ip, i);
                }
            }
        }
    }

    // (LTP)

    /* Jinidx is a list of lists (shape (2500 POSTsyn, n PREsyn)). E.g. if in
      row 15 we have number 10, it means that the weight between POSTsynaptic
      neuron 15 and presynaptc neuron 10 is greater than Jepsilon */

    /* we treat the currently chosen neuron as POSTtsynaptic, and we loop over
      all the presynaptic neurons that synapse on the current postsynaptic neuron.
      At time t (the latest time, and we don't yet know any spikes that will
      happen in the future) all the spikes on the presynaptic neurons with id j
      will have an earlier timing that the spike on the currently chosen neuron i
      (that we treat as postsynaptic for now). This indicates that the weights
      between the chosen neuron treated as post- synaptic for now and all the
      other neurons (treated as presynaptic for now) will be potentiated.  */

    for (const auto& j : Jinidx[i]) {
        // at each loop we get the id of the jth presynaptic neuron with J >
        // Jepsilon
        if (t > tinit) {
            for (const auto& tt : dspts[j]) {
                // we loop over all the spike times on the jth PRESYNAPTIC neuron
                if (frozens[i][j] == true) {
                    continue;
                }
                dw = Cp * exp(-(t - tt) / tpp);
                if (HAGA == 1) {
                    dw *= F[j] * D[j];
                    // isBadNumber(dw, 380);
                }

                // if (t > 100000.0) {
                //     dw = alpha_function_LTP(abs(Jo[i][j])) * dw;
                // }

                if (soft_clip_dw) {
                    dw *= adjustment_factor(Jo[i][j]);  // cosine
                    // dw *= tanh_LTP(Jo[i][j]);
                }

                if (j < NE) {
                    if (dump_dw) {
                        dw_fs << t << " " << dw << " " << i << " " << F[j] << " " << D[j] << " "
                              << endl;
                    }
                    Jo[i][j] += dw;
                    clip_pos(i, j);
                } else {
                    if (dump_dw) {
                        dw_fs << t << " " << -dw << " " << i << " " << F[j] << " " << D[j] << " "
                              << endl;
                    }
                    Jo[i][j] -= dw;
                    clip_neg(i, j);
                }
            }
        }
    }
}

void Model::symSTDP(uint i) {
    // First (LTD)

    for (const auto& ip : Jinidy[i]) {
        if (t > tinit) {
            // dspts is a deque of spiking times on the ith POSTSYNAPTIC neurons
            for (const auto& tt : dspts[ip]) {
                if (frozens[ip][i] == true) {
                    continue;
                }
                dw = Cp * exp((tt - t) / tpp) - fd(Jo[ip][i], alpha) * Cd * exp(-(t - tt) / tpd);
                if (HAGA == 1) {
                    dw *= F[i] * D[i];
                    // isBadNumber(dw, 415);
                }

                // if (t > 100000.0) {
                //     dw = alpha_function_LTD(abs(Jo[ip][i])) * dw;
                // }

                if (soft_clip_dw) {
                    dw *= adjustment_factor(Jo[ip][i]);  // cosine
                    // dw *= tanh_LTP(Jo[ip][i]);
                }

                if (i < NE) {
                    if (dump_dw) {
                        dw_fs << t << " " << dw << " " << i << " " << F[i] << " " << D[i] << " "
                              << endl;
                    }
                    Jo[ip][i] += dw;
                    clip_pos(ip, i);
                } else {
                    if (dump_dw) {
                        dw_fs << t << " " << -dw << " " << i << " " << F[i] << " " << D[i] << " "
                              << endl;
                    }
                    Jo[ip][i] -= dw;
                    clip_neg(ip, i);
                }
            }
        }
    }

    // (LTP)

    for (const auto& j : Jinidx[i]) {
        if (t > tinit) {
            for (const auto& tt : dspts[j]) {
                if (frozens[i][j] == true) {
                    continue;
                }
                dw = Cp * exp(-(t - tt) / tpp) - fd(Jo[i][j], alpha) * Cd * exp(-(t - tt) / tpd);
                if (HAGA == 1) {
                    dw *= F[j] * D[j];
                }
                // isBadNumber(dw, 446);

                // if (t > 100000.0) {
                //     dw = alpha_function_LTP(abs(Jo[i][j])) * dw;
                // }

                if (soft_clip_dw) {
                    dw *= adjustment_factor(Jo[i][j]);  // cosine
                    // dw *= tanh_LTP(Jo[i][j]);
                }

                if (j < NE) {
                    if (dump_dw) {
                        dw_fs << t << " " << dw << " " << i << " " << F[j] << " " << D[j] << " "
                              << endl;
                    }
                    Jo[i][j] += dw;
                    clip_pos(i, j);
                } else {
                    if (dump_dw) {
                        dw_fs << t << " " << -dw << " " << i << " " << F[j] << " " << D[j] << " "
                              << endl;
                    }
                    Jo[i][j] -= dw;
                    clip_neg(i, j);
                }
            }
        }
    }
}

// initialize the weight matrix
vector<vector<double>> Model::calc_J(double JEEinit, double JEI) {
    vector<vector<double>> J;
    cout << "JEEinit = " << JEEinit << endl;

    for (uint i = 0; i < NE; i++) {
        J.push_back(dvec);
        for (uint j = 0; j < NE; j++) {
            J[i].push_back(0.0);
            // first E-E weights consistent with the E-E connection probability

            if (i != j && dice() < cEE) {
                J[i][j] = JEEinit * (1.0 + sigJ * ngn());
                // J[i][j] = trunc_norm(JEEinit, sigJ, Jmin, Jmax);

                // if some weight is out of range, we clip it
                if (J[i][j] < Jmin) J[i][j] = Jmin;
                if (J[i][j] > Jmax) J[i][j] = Jmax;
            }
        }
        // then the E-I weights
        for (uint j = NE; j < N; j++) {
            J[i].push_back(0.0);  // here the matrix J is at first of size 2500,
                                  // we extend it
            if (dice() < cEI) {
                J[i][j] -= JEI; /* becuase jth presynaptic inhibitory synapsing
                        on an ith excitatory postsynaptic neuron should inhibit it.
                        Hence the minus */
            }
        }
    }

    // then the I-E and I-I weights
    for (uint i = NE; i < N; i++) {
        J.push_back(dvec);
        for (uint j = 0; j < NE; j++) {
            J[i].push_back(0.0);
            if (dice() < cIE) {
                J[i][j] += JIE;
            }
        }
        for (uint j = NE; j < N; j++) {
            J[i].push_back(0.0);
            if (i != j && dice() < cII) {
                J[i][j] -= JII;
            }
        }
    }

    // prohibit certain weights
    for (uint i = NE - NEo; i < N; i++) {  // NE-NEo or N
        for (uint j = NE - NEo; j < NE; j++) {
            J[i][j] = 0.0;
        }
    }
    for (uint i = NE - NEo; i < NE; i++) {
        for (uint j = NE - NEo; j < N; j++) {
            J[i][j] = 0.0;
        }
    }

    return J;
}

void Model::reinitFD() {
    for (uint i = 0; i < N; i++) {
        F[i] = U;
        D[i] = 1.0;
        UU[i] = U;
    }
}

void Model::reinit_Jinidx() {
    Jinidx.clear();
    for (uint i = 0; i < N; i++) {
        Jinidx.push_back(ivec);  // shape = (3000, max3000)
        for (uint j = 0; j < N; j++) {
            if (abs(Jo[i][j]) > Jepsilon) {
                Jinidx[i].push_back(j);
            }
        }
    }
}

void Model::reinit_Jinidy() {
    Jinidy.clear();
    for (uint j = 0; j < N; j++) {
        Jinidy.push_back(ivec);  // shape = (3000, max3000)
        for (uint i = 0; i < N; i++) {
            if (abs(Jo[i][j]) > Jepsilon) {
                Jinidy[j].push_back(i);
            }
        }
    }
}

void Model::saveSpts() {
    //   for (it = spts.begin(); it != spts.end(); it++) {
    //       cout << *it << endl;
    //   }
    // same as above (using an iterator, but more pythonic for ... in...)
    ofstream ofsSpts;
    // ofsSpts.open("SPTS_" + to_string(cell_id) + "_" + to_string(t));
    ofsSpts.open(datafolder + "/SPTS_" + to_string(cell_id));

    for (int spt : spts) {
        ofsSpts << spt << " ";
    }
}

void Model::loadSpts(string tt) {
    spts.clear();

    deque<uint> iideque;
    // ifstream file("SPTS_" + to_string(cell_id) + "_" + tt);
    ifstream file(datafolder + "/SPTS_" + to_string(cell_id));

    if (!file.is_open()) {
        cout << "SPTS file not found." << endl;
        throw "SPTS file not found.";
    }
    string line;
    while (getline(file, line)) {
        iideque = SplitString_int(line.c_str());
        for (auto spt : iideque) {
            spts.insert(spt);
        }
    }
    file.close();
    cout << "SPTS loaded" << endl;
}

void Model::setParams(py::dict params, bool reset_connection_probs) {
    if (params.contains("alpha")) alpha = params["alpha"].cast<double>();
    if (params.contains("h")) h = params["h"].cast<double>();
    if (params.contains("T")) T = params["T"].cast<double>();
    if (params.contains("itauh")) itauh = params["itauh"].cast<uint>();
    if (params.contains("hsd")) hsd = params["hsd"].cast<double>();
    if (params.contains("hh")) hh = params["hh"].cast<double>();
    if (params.contains("Ip")) Ip = params["Ip"].cast<double>();
    if (params.contains("a")) a = params["a"].cast<double>();
    if (params.contains("xEinit")) xEinit = params["xEinit"].cast<double>();
    if (params.contains("xIinit")) xIinit = params["xIinit"].cast<double>();
    if (params.contains("tinit")) tinit = params["tinit"].cast<double>();
    if (params.contains("cEE")) cEE = params["cEE"].cast<double>();
    if (params.contains("cIE")) cIE = params["cIE"].cast<double>();
    if (params.contains("cEI")) cEI = params["cEI"].cast<double>();
    if (params.contains("cII")) cII = params["cII"].cast<double>();
    if (params.contains("JEEinit")) JEEinit = params["JEEinit"].cast<double>();
    if (params.contains("JEE")) JEE = params["JEE"].cast<double>();
    if (params.contains("JEI")) JEI = params["JEI"].cast<double>();
    if (params.contains("JIE")) JIE = params["JIE"].cast<double>();
    if (params.contains("JEEh")) JEEh = params["JEEh"].cast<double>();
    if (params.contains("sigJ")) sigJ = params["sigJ"].cast<double>();
    if (params.contains("Jtmax")) Jtmax = params["Jtmax"].cast<double>();
    if (params.contains("Jtmin")) Jtmin = params["Jtmin"].cast<double>();
    if (params.contains("hE")) hE = params["hE"].cast<double>();
    if (params.contains("hI")) hI = params["hI"].cast<double>();
    if (params.contains("IEex")) IEex = params["IEex"].cast<double>();
    if (params.contains("IIex")) IIex = params["IIex"].cast<double>();
    if (params.contains("mex")) mex = params["mex"].cast<double>();
    if (params.contains("sigex")) sigex = params["sigex"].cast<double>();

    if (params.contains("tmE")) tmE = params["tmE"].cast<double>();
    if (params.contains("tmI")) tmI = params["tmI"].cast<double>();
    if (params.contains("trec")) trec = params["trec"].cast<double>();
    if (params.contains("Jepsilon")) Jepsilon = params["Jepsilon"].cast<double>();
    if (params.contains("tpp")) tpp = params["tpp"].cast<double>();
    if (params.contains("tpd")) tpd = params["tpd"].cast<double>();

    if (params.contains("Cp")) Cp = params["Cp"].cast<double>();
    if (params.contains("Cd")) Cd = params["Cd"].cast<double>();
    if (params.contains("twnd")) twnd = params["twnd"].cast<double>();
    if (params.contains("g")) g = params["g"].cast<double>();

    if (params.contains("taustf")) taustf = params["taustf"].cast<double>();
    if (params.contains("taustd")) taustd = params["taustd"].cast<double>();
    if (params.contains("HAGA")) HAGA = params["HAGA"].cast<bool>();
    if (params.contains("symmetric")) symmetric = params["symmetric"].cast<bool>();
    if (params.contains("U")) U = params["U"].cast<double>();
    if (params.contains("soft_clip_dw")) soft_clip_dw = params["soft_clip_dw"].cast<bool>();

    // recalculate values that depend on the parameters
    SNE = (uint)floor(NE * h / tmE + 0.001);
    SNI = max((uint)1, (uint)floor(NI * h / tmI + 0.001));

    Jmax = 5.0 * JEE;   // ???
    Jmin = 0.01 * JEE;  // ????
    // Cp = 0.1*JEE;              // must be 0.01875 (in the paper)
    // Cd = Cp*tpp/tpd;           // must be 0.0075 (in the paper)
    hsig = 0.001 * JEE;                // i.e. 0.00015 per time step (10 ms)
    NEa = (uint)floor(NE * a + 0.01);  // Exact number of excitatory neurons stimulated externally
    pmax = NE / NEa;

    tauh = itauh * 1000.0;  // decay time of homeostatic plasticity, in ms

    reinitFD();

    if (reset_connection_probs) {
        reinit_Jinidx();
        reinit_Jinidy();
        Jo = calc_J(JEEinit, JEI);
    }
}

py::dict Model::getState() {
    py::dict ret_dict;
    ret_dict["alpha"] = alpha;
    ret_dict["JEI"] = JEI;
    ret_dict["T"] = T;
    ret_dict["h"] = h;
    ret_dict["NE"] = NE;
    ret_dict["NI"] = NI;
    ret_dict["cEE"] = cEE;
    ret_dict["cIE"] = cIE;
    ret_dict["cEI"] = cEI;
    ret_dict["cII"] = cII;
    ret_dict["JEE"] = JEE;
    ret_dict["JEEinit"] = JEEinit;
    ret_dict["JIE"] = JIE;
    ret_dict["JII"] = JII;
    ret_dict["JEEh"] = JEEh;
    ret_dict["sigJ"] = sigJ;
    ret_dict["Jtmax"] = Jtmax;
    ret_dict["Jtmin"] = Jtmin;
    ret_dict["hE"] = hE;
    ret_dict["hI"] = hI;
    ret_dict["IEex"] = IEex;
    ret_dict["IIex"] = IIex;
    ret_dict["mex"] = mex;
    ret_dict["sigex"] = sigex;
    ret_dict["tmE"] = tmE;
    ret_dict["tmI"] = tmI;
    ret_dict["trec"] = trec;
    ret_dict["Jepsilon"] = Jepsilon;
    ret_dict["tpp"] = tpp;
    ret_dict["tpd"] = tpd;
    ret_dict["twnd"] = twnd;
    ret_dict["g"] = g;
    ret_dict["itauh"] = itauh;
    ret_dict["hsd"] = hsd;
    ret_dict["hh"] = hh;
    ret_dict["Ip"] = Ip;
    ret_dict["a"] = a;
    ret_dict["xEinit"] = xEinit;
    ret_dict["xIinit"] = xIinit;
    ret_dict["tinit"] = tinit;

    ret_dict["Jmin"] = Jmin;
    ret_dict["Jmax"] = Jmax;
    ret_dict["Cp"] = Cp;
    ret_dict["Cd"] = Cd;
    ret_dict["SNE"] = SNE;
    ret_dict["SNI"] = SNI;
    ret_dict["NEa"] = NEa;
    ret_dict["t"] = t;

    ret_dict["U"] = U;
    ret_dict["taustf"] = taustf;
    ret_dict["taustd"] = taustd;
    ret_dict["HAGA"] = HAGA;
    ret_dict["symmetric"] = symmetric;
    ret_dict["soft_clip_dw"] = soft_clip_dw;
    return ret_dict;
}

void Model::sim(uint interval) {
    // std::cout << "SNE = " << SNE << ", SNI = " << SNI << std::endl;
    // std::cout << "t = " << t << std::endl;
    // Timer timer;

    if (!dw_fs.is_open()) {
        // Open the file stream
        dw_fs.open(datafolder + "/dw_" + std::to_string(cell_id));
        dw_fs.precision(10);
    }

    if (!ofsr.is_open()) {
        // Open the file stream
        ofsr.open(datafolder + "/spike_times_" + std::to_string(cell_id));
        ofsr.precision(10);
    }

    // HAVING INITIALIZED THE NETWORK, WE go time step by time step
    while (interval > 0) {
        t += h;
        interval -= 1;

        smpld = rnd_sample(SNE, NE);

        // #pragma omp parallel for
        for (const int& i : smpld) {
            // stp on spike on the chosen neuron
            if (x[i] == 1) {
                F[i] += UU[i] * (1 - F[i]);  // U = 0.6
                D[i] -= D[i] * F[i];

                // remove it from the set of spiking neurons
                it = spts.find(i);
                if (it != spts.end()) {
                    spts.erase(it++);
                }
                x[i] = 0;  // and turn it OFF
            }

            // update membrane potential
            u = -hE + IEex * (mex + sigex * ngn());  // pentagon, p.12
            dequeMaintenance(i);
            for (const auto& j : spts) {
                if (j < NE) {
                    delta_stp = F[j] * D[j];
                } else {
                    if (stp_on_I) {
                        delta_stp = F[j] * D[j];
                    } else {
                        delta_stp = 1.0;
                    }
                }

                u += delta_stp * Jo[i][j];

                if (j < NE) {
                    Uexc[i].back() += delta_stp * Jo[i][j];
                } else {
                    Uinh[i].back() += delta_stp * Jo[i][j];
                }
            }

            dumpXy(i, u);

            checkIfStim(i);

            if (u > theta[i]) {
                spts.insert(i);
                dspts[i].push_back(t);  // SHAPE: (n_postsyn x pytsyn_sp_times)
                x[i] = 1;

                if (saveflag) {
                    // #pragma omp critical
                    ofsr << t << " " << i << endl;  // record a line to file
                    saveRecentSpikes(i, t);
                }

                if ((STDPon) && (t > 200)) {
                    if (symmetric) {
                        symSTDP(i);
                    } else {
                        asymSTDP(i);
                    }
                }
                // ????
                // if (use_thetas) {
                //     theta[i] += 0.13;
                // }
            }
        }

        // if (use_thetas) {
        //     // exponentially decaying threshold for excitatory neurons
        //     for (uint i_ = 0; i_ < N; i_++) {
        //         theta[i_] *= 0.99995;
        //     }
        // }

        // we sample INHIBITORY neurons to be updated at the current step
        smpld = rnd_sample(SNI, NI);

        // #pragma omp parallel for
        for (const uint i_ : smpld) {
            uint i = NE + i_;

            // stp on spike on the chosen neuron
            if (x[i] == 1) {
                if (stp_on_I) {
                    cout << stp_on_I << endl;
                    F[i] += UU[i] * (1 - F[i]);
                    D[i] -= D[i] * F[i];
                }

                // remove it from the set of spiking neurons
                it = spts.find(i);
                if (it != spts.end()) {
                    spts.erase(it++);
                }
                x[i] = 0;  // and turn it OFF
            }

            // update membrane potential
            u = -hI + IIex * (mex + sigex * ngn());  // hexagon, eq.5, p.12
            dequeMaintenance(i);
            for (const auto& j : spts) {
                if (stp_on_I) {
                    delta_stp = F[j] * D[j];
                } else {
                    delta_stp = 1.0;
                }

                u += delta_stp * Jo[i][j];

                if (j < NE) {
                    Uexc[i].back() += delta_stp * Jo[i][j];
                } else {
                    Uinh[i].back() += delta_stp * Jo[i][j];
                }
            }

            // NOTE: experimental. Saving P(X, y). Presynaptic and postsynaptic spikes.
            // if (saveflag) {
            //     distrib_FS << t << " " << i << " ";
            //     for (const auto& j : spts) {
            //         distrib_FS << j << " ";
            //     }
            //     if (u > 0) {
            //         distrib_FS << "1" << endl;
            //     } else {
            //         distrib_FS << "0" << endl;
            //     }
            // }

            if (u > 0) {
                dspts[i].push_back(t);
                spts.insert(i);
                x[i] = 1;

                if (saveflag) {
                    // #pragma omp critical
                    ofsr << t << " " << i << endl;
                }

                // NOTE: STDP on inhibitory (unless masked)
                if ((STDPon) && (t > 200)) {
                    if (symmetric) {
                        symSTDP(i);
                    } else {
                        asymSTDP(i);
                    }
                }
            }
        }

        // EVERY 10 ms
        if (((uint)floor(t / h)) % 10 == 0) {
            STPonNoSpike();
        }

        // EVERY 100 ms (1) remove spikes older than 500 ms and (2) logFD
        if (((uint)floor(t / h)) % 100 == 0) {
            for (uint i = 0; i < N; i++) {
                // if we have spike times that are occured more than 500 ms
                while (!dspts[i].empty() && t - dspts[i].front() > twnd) {
                    dspts[i].pop_front();
                }
            }
            // logFD_inh_exc();
        }

        // EVERY 1000 ms
        if (((int)floor(t / h)) % 1000 == 0) {
            // FIXME: reset
            for (uint ii = 0; ii < N; ii++) {
                Jo[ii][ii] = 0.0;
            }
            reinit_Jinidx();
            reinit_Jinidy();

            if (homeostatic) {
                homeostatic_plasticity();
                boundary_condition();
            }
        }
    }
}

void Model::dequeMaintenance(uint i) {
    Uexc[i].push_back(0.0);
    while (Uexc[i].size() > 10) {
        Uexc[i].pop_front();
    }
    Uinh[i].push_back(0.0);
    while (Uinh[i].size() > 10) {
        Uinh[i].pop_front();
    }
}

void Model::isBadNumber(double n, uint i) {
    if (isnan(n)) {
        cout << "NaN on line " << i << endl;
    }
    if (isinf(n)) {
        cout << "Inf on line " << i << endl;
    }
}

void Model::setWeights(vector<vector<double>> _Jo) {
    for (uint i = 0; i < N; i++) {
        for (uint j = 0; j < N; j++) {
            Jo[i][j] = _Jo[i][j];
        }
    }
    reinit_Jinidx();
    reinit_Jinidy();
    cout << "Weights set\nJinidx recalculated" << endl;
}

void Model::homeostatic_plasticity() {
    for (uint i = 0; i < N; i++) {
        for (const uint& j : Jinidx[i]) {
            if (frozens[i][j]) {
                continue;
            }
            double w_ = abs(Jo[i][j]);
            k1 = (JEEh - w_) / tauh;
            k2 = (JEEh - (w_ + 0.5 * hh * k1)) / tauh;
            k3 = (JEEh - (w_ + 0.5 * hh * k2)) / tauh;
            k4 = (JEEh - (w_ + hh * k3)) / tauh;
            if (j < NE) {  // for E->E and E->I weights (they are positive)
                Jo[i][j] += hh * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 + hsig * ngn();
                // isBadNumber(Jo[i][j], 902);
                clip_pos(i, j);
            } else {  // for I->E and I->I weights (they are negative)
                Jo[i][j] -= hh * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 + hsig * ngn();
                // isBadNumber(Jo[i][j], 906);
                clip_neg(i, j);
            }
        }
    }
}

void Model::boundary_condition() {
    // E->E
    for (uint i = 0; i < NE; i++) {
        double Jav = 0.0;
        uint c = 0;
        for (const uint& j : Jinidx[i]) {
            if (j < NE) {
                Jav += Jo[i][j];
                c += 1;
            }
        }

        Jav = Jav / ((double)c);  // find mean weight per each postsynaptic neuron

        for (const uint& j : Jinidx[i]) {
            if ((j < NE) && (!frozens[i][j])) {
                if (Jav > Jtmax) Jo[i][j] -= abs(Jav - Jtmax);  // subtract the excess
                if (Jav < Jtmin) Jo[i][j] += abs(Jav - Jtmin);  // add shortage
                // isBadNumber(Jo[i][j], 925);
                clip_pos(i, j);
            }
        }
    }

    // E->I
    for (uint i = NE; i < N; i++) {
        double Jav = 0.0;
        uint c = 0;
        for (const uint& j : Jinidx[i]) {
            if (j < NE) {
                Jav += Jo[i][j];
                c += 1;
            }
        }

        Jav = Jav / ((double)c);  // find mean weight per each postsynaptic neuron

        for (const uint& j : Jinidx[i]) {
            if ((j < NE) && (!frozens[i][j])) {
                if (Jav > Jtmax) Jo[i][j] -= abs(Jav - Jtmax);  // subtract the excess
                if (Jav < Jtmin) Jo[i][j] += abs(Jav - Jtmin);  // add shortage
                clip_pos(i, j);
            }
        }
    }

    // I->E
    for (uint i = 0; i < NE; i++) {
        double Jav = 0.0;
        uint c = 0;
        for (const uint& j : Jinidx[i]) {
            if (j >= NE) {
                Jav += Jo[i][j];
                c += 1;
            }
        }

        Jav = Jav / ((double)c);  // find mean weight per each postsynaptic neuron

        for (const uint& j : Jinidx[i]) {
            if ((j >= NE) && (!frozens[i][j])) {
                if (abs(Jav) > Jtmax) Jo[i][j] += abs(Jav - Jtmax);  // subtract the excess
                if (abs(Jav) < Jtmin) Jo[i][j] -= abs(Jav - Jtmin);  // add shortage
                clip_neg(i, j);
            }
        }
    }

    // I-I
    for (uint i = NE; i < N; i++) {
        double Jav = 0.0;
        uint c = 0;
        for (const uint& j : Jinidx[i]) {
            if (j >= NE) {
                Jav += Jo[i][j];
                c += 1;
            }
        }

        Jav = Jav / ((double)c);  // find mean weight per each postsynaptic neuron

        for (const uint& j : Jinidx[i]) {
            if ((j >= NE) && (!frozens[i][j])) {
                if (abs(Jav) > Jtmax) Jo[i][j] += abs(Jav - Jtmax);  // subtract the excess
                if (abs(Jav) < Jtmin) Jo[i][j] -= abs(Jav - Jtmin);  // add shortage
                clip_neg(i, j);
            }
        }
    }
}

void Model::clip_pos(uint i, uint j) {
    if (Jo[i][j] < Jmin) Jo[i][j] = Jmin;
    if (Jo[i][j] > Jmax) Jo[i][j] = Jmax;
}

void Model::clip_neg(uint i, uint j) {
    if (Jo[i][j] > -Jmin) Jo[i][j] = -Jmin;
    if (Jo[i][j] < -Jmax) Jo[i][j] = -Jmax;
}

// vector<deque<double>>& Model::getUinh() {
//     return limdeque.Uinh;
//     ;
// }

// vector<deque<double>>& Model::getUexc() {
//     return limdeque.Uexc;
//     ;
// }

void Model::sim_lif(int interval) {
    while (interval > 0) {
        t += h;
        interval -= 1;
        neurons_that_spiked_at_this_step.clear();  // to avoid data race with multithreading

#pragma omp parallel for
        for (uint ii = 0; ii < N; ii++) {
            if (AP[ii] == 1) {
                in_refractory[ii] = refractory_period + dice();
                AP[ii] = 0;
            }

            if (abs(in_refractory[ii]) < EPSILON) {
                delayed_spike[ii] = 1.0;
            } else {
                delayed_spike[ii] = 0.0;
            }

            // reset the currents(we recalculate from scratch for each neuron)
            I_E[ii] = 0.0;
            I_I[ii] = 0.0;

            for (uint jj = 0; jj < N; jj++) {
                // on DELAYED spike, bump the conductances
                if (Jo[ii][jj] > Jepsilon) {
                    if (delayed_spike[jj] == 1) {
                        // #pragma omp atomic
                        ampa[ii][jj] += F[jj] * D[jj] * Jo[ii][jj];
                        // #pragma omp atomic
                        nmda[ii][jj] += F[jj] * D[jj] * Jo[ii][jj];
                    }
                    // #pragma omp atomic
                    ampa[ii][jj] += (-ampa[ii][jj] / tau_ampa) * h;
                    nmda[ii][jj] += (-nmda[ii][jj] / tau_nmda) * h;
                } else if (Jo[ii][jj] < -Jepsilon) {
                    if (delayed_spike[jj] == 1) {
                        gaba[ii][jj] += abs(Jo[ii][jj]);
                    }
                    // #pragma omp atomic
                    gaba[ii][jj] += (-gaba[ii][jj] / tau_gaba) * h;
                } else {
                    ;
                }
                // accumulate currents (as in the HH model)
                I_E[ii] += -(ampa[ii][jj] * (V[ii] - V_E) + 0.1 * nmda[ii][jj] * (V[ii] - V_E));
                I_I[ii] += gaba[ii][jj] * (V[ii] - V_I);
            }

            // keep track of tot theoretical exc and inh onto this neuron
            dequeMaintenance(ii);
            Uexc[ii].back() = I_E[ii];
            Uinh[ii].back() = I_I[ii];

            // inhibitory current must polarize the neuron (make it more negative)
            // excitatory current causes dV to be positive (depolarize the neur)
            dV[ii] = (-(V[ii] - EL) / tau[ii] + I_E[ii] - I_I[ii]) * h;

            if (in_refractory[ii] > EPSILON) {
                dV[ii] = 0.0;
            }

            V[ii] += dV[ii];

            // stimulate if needed
            if (hStim[ii] == 1) {
                if (dice() < stimIntensity[ii]) {
                    if ((ii < NE) && (in_refractory[ii] < EPSILON)) {
                        V[ii] = Vth + 1.0;
                    }
                }
            }

            // FIXME: inject voltage spikes
            if (dice() < mex) {
                if ((ii < N) && (in_refractory[ii] < EPSILON)) {
                    V[ii] = Vth + 1.0;
                }
            }

            if (V[ii] > Vth) {  // NOTE: why not check refrac?
                V[ii] = Vr;
                AP[ii] = 1;

                // STP on spike on Excitatory
                if (ii < NE) {
                    F[ii] += UU[ii] * (1 - F[ii]);  // U = 0.6
                    D[ii] -= D[ii] * F[ii];
                }

                // record spike
                if (saveflag == 1) {
#pragma omp critical
                    neurons_that_spiked_at_this_step.push_back(ii);  // don't write immediately
                }

                // remember the spike time
                dspts[ii].push_back(t);  // SHAPE: (n_postsyn x pytsyn_sp_times)

                // perform weight change on spike only
                if ((STDPon) && (t > 200)) {
                    if (symmetric) {
                        // #pragma omp critical
                        symSTDP(ii);
                    } else {
                        // #pragma omp critical
                        asymSTDP(ii);
                    }
                }
            }

            // EVERY 10 ms STP on no spike
            if (((uint)floor(t / h)) % 10 == 0) {
                F[ii] += hsd * (UU[ii] - F[ii]) / taustf;  // @@ don't forget about hsd!!!
                D[ii] += hsd * (1.0 - D[ii]) / taustd;
            }

            in_refractory[ii] -= h;
        }

        // once out of the parfor region, dump the spikes
        if (saveflag == 1) {
            for (int kk : neurons_that_spiked_at_this_step) {
                ofsr << t << " " << kk << endl;  // record a line to file
            }
        }

        // EVERY 100 ms (1) remove spikes older than 500 ms and (2) logFD
        if (((uint)floor(t / h)) % 100 == 0) {
            for (uint i = 0; i < N; i++) {
                while (!dspts[i].empty() && t - dspts[i].front() > twnd) {
                    dspts[i].pop_front();
                }
            }
            // logFD_inh_exc();
        }

        // EVERY 1000 ms
        if (((int)floor(t / h)) % 1000 == 0) {
            // FIXME: reset
            for (uint ii = 0; ii < N; ii++) {
                Jo[ii][ii] = 0.0;
            }
            reinit_Jinidx();
            reinit_Jinidy();

            if (homeostatic) {
                homeostatic_plasticity();
                boundary_condition();
            }
        }
    }
}

void Model::increment_array(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    auto ptr = static_cast<double*>(buf.ptr);

    int rows = buf.shape[0];
    int cols = buf.shape[1];

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ptr[i * cols + j] += 1.0;
        }
    }
}

void Model::initLIF() {
    for (uint i = 0; i < N; i++) {
        if (dice() > 0.5) {
            AP.push_back(0);
        } else {
            AP.push_back(1);
        }
        V.push_back(EL);
        in_refractory.push_back(0.0);
        dV.push_back(0.0);
        I_E.push_back(0.0);
        I_I.push_back(0.0);
        delayed_spike.push_back(0);
        if (i < NE) {
            neur_type_mask.push_back(0.0);
            tau.push_back(TAU_EXCITATORY);
        } else {
            neur_type_mask.push_back(1.0);
            tau.push_back(TAU_INHIBITORY);
        }
        ampa.push_back(dvec);
        nmda.push_back(dvec);
        gaba.push_back(dvec);
        for (uint j = 0; j < N; j++) {
            ampa[i].push_back(0.0);
            nmda[i].push_back(0.0);
            gaba[i].push_back(0.0);
        }
    }
}

void Model::dumpXy(uint i, double u) {
    // save X and y for the (possible) calcualtion of I(X, y)
    // X are inputs to and y is the output of a neurons
    if (dump_xy) {
        if (!xy_fs.is_open()) {
            xy_fs.open(datafolder + "/xy_" + std::to_string(cell_id));
            xy_fs.precision(10);
        }
        uint out;
        if (u > 0.0) {
            out = 1;
        } else {
            out = 0;
        }
        xy_fs << t << " " << i << " " << out;
        for (const auto& j : spts) {
            xy_fs << " " << j;
        }
        xy_fs << endl;
    }
}