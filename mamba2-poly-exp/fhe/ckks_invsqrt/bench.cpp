// CKKS-only 1/sqrt(v) per RMSNorm instance in stock OpenFHE: Chebyshev series + Newton.
//
// The CKKS-native counterpart of fhe/seglut_invsqrt/bench.cpp (scheme switching + a
// TFHE table). No comparisons, no table, no scheme switch: each case is one
// EvalChebyshevSeries of 1/sqrt(v) on that layer's calibrated range [a, b], then `newton`
// steps y <- y (1.5 - 0.5 v y^2), exactly the Newton form the table benchmark uses.
//
// Cases come from design.py --emit, one line each:
//     name a b newton degree c_0 c_1 ... c_degree
// in numpy's Chebyshev convention (sum c_i T_i(t), t = (2v - a - b) / (b - a)). OpenFHE's
// EvalChebyshevSeries adds coefficients[0] / 2, so c_0 is doubled here.
//
// Per case: v log-uniform on [a, b] in every slot, encrypt fresh, time the evaluation,
// decrypt, and report levels consumed and worst / RMS relative bits against exact
// 1/sqrt(v). One ciphertext costs the same whether it holds 1 value (one token while
// decoding) or every slot (a whole sequence), so both per-value figures are printed.

#include "openfhe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

using namespace lbcrypto;

namespace {

using Clock = std::chrono::steady_clock;
double Since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Args {
    std::string cases = "cases.txt";
    bool toy = false;
    uint32_t multDepth = 0;      // 0: the deepest case's need + 1
    uint32_t ringDim = 0;        // 0: smallest ring 128-bit security allows
    uint32_t slots = 0;          // 0: ring / 2
    uint32_t seed = 0;
    int threads = 0;
    std::string only;            // run only cases whose name contains this
};

Args Parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (k == "--cases") a.cases = next();
        else if (k == "--toy") a.toy = true;
        else if (k == "--depth") a.multDepth = std::stoul(next());
        else if (k == "--ring") a.ringDim = std::stoul(next());
        else if (k == "--slots") a.slots = std::stoul(next());
        else if (k == "--seed") a.seed = std::stoul(next());
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--only") a.only = next();
        else { std::cerr << "unknown argument " << k << "\n"; std::exit(2); }
    }
    return a;
}

struct Case {
    std::string name;
    double a = 0, b = 0;
    uint32_t newton = 0, degree = 0;
    std::vector<double> coef;
};

std::vector<Case> ReadCases(const std::string& path, const std::string& only) {
    std::ifstream in(path);
    if (!in) { std::cerr << "cannot read " << path << "\n"; std::exit(2); }
    std::vector<Case> cs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Case c;
        ss >> c.name >> c.a >> c.b >> c.newton >> c.degree;
        c.coef.resize(c.degree + 1);
        for (auto& x : c.coef) ss >> x;
        if (!ss) { std::cerr << "bad case line: " << line.substr(0, 80) << "\n"; std::exit(2); }
        c.coef[0] *= 2.0;
        if (only.empty() || c.name.find(only) != std::string::npos) cs.push_back(std::move(c));
    }
    return cs;
}

// OpenFHE 1.2.1 EvalChebyshevSeries depth by degree (Paterson-Stockmeyer above degree 5)
uint32_t ChebDepth(uint32_t d) {
    const uint32_t top[] = {5, 13, 27, 59, 119, 247, 495, 1007, 2031};
    for (uint32_t i = 0; i < 9; ++i)
        if (d <= top[i]) return i + 3;
    return 12;
}

}  // namespace

int main(int argc, char** argv) {
    const Args A = Parse(argc, argv);
    if (A.threads > 0) omp_set_num_threads(A.threads);
    auto cases = ReadCases(A.cases, A.only);
    if (cases.empty()) { std::cerr << "no cases\n"; return 2; }

    uint32_t need = 0;
    for (const auto& c : cases) need = std::max(need, ChebDepth(c.degree) + 3 * c.newton);
    const uint32_t depth = A.multDepth ? A.multDepth : need + 1;

    CCParams<CryptoContextCKKSRNS> P;
    P.SetMultiplicativeDepth(depth);
    P.SetScalingModSize(50);
    P.SetFirstModSize(60);
    P.SetScalingTechnique(FLEXIBLEAUTO);
    P.SetSecurityLevel(A.toy ? HEStd_NotSet : HEStd_128_classic);
    if (A.toy) P.SetRingDim(A.ringDim ? A.ringDim : 8192);
    else if (A.ringDim) P.SetRingDim(A.ringDim);
    P.SetKeySwitchTechnique(HYBRID);
    if (A.slots) P.SetBatchSize(A.slots);
    auto cc = GenCryptoContext(P);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    const uint32_t N = cc->GetRingDimension();
    const uint32_t S = A.slots ? A.slots : N / 2;
    auto t0 = Clock::now();
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    const double tKey = Since(t0);

    std::cout << std::setprecision(4);
    std::cout << "== CKKS-only 1/sqrt(v), OpenFHE " << (A.toy ? "TOY (insecure)" : "128-bit")
              << ", threads " << omp_get_max_threads() << "\n"
              << "CKKS ring " << N << ", slots " << S << ", depth " << depth << " (cases need " << need
              << "), log2 Q " << std::floor(cc->GetModulus().GetMSB()) << ", keygen " << tKey << " s\n\n";

    std::cout << std::left << std::setw(18) << "case" << std::right << std::setw(9) << "R=b/a"
              << std::setw(6) << "deg" << std::setw(4) << "N" << std::setw(8) << "levels"
              << std::setw(10) << "pred" << std::setw(11) << "worst-bits" << std::setw(10) << "rms-bits"
              << std::setw(10) << "time s" << std::setw(13) << "us/value" << "\n";

    std::mt19937_64 rng(A.seed);
    double tSum = 0;
    std::vector<double> worstAll;
    for (const auto& c : cases) {
        std::uniform_real_distribution<double> U(std::log(c.a), std::log(c.b));
        std::vector<double> v(S);
        for (auto& x : v) x = std::exp(U(rng));
        auto ctV = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));

        auto t1 = Clock::now();
        auto y = cc->EvalChebyshevSeries(ctV, c.coef, c.a, c.b);
        for (uint32_t k = 0; k < c.newton; ++k) {
            auto vy2 = cc->EvalMult(cc->EvalSquare(y), ctV);
            y = cc->EvalMult(y, cc->EvalSub(1.5, cc->EvalMult(vy2, 0.5)));
        }
        const double t = Since(t1);
        tSum += t;

        Plaintext pt;
        cc->Decrypt(keys.secretKey, y, &pt);
        pt->SetLength(S);
        const auto out = pt->GetRealPackedValue();
        double worst = 0, ss = 0;
        for (uint32_t i = 0; i < S; ++i) {
            const double rel = out[i] * std::sqrt(v[i]) - 1.0;
            worst = std::max(worst, std::fabs(rel));
            ss += rel * rel;
        }
        const double wb = -std::log2(worst), rb = -std::log2(std::sqrt(ss / S));
        worstAll.push_back(wb);
        std::cout << std::left << std::setw(18) << c.name << std::right << std::setw(9) << std::setprecision(0)
                  << std::fixed << c.b / c.a << std::setw(6) << c.degree << std::setw(4) << c.newton
                  << std::setw(8) << y->GetLevel() << std::setw(10) << ChebDepth(c.degree) + 3 * c.newton
                  << std::setprecision(1) << std::setw(11) << wb << std::setw(10) << rb << std::setprecision(3)
                  << std::setw(10) << t << std::setprecision(2) << std::setw(13) << 1e6 * t / S << "\n"
                  << std::defaultfloat << std::setprecision(4);
    }
    std::cout << "\n" << cases.size() << " cases, " << tSum << " s total evaluation; worst case "
              << *std::min_element(worstAll.begin(), worstAll.end()) << " worst-case rel bits\n"
              << "per value: time / " << S << " slots for a whole sequence; the full time for one "
              << "token while decoding\n";

    // reference: one ct x ct multiply + relinearise + rescale at the top of the chain
    auto ctA = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(std::vector<double>(S, 0.5)));
    auto t2 = Clock::now();
    for (int i = 0; i < 10; ++i) {
        auto m = cc->EvalMult(ctA, ctA);
        cc->RescaleInPlace(m);
    }
    std::cout << "reference: one EvalMult + Rescale at level 0: " << Since(t2) / 10 << " s\n";
    return 0;
}
