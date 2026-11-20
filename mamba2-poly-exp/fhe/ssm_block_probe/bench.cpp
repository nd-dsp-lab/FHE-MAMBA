// One HE-native selective SSM block under CKKS, non-interactive, OpenFHE 1.2.1 on the CPU.
//
// The go/no-go latency probe for "pretrain an HE-native Mamba". It answers: how long does one
// block take at d_model 128, L 512, how many levels does it consume, and does it fit at
// N = 2^16 without bootstrapping. Nothing here is trained; weights are random with the
// right shapes and ranges, because latency and depth do not depend on their values.
//
// The block (HE-native: every operator is add/multiply by construction, nothing approximated
// after the fact):
//     x, z, B, C, v = in_proj(u)                  linear, no bias except v
//     x, B, C       = act(conv(x)), act(conv(B)), act(conv(C))     causal depthwise conv, K = 4
//     w             = v^2                         = |A_h| * Delta  (softplus replaced by a square)
//     a             = P_h(w)                      degree-4 Chebyshev of exp(-w) on [0, wmax_h]
//     h_t           = a_t h_{t-1} + (w_t / |A_h|) x_t B_t^T          selective scan, log2 L depth
//     y             = h_t C_t + D_h x_t
//     out           = out_proj(y * act(z))        gated norm dropped (folded / trained away)
// with act(x) = 0.5 x + 0.25 x^2. Heads past --selective-heads use a public (plaintext)
// decay instead: no v, no Delta, and the scan multiplies by a scalar instead of a ciphertext.
//
// Layout: a ciphertext holds S = slots / (2L) sequences. Each sequence owns a block of 2L slots,
// L pad slots then its L real positions. The pads are what make the scan correct: a shift by
// 2^k reads into the pad instead of the previous sequence, and pads of every plaintext weight
// are zero, so pads stay zero through the whole block. sim.py runs this exact algorithm on
// plaintext vectors against a sequential reference, including two negative controls.
//
// Usage (tjws-03):
//     cmake -B build && cmake --build build -j
//     ./build/bench --toy                      # insecure N = 2^13, L 64: API and correctness check
//     ./build/bench                            # 128-bit, D 128, L 512: the probe
//     ./build/bench --selective-heads 0        # public decay only (the HSSM end of the knob)
//     ./build/bench --boot                     # cost of one bootstrap with the block's depth left

#include "openfhe.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <omp.h>

using namespace lbcrypto;
using Ct = Ciphertext<DCRTPoly>;
using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

namespace {

using Clock = std::chrono::steady_clock;
double Since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }
double PeakRssGB() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1048576.0;  // Linux reports KiB
}

// ------------------------------------------------------------------------------ arguments

struct Args {
    bool toy = false, boot = false, ptLevel0 = false, naiveInproj = false;
    uint32_t bootIters = 1, levelsAfter = 0, budget0 = 4, budget1 = 4, bootScale = 59;  // levelsAfter 0: block depth
    uint32_t stack = 0, bootPrec = 0;  // stack N: N blocks with a bootstrap between; bootPrec 0: measure it
    uint32_t D = 128, L = 512, P = 64, N = 16, K = 4, expand = 2;
    int selectiveHeads = -1;  // -1: all
    uint32_t depth = 0;       // 0: 10 + log2 L
    uint32_t ring = 0;        // 0: smallest the security level allows
    uint32_t seed = 0;
    int threads = 0;
};

Args Parse(int argc, char** argv) {
    Args a;
    bool shapeSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (k == "--toy") a.toy = true;
        else if (k == "--boot") a.boot = true;
        else if (k == "--boot-iters") a.bootIters = std::stoul(next());
        else if (k == "--levels-after") a.levelsAfter = std::stoul(next());
        else if (k == "--budget") { a.budget0 = std::stoul(next()); a.budget1 = std::stoul(next()); }
        else if (k == "--boot-scale") a.bootScale = std::stoul(next());
        else if (k == "--stack") a.stack = std::stoul(next());
        else if (k == "--boot-prec") a.bootPrec = std::stoul(next());
        else if (k == "--pt-level0") a.ptLevel0 = true;
        else if (k == "--naive-inproj") a.naiveInproj = true;
        else if (k == "--D") { a.D = std::stoul(next()); shapeSet = true; }
        else if (k == "--L") { a.L = std::stoul(next()); shapeSet = true; }
        else if (k == "--P") { a.P = std::stoul(next()); shapeSet = true; }
        else if (k == "--state") { a.N = std::stoul(next()); shapeSet = true; }
        else if (k == "--selective-heads") a.selectiveHeads = std::stoi(next());
        else if (k == "--depth") a.depth = std::stoul(next());
        else if (k == "--ring") a.ring = std::stoul(next());
        else if (k == "--seed") a.seed = std::stoul(next());
        else if (k == "--threads") a.threads = std::stoi(next());
        else { std::cerr << "unknown argument " << k << "\n"; std::exit(2); }
    }
    if (a.toy && !shapeSet) { a.D = 64; a.L = 64; a.P = 32; a.N = 8; }
    return a;
}

// ------------------------------------------------------------------------------- the model

struct Shape {
    uint32_t slots, D, L, P, N, K, Di, H, Bk, S, lgL;
};

// Chebyshev interpolant at the d+1 Lobatto nodes (endpoints exact), in OpenFHE's convention
// value = c0/2 + sum_j c_j T_j((2x - a - b) / (b - a)). Same as sim.py: cheb_lobatto.
Vec ChebLobatto(double (*f)(double), double a, double b, uint32_t d) {
    Vec c(d + 1, 0.0);
    for (uint32_t j = 0; j <= d; ++j) {
        double s = 0;
        for (uint32_t k = 0; k <= d; ++k) {
            const double x = std::cos(M_PI * k / d);
            const double w = (k == 0 || k == d) ? 0.5 : 1.0;
            s += w * f(0.5 * (b - a) * x + 0.5 * (a + b)) * std::cos(M_PI * j * k / d);
        }
        c[j] = 2.0 / d * s;
    }
    c[d] *= 0.5;
    return c;
}

double ChebEval(const Vec& c, double a, double b, double x) {
    const double t = (2 * x - a - b) / (b - a);
    double b1 = 0, b2 = 0;  // Clenshaw
    for (int j = static_cast<int>(c.size()) - 1; j >= 1; --j) {
        const double b0 = 2 * t * b1 - b2 + c[j];
        b2 = b1;
        b1 = b0;
    }
    return t * b1 - b2 + 0.5 * c[0];
}

constexpr double kAct1 = 0.5, kAct2 = 0.25;
constexpr uint32_t kDecayDeg = 4;
double Act(double x) { return kAct1 * x + kAct2 * x * x; }
double NegExp(double w) { return std::exp(-w); }

struct Weights {
    Mat Wx, Wz, WB, WC, Wv, Wout;   // rows = outputs
    Vec bv;
    Mat convx, convB, convC;        // (channels, K); tap k reads t - k
    Vec cbx, cbB, cbC;
    Vec absA, wmax, aPub, Dskip;
    std::vector<Vec> decay;
    std::vector<bool> selective;
};

Weights MakeWeights(const Shape& s, int selectiveHeads, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> G(0, 1);
    auto U = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
    auto mat = [&](uint32_t r, uint32_t c, double scale) {
        Mat m(r, Vec(c));
        for (auto& row : m) for (auto& x : row) x = G(rng) * scale;
        return m;
    };
    auto vec = [&](uint32_t n, double scale) { Vec v(n); for (auto& x : v) x = G(rng) * scale; return v; };
    Weights W;
    W.wmax.resize(s.H); W.absA.resize(s.H); W.Dskip.resize(s.H); W.aPub.resize(s.H);
    for (uint32_t h = 0; h < s.H; ++h) {
        W.wmax[h] = U(1.0, 6.0);
        W.absA[h] = U(0.5, 4.0);
        W.Dskip[h] = U(0.5, 1.5);
        W.aPub[h] = std::exp(-W.wmax[h] / 4.0);
        W.decay.push_back(ChebLobatto(NegExp, 0.0, W.wmax[h], kDecayDeg));
        W.selective.push_back(selectiveHeads < 0 || static_cast<int>(h) < selectiveHeads);
    }
    const double iD = 1.0 / std::sqrt(s.D), iDi = 1.0 / std::sqrt(s.Di), iK = 0.5 / std::sqrt(s.K);
    W.Wx = mat(s.Di, s.D, iD); W.Wz = mat(s.Di, s.D, iD);
    W.WB = mat(s.N, s.D, iD); W.WC = mat(s.N, s.D, iD);
    W.Wv = mat(s.H, s.D, iD); W.bv = vec(s.H, 0.1);
    for (uint32_t h = 0; h < s.H; ++h) {  // keep v^2 mostly inside [0, wmax_h]
        const double sv = std::sqrt(W.wmax[h]) / 2.0;
        for (auto& x : W.Wv[h]) x *= sv;
        W.bv[h] *= sv;
    }
    W.convx = mat(s.Di, s.K, iK); W.convB = mat(s.N, s.K, iK); W.convC = mat(s.N, s.K, iK);
    W.cbx = vec(s.Di, 0.1); W.cbB = vec(s.N, 0.1); W.cbC = vec(s.N, 0.1);
    W.Wout = mat(s.D, s.Di, iDi);
    return W;
}

// Sequential forward pass, no packing. u: (D, L) -> (D, L). Same as sim.py: reference.
Mat Reference(const Shape& s, const Weights& W, const Mat& u) {
    auto matmul = [&](const Mat& M, uint32_t rows) {
        Mat y(rows, Vec(s.L, 0.0));
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < s.D; ++c)
                for (uint32_t t = 0; t < s.L; ++t) y[r][t] += M[r][c] * u[c][t];
        return y;
    };
    auto convAct = [&](const Mat& x, const Mat& taps, const Vec& bias) {
        Mat y(x.size(), Vec(s.L));
        for (size_t c = 0; c < x.size(); ++c)
            for (uint32_t t = 0; t < s.L; ++t) {
                double acc = bias[c];
                for (uint32_t k = 0; k < s.K && k <= t; ++k) acc += taps[c][k] * x[c][t - k];
                y[c][t] = Act(acc);
            }
        return y;
    };
    const Mat xa = convAct(matmul(W.Wx, s.Di), W.convx, W.cbx);
    const Mat Ba = convAct(matmul(W.WB, s.N), W.convB, W.cbB);
    const Mat Ca = convAct(matmul(W.WC, s.N), W.convC, W.cbC);
    Mat za = matmul(W.Wz, s.Di);
    for (auto& r : za) for (auto& x : r) x = Act(x);
    Mat v = matmul(W.Wv, s.H);
    Mat a(s.H, Vec(s.L)), Dl(s.H, Vec(s.L));
    for (uint32_t h = 0; h < s.H; ++h)
        for (uint32_t t = 0; t < s.L; ++t) {
            if (W.selective[h]) {
                const double w = std::pow(v[h][t] + W.bv[h], 2);
                a[h][t] = ChebEval(W.decay[h], 0.0, W.wmax[h], w);
                Dl[h][t] = w / W.absA[h];
            } else {
                a[h][t] = W.aPub[h];
                Dl[h][t] = 1.0;
            }
        }
    Mat g(s.Di, Vec(s.L));
    for (uint32_t c = 0; c < s.Di; ++c) {
        const uint32_t h = c / s.P;
        Vec state(s.N, 0.0);
        for (uint32_t t = 0; t < s.L; ++t) {
            double y = W.Dskip[h] * xa[c][t];
            for (uint32_t n = 0; n < s.N; ++n) {
                state[n] = a[h][t] * state[n] + Dl[h][t] * xa[c][t] * Ba[n][t];
                y += state[n] * Ca[n][t];
            }
            g[c][t] = y * za[c][t];
        }
    }
    Mat out(s.D, Vec(s.L, 0.0));
    for (uint32_t d = 0; d < s.D; ++d)
        for (uint32_t c = 0; c < s.Di; ++c)
            for (uint32_t t = 0; t < s.L; ++t) out[d][t] += W.Wout[d][c] * g[c][t];
    return out;
}

// ------------------------------------------------------------------------ the packed block

struct Counts {
    uint64_t ctct = 0, ctpt = 0, ctscalar = 0, rot = 0, add = 0;
};

class Block {
public:
    Block(CryptoContext<DCRTPoly> cc, const Shape& s, const Weights& W, bool ptLevel0, bool naiveInproj)
        : cc_(cc), s_(s), W_(W), ptLevel0_(ptLevel0), naive_(naiveInproj), M_(cc->GetCyclotomicOrder()) {}

    Counts c;
    std::vector<std::pair<std::string, double>> stageTime;
    std::vector<std::pair<std::string, uint32_t>> stageLevel;

    // Plaintext with per_block[b] on block b's real slots and zero on every pad.
    Vec BlockConst(const Vec& perBlock) const {
        Vec v(s_.slots, 0.0);
        for (uint32_t b = 0; b < s_.S && b < perBlock.size(); ++b)
            std::fill(v.begin() + b * s_.Bk + s_.L, v.begin() + (b + 1) * s_.Bk, perBlock[b]);
        return v;
    }

    // Encode at the level the ciphertext will be at when the multiply happens: FLEXIBLEAUTO
    // rescales a degree-2 ciphertext first. --pt-level0 encodes at the top level instead and
    // lets OpenFHE drop towers (slower, the fallback if the level matching misbehaves).
    Plaintext Enc(const Vec& v, const Ct& ct, bool forAdd = false) const {
        if (ptLevel0_) return cc_->MakeCKKSPackedPlaintext(v);
        if (forAdd) return cc_->MakeCKKSPackedPlaintext(v, ct->GetNoiseScaleDeg(), ct->GetLevel());
        const uint32_t lvl = ct->GetLevel() + (ct->GetNoiseScaleDeg() == 2 ? 1 : 0);
        return cc_->MakeCKKSPackedPlaintext(v, 1, lvl);
    }

    Ct Rot(const Ct& x, int32_t k) { ++c.rot; return cc_->EvalRotate(x, k); }
    Ct Mul(const Ct& x, const Ct& y) { ++c.ctct; return cc_->EvalMult(x, y); }
    Ct MulPt(const Ct& x, const Vec& v) { ++c.ctpt; return cc_->EvalMult(x, Enc(v, x)); }
    Ct MulScalar(const Ct& x, double a) { ++c.ctscalar; return cc_->EvalMult(x, a); }
    Ct Add(const Ct& x, const Ct& y) { ++c.add; return cc_->EvalAdd(x, y); }
    Ct AddPt(const Ct& x, const Vec& v) { return cc_->EvalAdd(x, Enc(v, x, true)); }

    using Rots = std::vector<std::vector<Ct>>;

    // rot(X_i, r * Bk) for r = 0..S-1, hoisted: one decomposition per input ciphertext.
    Rots BlockRotations(const std::vector<Ct>& X) {
        Rots R(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            auto pre = cc_->EvalFastRotationPrecompute(X[i]);
            R[i].push_back(X[i]);
            for (uint32_t r = 1; r < s_.S; ++r) {
                ++c.rot;
                R[i].push_back(cc_->EvalFastRotation(X[i], r * s_.Bk, M_, pre));
            }
        }
        return R;
    }

    // rows: (n_out_cts * S, Min). Y_j = sum_i sum_r diag(j,i,r) * rot(X_i, r * Bk).
    std::vector<Ct> Linear(const Rots& R, const Mat& rows, const Vec* bias = nullptr) {
        const uint32_t nout = rows.size() / s_.S, Min = rows[0].size();
        std::vector<Ct> Y;
        for (uint32_t j = 0; j < nout; ++j) {
            Ct acc;
            for (uint32_t i = 0; i < R.size(); ++i)
                for (uint32_t r = 0; r < s_.S; ++r) {
                    Vec coef(s_.S, 0.0);
                    bool any = false;
                    for (uint32_t b = 0; b < s_.S; ++b) {
                        const uint32_t col = i * s_.S + (b + r) % s_.S;
                        if (col < Min) { coef[b] = rows[j * s_.S + b][col]; any = any || coef[b] != 0; }
                    }
                    if (!any) continue;
                    Ct t = MulPt(R[i][r], BlockConst(coef));
                    acc = acc ? Add(acc, t) : t;
                }
            if (bias) {
                Vec bb(s_.S);
                for (uint32_t b = 0; b < s_.S; ++b) bb[b] = (*bias)[j * s_.S + b];
                acc = AddPt(acc, BlockConst(bb));
            }
            Y.push_back(acc);
        }
        return Y;
    }

    // Causal depthwise conv; taps (n_cts * S, K) aligned with X's blocks; tap k = EvalRotate(-k).
    std::vector<Ct> Conv(const std::vector<Ct>& X, const Mat& taps, const Vec& bias) {
        std::vector<Ct> Y;
        for (size_t j = 0; j < X.size(); ++j) {
            Ct acc;
            for (uint32_t k = 0; k < s_.K; ++k) {
                Ct xs = k == 0 ? X[j] : Rot(X[j], -static_cast<int32_t>(k));
                Vec tk(s_.S);
                for (uint32_t b = 0; b < s_.S; ++b) tk[b] = taps[j * s_.S + b][k];
                Ct t = MulPt(xs, BlockConst(tk));
                acc = acc ? Add(acc, t) : t;
            }
            Vec bb(s_.S);
            for (uint32_t b = 0; b < s_.S; ++b) bb[b] = bias[j * s_.S + b];
            Y.push_back(AddPt(acc, BlockConst(bb)));
        }
        return Y;
    }

    // Value v on block b's real slots, zero everywhere else.
    Vec OneBlock(uint32_t b, double v) const {
        Vec per(s_.S, 0.0);
        per[b] = v;
        return BlockConst(per);
    }

    // Copy the one nonzero block to every block: log2 S rotate-and-adds, no level.
    // Needs slots == S * Bk so rotating by Bk * 2^i is cyclic over blocks.
    Ct Replicate(Ct x) {
        for (uint32_t i = 1; i < s_.S; i <<= 1) x = Add(x, Rot(x, static_cast<int32_t>(i * s_.Bk)));
        return x;
    }

    Ct ActCt(const Ct& x) {
        Ct x2 = Mul(x, x);
        return Add(MulScalar(x, kAct1), MulScalar(x2, kAct2));
    }

    std::vector<Ct> Forward(const std::vector<Ct>& U) {
        const Shape& s = s_;
        const Weights& W = W_;
        const uint32_t nX = s.Di / s.S;
        auto rep = [&](const Vec& row) { return Mat(s.S, row); };
        auto tile = [&](const Vec& row) { return Mat(s.S, row); };
        auto stage = [&](const std::string& name, Clock::time_point t0, const Ct& probe) {
            stageTime.emplace_back(name, Since(t0));
            stageLevel.emplace_back(name, probe->GetLevel());
        };
        std::vector<uint32_t> sel;
        for (uint32_t h = 0; h < s.H; ++h) if (W.selective[h]) sel.push_back(h);

        auto t0 = Clock::now();
        const Rots RU = BlockRotations(U);
        std::vector<Ct> X = Linear(RU, W.Wx), Z = Linear(RU, W.Wz), Brep, Crep, Sm;
        std::vector<std::vector<Ct>> SmR;
        std::map<uint32_t, Ct> V;
        if (naive_) {
            // the first version: B, C, v each computed directly in replicated form
            for (uint32_t n = 0; n < s.N; ++n) Brep.push_back(Linear(RU, rep(W.WB[n]))[0]);
            for (uint32_t n = 0; n < s.N; ++n) Crep.push_back(Linear(RU, rep(W.WC[n]))[0]);
            for (uint32_t h : sel) {
                const Vec bb(s.S, W.bv[h]);
                V[h] = Linear(RU, rep(W.Wv[h]), &bb)[0];
            }
        } else {
            // B, C, v once each, one per block of a small stream (sim.py: forward)
            Mat rows(W.WB);
            rows.insert(rows.end(), W.WC.begin(), W.WC.end());
            for (uint32_t h : sel) rows.push_back(W.Wv[h]);
            const uint32_t nsmall = (rows.size() + s.S - 1) / s.S;
            Vec bias(nsmall * s.S, 0.0);
            for (size_t i = 0; i < sel.size(); ++i) bias[2 * s.N + i] = W.bv[sel[i]];
            rows.resize(nsmall * s.S, Vec(s.D, 0.0));
            Sm = Linear(RU, rows, &bias);
            for (size_t i = 0; i < sel.size(); ++i) {  // v: select its block (1 level), replicate
                const uint32_t m = 2 * s.N + i, j = m / s.S, b = m % s.S;
                V[sel[i]] = Replicate(MulPt(Sm[j], OneBlock(b, 1.0)));
            }
            for (auto& c : Sm) {  // shifted copies for the conv, shared by every B and C row
                std::vector<Ct> r{c};
                for (uint32_t k = 1; k < s.K; ++k) r.push_back(Rot(c, -static_cast<int32_t>(k)));
                SmR.push_back(std::move(r));
            }
        }
        stage("in_proj", t0, X[0]);

        t0 = Clock::now();
        std::vector<Ct> Xa, Ba, Ca, Za;
        // conv that keeps only row m's block (the selection rides on the taps), then replicate
        auto convSelect = [&](uint32_t m, const Vec& taps, double cb) {
            const uint32_t j = m / s.S, b = m % s.S;
            Ct acc;
            for (uint32_t k = 0; k < s.K; ++k) {
                Ct t = MulPt(SmR[j][k], OneBlock(b, taps[k]));
                acc = acc ? Add(acc, t) : t;
            }
            return Replicate(AddPt(acc, OneBlock(b, cb)));
        };
        if (naive_) {
            for (uint32_t n = 0; n < s.N; ++n) {
                Ba.push_back(ActCt(Conv({Brep[n]}, tile(W.convB[n]), Vec(s.S, W.cbB[n]))[0]));
                Ca.push_back(ActCt(Conv({Crep[n]}, tile(W.convC[n]), Vec(s.S, W.cbC[n]))[0]));
            }
        } else {
            for (uint32_t n = 0; n < s.N; ++n) Ba.push_back(ActCt(convSelect(n, W.convB[n], W.cbB[n])));
            for (uint32_t n = 0; n < s.N; ++n) Ca.push_back(ActCt(convSelect(s.N + n, W.convC[n], W.cbC[n])));
        }
        for (auto& x : Conv(X, W.convx, W.cbx)) Xa.push_back(ActCt(x));
        for (auto& z : Z) Za.push_back(ActCt(z));
        stage("conv+act", t0, Xa[0]);

        t0 = Clock::now();
        std::map<uint32_t, Ct> A, Dlt;
        for (uint32_t h : sel) {
            Ct w = Mul(V[h], V[h]);
            A[h] = cc_->EvalChebyshevSeries(w, W.decay[h], 0.0, W.wmax[h]);
            Dlt[h] = MulScalar(w, 1.0 / W.absA[h]);
        }
        auto headOf = [&](uint32_t g) { return (g * s.S) / s.P; };
        std::vector<Ct> XD;
        for (uint32_t g = 0; g < nX; ++g)
            XD.push_back(W.selective[headOf(g)] ? Mul(Xa[g], Dlt[headOf(g)]) : Xa[g]);
        std::map<std::pair<uint32_t, uint32_t>, Ct> Bs;
        for (uint32_t n = 0; n < s.N; ++n)
            for (uint32_t g = 0; g < nX; ++g) Bs[{n, g}] = Mul(XD[g], Ba[n]);
        stage("decay+b", t0, Bs.begin()->second);

        t0 = Clock::now();
        for (uint32_t k = 0; k < s.lgL; ++k) {
            const int32_t sh = 1 << k;
            for (auto& [key, b] : Bs) {
                const uint32_t h = headOf(key.second);
                Ct shifted = Rot(b, -sh);
                b = W.selective[h] ? Add(b, Mul(A[h], shifted))
                                   : Add(b, MulScalar(shifted, std::pow(W.aPub[h], sh)));
            }
            for (uint32_t h : sel) A[h] = Mul(A[h], Rot(A[h], -sh));
        }
        stage("scan", t0, Bs.begin()->second);

        t0 = Clock::now();
        std::vector<Ct> G;
        for (uint32_t g = 0; g < nX; ++g) {
            Ct y;
            for (uint32_t n = 0; n < s.N; ++n) {
                Ct t = Mul(Ca[n], Bs[{n, g}]);
                y = y ? Add(y, t) : t;
            }
            y = Add(y, MulScalar(Xa[g], W.Dskip[headOf(g)]));
            G.push_back(Mul(y, Za[g]));
        }
        stage("readout+gate", t0, G[0]);

        t0 = Clock::now();
        auto out = Linear(BlockRotations(G), W.Wout);
        stage("out_proj", t0, out[0]);
        return out;
    }

private:
    CryptoContext<DCRTPoly> cc_;
    const Shape& s_;
    const Weights& W_;
    bool ptLevel0_, naive_;
    uint32_t M_;
};

// Encrypt-side sanity check of the plaintext level matching before spending minutes on a block.
bool CheckPtLevels(CryptoContext<DCRTPoly> cc, const KeyPair<DCRTPoly>& keys, uint32_t slots, bool ptLevel0) {
    Vec x(slots), p(slots), q(slots);
    for (uint32_t i = 0; i < slots; ++i) { x[i] = std::sin(i * 0.01); p[i] = 0.5 + 0.001 * (i % 7); q[i] = 0.1; }
    auto ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(x));
    auto sq = cc->EvalMult(ct, ct);  // degree 2, not yet rescaled under FLEXIBLEAUTO
    auto enc = [&](const Vec& v, const Ct& c, bool add) {
        if (ptLevel0) return cc->MakeCKKSPackedPlaintext(v);
        if (add) return cc->MakeCKKSPackedPlaintext(v, c->GetNoiseScaleDeg(), c->GetLevel());
        return cc->MakeCKKSPackedPlaintext(v, 1, c->GetLevel() + (c->GetNoiseScaleDeg() == 2 ? 1 : 0));
    };
    auto m = cc->EvalMult(sq, enc(p, sq, false));
    auto r = cc->EvalAdd(m, enc(q, m, true));
    Plaintext out;
    cc->Decrypt(keys.secretKey, r, &out);
    out->SetLength(slots);
    const auto v = out->GetRealPackedValue();
    double worst = 0;
    for (uint32_t i = 0; i < slots; ++i) worst = std::max(worst, std::fabs(v[i] - (x[i] * x[i] * p[i] + q[i])));
    std::cout << "plaintext level check: worst abs error " << worst << (worst < 1e-5 ? "  ok\n" : "  FAILED\n");
    return worst < 1e-5;
}

int RunBoot(const Args& A, const Shape& s, uint32_t levelsAfter) {
    std::vector<uint32_t> budget = {A.budget0, A.budget1};
    if (A.levelsAfter) levelsAfter = A.levelsAfter;
    // iterative bootstrapping consumes one more level per extra iteration (OpenFHE's example)
    const uint32_t depth =
        levelsAfter + FHECKKSRNS::GetBootstrapDepth(budget, UNIFORM_TERNARY) + (A.bootIters - 1);
    CCParams<CryptoContextCKKSRNS> P;
    P.SetSecretKeyDist(UNIFORM_TERNARY);
    P.SetMultiplicativeDepth(depth);
    P.SetScalingModSize(A.bootScale);
    P.SetFirstModSize(60);
    P.SetScalingTechnique(FLEXIBLEAUTO);
    P.SetKeySwitchTechnique(HYBRID);
    P.SetSecurityLevel(A.toy ? HEStd_NotSet : HEStd_128_classic);
    if (A.toy) P.SetRingDim(A.ring ? A.ring : 8192);
    else if (A.ring) P.SetRingDim(A.ring);
    P.SetBatchSize(s.slots);
    auto cc = GenCryptoContext(P);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) cc->Enable(f);
    std::cout << "== bootstrap probe: levels after bootstrap " << levelsAfter << ", budget {" << A.budget0 << ","
              << A.budget1 << "}, iterations " << A.bootIters << ", scale " << A.bootScale << " bits, total depth " << depth
              << ", ring " << cc->GetRingDimension() << ", log2 Q " << cc->GetModulus().GetMSB() << "\n";
    auto t0 = Clock::now();
    cc->EvalBootstrapSetup(budget, {0, 0}, s.slots);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapKeyGen(keys.secretKey, s.slots);
    std::cout << "setup + keygen " << Since(t0) << " s, peak RSS " << PeakRssGB() << " GB\n";
    Vec x(s.slots);
    for (uint32_t i = 0; i < s.slots; ++i) x[i] = 0.3 * std::sin(i * 0.37);
    auto ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(x, 1, depth - 1));
    auto measure = [&](const Ct& c, double& worstBits, double& rmsBits) {
        Plaintext out;
        cc->Decrypt(keys.secretKey, c, &out);
        out->SetLength(s.slots);
        const auto v = out->GetRealPackedValue();
        double worst = 0, ss = 0;
        for (uint32_t i = 0; i < s.slots; ++i) {
            const double e = std::fabs(v[i] - x[i]);
            worst = std::max(worst, e);
            ss += e * e;
        }
        worstBits = -std::log2(worst);
        rmsBits = -std::log2(std::sqrt(ss / s.slots));
    };
    t0 = Clock::now();
    auto bt = cc->EvalBootstrap(ct);
    double tb = Since(t0), wb = 0, rb = 0;
    measure(bt, wb, rb);
    std::cout << "1 iteration: " << tb << " s, worst " << wb << " bits, rms " << rb << " bits, levels left "
              << depth - bt->GetLevel() << "\n";
    if (A.bootIters > 1) {
        // the second iteration corrects the first one's error; it needs that error's size in bits
        const uint32_t prec = static_cast<uint32_t>(std::floor(rb));
        t0 = Clock::now();
        bt = cc->EvalBootstrap(ct, A.bootIters, prec);
        tb = Since(t0);
        measure(bt, wb, rb);
        std::cout << A.bootIters << " iterations (precision hint " << prec << "): " << tb << " s, worst " << wb
                  << " bits, rms " << rb << " bits, levels left " << depth - bt->GetLevel() << "\n";
    }
    const uint32_t nOut = (s.D + s.S - 1) / s.S;
    std::cout << "SUMMARY ring " << cc->GetRingDimension() << " budget " << A.budget0 << "," << A.budget1
              << " iters " << A.bootIters << " scale " << A.bootScale << " levels_after_req " << levelsAfter
              << " levels_left " << depth - bt->GetLevel() << " boot_s " << tb << " worst_bits " << wb
              << " rms_bits " << rb << " per_block_s " << nOut * tb << " peak_rss_gb " << PeakRssGB() << "\n";
    return 0;
}

// N blocks in a row with a bootstrap of every output ciphertext between blocks, all in one
// context with bootstrapping enabled. The input is encrypted with exactly the levels a
// bootstrap leaves, so block 1 runs under the same conditions as every later block.
int RunStack(const Args& A, Shape s) {
    // +1 over the standalone block: its last output is left unrescaled, and the bootstrap needs a level to rescale it
    const uint32_t levelsAfter = A.levelsAfter ? A.levelsAfter : (A.depth ? A.depth : 11 + s.lgL);
    std::vector<uint32_t> budget = {A.budget0, A.budget1};
    const uint32_t iters = std::max<uint32_t>(A.bootIters, 1);
    const uint32_t depth = levelsAfter + FHECKKSRNS::GetBootstrapDepth(budget, UNIFORM_TERNARY) + (iters - 1);
    CCParams<CryptoContextCKKSRNS> P;
    P.SetSecretKeyDist(UNIFORM_TERNARY);
    P.SetMultiplicativeDepth(depth);
    P.SetScalingModSize(A.bootScale);
    P.SetFirstModSize(60);
    P.SetScalingTechnique(FLEXIBLEAUTO);
    P.SetKeySwitchTechnique(HYBRID);
    P.SetSecurityLevel(A.toy ? HEStd_NotSet : HEStd_128_classic);
    if (A.toy) P.SetRingDim(A.ring ? A.ring : 8192);
    else if (A.ring) P.SetRingDim(A.ring);
    auto cc = GenCryptoContext(P);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) cc->Enable(f);
    s.slots = cc->GetRingDimension() / 2;
    s.S = s.slots / s.Bk;
    if (s.S == 0 || s.S * s.Bk != s.slots || (s.S & (s.S - 1)) || s.P % s.S != 0 || s.Di % s.S != 0 || s.D % s.S != 0) {
        std::cerr << "layout needs S = slots/(2L) >= 1 dividing P, D and d_inner; got S = " << s.S << "\n";
        return 2;
    }
    std::cout << std::setprecision(4) << "== " << A.stack << " stacked HE-native SSM blocks, OpenFHE CKKS "
              << (A.toy ? "TOY (insecure)" : "128-bit") << ", threads " << omp_get_max_threads() << "\n"
              << "ring " << cc->GetRingDimension() << ", slots " << s.slots << ", total depth " << depth
              << " (block " << levelsAfter << " + bootstrap {" << A.budget0 << "," << A.budget1 << "} x" << iters
              << "), scale " << A.bootScale << " bits, log2 Q " << cc->GetModulus().GetMSB() << "\n"
              << "D " << s.D << ", L " << s.L << ", d_inner " << s.Di << ", heads " << s.H << ", headdim " << s.P
              << ", d_state " << s.N << ", " << s.S << " sequences per ciphertext\n";

    std::set<int32_t> idx;
    for (uint32_t r = 1; r < s.S; ++r) idx.insert(r * s.Bk);
    for (uint32_t k = 1; k < s.K; ++k) idx.insert(-static_cast<int32_t>(k));
    for (uint32_t k = 0; k < s.lgL; ++k) idx.insert(-(1 << k));
    auto t0 = Clock::now();
    cc->EvalBootstrapSetup(budget, {0, 0}, s.slots);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, std::vector<int32_t>(idx.begin(), idx.end()));
    cc->EvalBootstrapKeyGen(keys.secretKey, s.slots);
    std::cout << "setup + keygen (" << idx.size() << " block rotation keys + bootstrap keys) " << Since(t0)
              << " s, peak RSS " << PeakRssGB() << " GB\n";
    if (!CheckPtLevels(cc, keys, s.slots, A.ptLevel0)) return 1;

    std::vector<Weights> Ws;
    for (uint32_t l = 0; l < A.stack; ++l) Ws.push_back(MakeWeights(s, A.selectiveHeads, A.seed + 1000 * l));
    std::mt19937_64 rng(A.seed + 1);
    std::uniform_real_distribution<double> Uu(-1, 1);
    Mat u(s.D, Vec(s.L));
    for (auto& r : u) for (auto& x : r) x = Uu(rng);

    auto pack = [&](const Mat& m) {
        std::vector<Ct> cts;
        for (uint32_t j = 0; j * s.S < s.D; ++j) {
            Vec v(s.slots, 0.0);
            for (uint32_t b = 0; b < s.S && j * s.S + b < s.D; ++b)
                std::copy(m[j * s.S + b].begin(), m[j * s.S + b].end(), v.begin() + b * s.Bk + s.L);
            cts.push_back(cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v, 1, depth - levelsAfter)));
        }
        return cts;
    };
    auto unpack = [&](const std::vector<Ct>& cts) {
        Mat got(s.D, Vec(s.L));
        for (uint32_t j = 0; j < cts.size(); ++j) {
            Plaintext pt;
            cc->Decrypt(keys.secretKey, cts[j], &pt);
            pt->SetLength(s.slots);
            const auto v = pt->GetRealPackedValue();
            for (uint32_t b = 0; b < s.S && j * s.S + b < s.D; ++b)
                for (uint32_t t = 0; t < s.L; ++t) got[j * s.S + b][t] = v[b * s.Bk + s.L + t];
        }
        return got;
    };
    auto relErr = [&](const Mat& got, const Mat& ref) {
        double worst = 0, scale = 0;
        for (uint32_t d = 0; d < s.D; ++d)
            for (uint32_t t = 0; t < s.L; ++t) {
                worst = std::max(worst, std::fabs(got[d][t] - ref[d][t]));
                scale = std::max(scale, std::fabs(ref[d][t]));
            }
        return worst / scale;
    };
    auto report = [&](const std::string& what, const Mat& got, const Mat& ref) {
        const double e = relErr(got, ref);
        std::cout << what << ": max error " << e << " relative to max |out| (" << -std::log2(e) << " bits)\n";
        return e;
    };

    std::vector<Ct> X = pack(u);
    Mat ref = u;
    double tBlocks = 0, tBoot = 0, eLast = 0;
    for (uint32_t l = 0; l < A.stack; ++l) {
        if (l > 0) {
            // bootstrap every ciphertext the previous block produced
            uint32_t prec = A.bootPrec;
            t0 = Clock::now();
            std::vector<Ct> Y;
            for (auto& c : X) {
                if (iters > 1 && prec == 0) {  // first time: measure one iteration's precision
                    auto one = cc->EvalBootstrap(c);
                    Plaintext a, b;
                    cc->Decrypt(keys.secretKey, c, &a);
                    cc->Decrypt(keys.secretKey, one, &b);
                    a->SetLength(s.slots);
                    b->SetLength(s.slots);
                    const auto va = a->GetRealPackedValue(), vb = b->GetRealPackedValue();
                    double ss = 0;
                    for (uint32_t i = 0; i < s.slots; ++i) ss += std::pow(va[i] - vb[i], 2);
                    prec = static_cast<uint32_t>(std::floor(-std::log2(std::sqrt(ss / s.slots))));
                    std::cout << "measured 1-iteration bootstrap precision: " << prec << " bits (rms)\n";
                    t0 = Clock::now();  // don't charge the measurement to the bootstrap time
                }
                Y.push_back(iters > 1 ? cc->EvalBootstrap(c, iters, prec) : cc->EvalBootstrap(c));
            }
            const double tb = Since(t0);
            tBoot += tb;
            X = std::move(Y);
            std::cout << "\nbootstrap " << X.size() << " ciphertexts: " << tb << " s (" << tb / X.size()
                      << " s each), levels left " << depth - X[0]->GetLevel() << ", peak RSS " << PeakRssGB()
                      << " GB\n";
            report("  after bootstrap", unpack(X), ref);
        }
        Block blk(cc, s, Ws[l], A.ptLevel0, A.naiveInproj);
        t0 = Clock::now();
        try {
            X = blk.Forward(X);
        } catch (const std::exception& e) {
            std::cerr << "block " << l + 1 << " failed: " << e.what() << "\n";
            return 1;
        }
        const double t = Since(t0);
        tBlocks += t;
        ref = Reference(s, Ws[l], ref);
        std::cout << "\nblock " << l + 1 << ": " << t << " s, levels used " << X[0]->GetLevel() - (depth - levelsAfter)
                  << " of " << levelsAfter << ";";
        for (size_t i = 0; i < blk.stageTime.size(); ++i)
            std::cout << " " << blk.stageTime[i].first << " " << blk.stageTime[i].second;
        std::cout << "\n";
        eLast = report("  after block " + std::to_string(l + 1), unpack(X), ref);
    }
    std::cout << "\nSUMMARY blocks " << A.stack << " ring " << cc->GetRingDimension() << " block_s " << tBlocks
              << " boot_s " << tBoot << " total_s " << tBlocks + tBoot << " per_block_s " << (tBlocks + tBoot) / A.stack
              << " final_rel_err " << eLast << " final_bits " << -std::log2(eLast) << " peak_rss_gb " << PeakRssGB()
              << "\n";
    return eLast < 1e-3 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const Args A = Parse(argc, argv);
    if (A.threads > 0) omp_set_num_threads(A.threads);
    Shape s{};
    s.D = A.D; s.L = A.L; s.P = A.P; s.N = A.N; s.K = A.K;
    s.Di = A.expand * A.D; s.H = s.Di / s.P; s.Bk = 2 * s.L;
    s.lgL = static_cast<uint32_t>(std::lround(std::log2(s.L)));
    if ((1u << s.lgL) != s.L) { std::cerr << "L must be a power of two\n"; return 2; }

    if (A.stack) return RunStack(A, s);
    const uint32_t depth = A.depth ? A.depth : 10 + s.lgL;  // sim.py: 9 + log2 L consumed, +1 margin
    CCParams<CryptoContextCKKSRNS> P;
    P.SetMultiplicativeDepth(depth);
    P.SetScalingModSize(50);
    P.SetFirstModSize(60);
    P.SetScalingTechnique(FLEXIBLEAUTO);
    P.SetKeySwitchTechnique(HYBRID);
    P.SetSecurityLevel(A.toy ? HEStd_NotSet : HEStd_128_classic);
    if (A.toy) P.SetRingDim(A.ring ? A.ring : 8192);
    else if (A.ring) P.SetRingDim(A.ring);
    auto cc = GenCryptoContext(P);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE}) cc->Enable(f);
    s.slots = cc->GetRingDimension() / 2;
    s.S = s.slots / s.Bk;
    if (s.S == 0 || s.S * s.Bk != s.slots || (s.S & (s.S - 1)) || s.P % s.S != 0 || s.Di % s.S != 0 || s.D % s.S != 0) {
        std::cerr << "layout needs S = slots/(2L) >= 1 dividing P, D and d_inner; got S = " << s.S << "\n";
        return 2;
    }
    if (A.boot) return RunBoot(A, s, depth);

    const Weights W = MakeWeights(s, A.selectiveHeads, A.seed);
    uint32_t nsel = 0;
    for (bool b : W.selective) nsel += b;
    std::cout << std::setprecision(4)
              << "== HE-native SSM block, OpenFHE CKKS " << (A.toy ? "TOY (insecure)" : "128-bit")
              << ", threads " << omp_get_max_threads() << "\n"
              << "ring " << cc->GetRingDimension() << ", slots " << s.slots << ", depth " << depth
              << ", log2 Q " << cc->GetModulus().GetMSB() << "\n"
              << "D " << s.D << ", L " << s.L << ", d_inner " << s.Di << ", heads " << s.H << " (selective " << nsel
              << "), headdim " << s.P << ", d_state " << s.N << ", " << s.S << " sequences per ciphertext\n";

    // rotation keys: block shifts for the linear layers, -1..-(K-1) for conv, -2^k for the scan
    std::set<int32_t> idx;
    for (uint32_t r = 1; r < s.S; ++r) idx.insert(r * s.Bk);
    for (uint32_t k = 1; k < s.K; ++k) idx.insert(-static_cast<int32_t>(k));
    for (uint32_t k = 0; k < s.lgL; ++k) idx.insert(-(1 << k));
    auto t0 = Clock::now();
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, std::vector<int32_t>(idx.begin(), idx.end()));
    std::cout << "keygen (" << idx.size() << " rotation keys) " << Since(t0) << " s, peak RSS " << PeakRssGB()
              << " GB\n";
    if (!CheckPtLevels(cc, keys, s.slots, A.ptLevel0)) {
        std::cerr << "plaintext level matching is wrong for this OpenFHE build; rerun with --pt-level0\n";
        return 1;
    }

    // client: pack u (D sequences) into ceil(D / S) ciphertexts, pads zero
    std::mt19937_64 rng(A.seed + 1);
    std::uniform_real_distribution<double> Uu(-1, 1);
    Mat u(s.D, Vec(s.L));
    for (auto& r : u) for (auto& x : r) x = Uu(rng);
    Block blk(cc, s, W, A.ptLevel0, A.naiveInproj);
    std::vector<Ct> U;
    t0 = Clock::now();
    for (uint32_t j = 0; j * s.S < s.D; ++j) {
        Vec v(s.slots, 0.0);
        for (uint32_t b = 0; b < s.S && j * s.S + b < s.D; ++b)
            std::copy(u[j * s.S + b].begin(), u[j * s.S + b].end(), v.begin() + b * s.Bk + s.L);
        U.push_back(cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v)));
    }
    const double tEnc = Since(t0);

    std::vector<Ct> out;
    t0 = Clock::now();
    try {
        out = blk.Forward(U);
    } catch (const std::exception& e) {
        std::cerr << "evaluation failed: " << e.what() << "\n(if this is a level error, raise --depth)\n";
        return 1;
    }
    const double tBlock = Since(t0);

    t0 = Clock::now();
    Mat got(s.D, Vec(s.L));
    for (uint32_t j = 0; j < out.size(); ++j) {
        Plaintext pt;
        cc->Decrypt(keys.secretKey, out[j], &pt);
        pt->SetLength(s.slots);
        const auto v = pt->GetRealPackedValue();
        for (uint32_t b = 0; b < s.S && j * s.S + b < s.D; ++b)
            for (uint32_t t = 0; t < s.L; ++t) got[j * s.S + b][t] = v[b * s.Bk + s.L + t];
    }
    const double tDec = Since(t0);

    const Mat ref = Reference(s, W, u);
    double worst = 0, scale = 0, ss = 0;
    for (uint32_t d = 0; d < s.D; ++d)
        for (uint32_t t = 0; t < s.L; ++t) {
            worst = std::max(worst, std::fabs(got[d][t] - ref[d][t]));
            scale = std::max(scale, std::fabs(ref[d][t]));
            ss += std::pow(got[d][t] - ref[d][t], 2);
        }

    std::cout << "\n" << std::left << std::setw(16) << "stage" << std::right << std::setw(10) << "time s"
              << std::setw(16) << "level after" << "\n";
    for (size_t i = 0; i < blk.stageTime.size(); ++i)
        std::cout << std::left << std::setw(16) << blk.stageTime[i].first << std::right << std::setw(10)
                  << blk.stageTime[i].second << std::setw(16) << blk.stageLevel[i].second << "\n";
    std::cout << "\nblock " << tBlock << " s (client encrypt " << tEnc << " s, decrypt " << tDec << " s)\n"
              << "levels consumed " << out[0]->GetLevel() << " of " << depth << "  (sim.py predicts "
              << 9 + s.lgL - (nsel == 0 ? 1 : 0) << ")\n"
              << "ops: ct-ct " << blk.c.ctct << ", ct-pt " << blk.c.ctpt << ", ct-scalar " << blk.c.ctscalar
              << ", rotations " << blk.c.rot << ", adds " << blk.c.add << "  (compare sim.py)\n"
              << "max abs error " << worst << " = " << worst / scale << " relative to max |out| ("
              << -std::log2(worst / scale) << " bits), rms " << std::sqrt(ss / (s.D * s.L)) << "\n"
              << "peak RSS " << PeakRssGB() << " GB\n";
    return worst / scale < 1e-3 ? 0 : 1;
}
