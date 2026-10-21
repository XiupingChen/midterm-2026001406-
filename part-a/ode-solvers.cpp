// Complete only euler() and improvedEuler(), then add your observations at the end.
// The remaining code is for testing and does not need modification.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

double f(double t, double y) { return y - t * t + 1.0; }
double exactSolution(double t) {
    return (t + 1.0) * (t + 1.0) - 0.5 * std::exp(t);
}

std::vector<double> euler(const std::vector<double>& times, double initial) {
    std::vector<double> result(times.size());
    if (times.empty()) return result;
    result[0] = initial;
    for (std::size_t i = 0; i + 1 < times.size(); ++i) {
        const double h = times[i + 1] - times[i];
        result[i + 1] = result[i] + h * f(times[i], result[i]);
    }
    return result;
}

std::vector<double> improvedEuler(const std::vector<double>& times, double initial) {
    std::vector<double> result(times.size());
    if (times.empty()) return result;
    result[0] = initial;
    for (std::size_t i = 0; i + 1 < times.size(); ++i) {
        const double h = times[i + 1] - times[i];
        const double slope = f(times[i], result[i]);
        const double predicted = result[i] + h * slope;
        result[i + 1] = result[i] + 0.5 * h *
            (slope + f(times[i + 1], predicted));
    }
    return result;
}

double maxError(const std::vector<double>& result,
                const std::vector<double>& truth) {
    if (result.empty() || result.size() != truth.size() ||
        std::abs(result.front() - 0.5) > 1e-12) {
        throw std::runtime_error("Check output size and initial value.");
    }
    double error = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i])) throw std::runtime_error("Non-finite result.");
        error = std::max(error, std::abs(result[i] - truth[i]));
    }
    return error;
}

// Provided discrete-solution fixtures for h = 0.1 and h = 0.05.
// These are numerical-method results, not samples of the exact solution.
const std::vector<std::vector<double>> eulerReference{
    {
        0.50000000000000000, 0.65000000000000002, 0.81400000000000006,
        0.99140000000000006, 1.1815400000000000, 1.3836940000000000,
        1.5970634000000001, 1.8207697400000002, 2.0538467140000001,
        2.2952313854000002, 2.5437545239400001, 2.7981299763340002,
        3.0569429739674003, 3.3186372713641403, 3.5815009985005544,
        3.8436510983506098, 4.1030162081856707, 4.3573178290042378,
        4.6040496119046619, 4.8404545730951281, 5.0635000304046409
    },
    {
        0.50000000000000000, 0.57499999999999996, 0.65362499999999990,
        0.73580624999999988, 0.82147156249999986, 0.91054514062499980,
        1.0029473976562497, 1.0985947675390622, 1.1973995059160152,
        1.2992694812118160, 1.4041079552724067, 1.5118133530360272,
        1.6222790206878286, 1.7353929717222201, 1.8510376203083310,
        1.9690895013237475, 2.0894189763899349, 2.2118899252094315,
        2.3363594214699033, 2.4626773925433985, 2.5906862621705682,
        2.7202205752790967, 2.8511066040430517, 2.9831619342452043,
        3.1161950309574644, 3.2500047825053375, 3.3843800216306046,
        3.5190990227121350, 3.6539289738477416, 3.7886254225401288,
        3.9229316936671350, 4.0565782783504920, 4.1892821922680170,
        4.3207463018814174, 4.4506586169754883, 4.5786915478242625,
        4.7045011252154758, 4.8277261814762493, 4.9479874905500614,
        5.0648868650775647, 5.1780062083314427
    }
};
const std::vector<std::vector<double>> improvedEulerReference{
    {
        0.50000000000000000, 0.65700000000000003, 0.82843500000000003,
        1.0137206750000001, 1.2122113458750001, 1.4231935371918751,
        1.6458788585970221, 1.8793961387497093, 2.1227827333184290,
        2.3749749203168640, 2.6347972869501346, 2.9009510020798985,
        3.1720008572982881, 3.4463609473146084, 3.7222788467826424,
        3.9978181256948200, 4.2708390288927758, 4.5389771269265173,
        4.7996197252538018, 5.0498797964054507, 5.2865671750280230
    },
    {
        0.50000000000000000, 0.57681249999999995, 0.65730851562499992,
        0.74142057705078113, 0.82907775662463368, 0.92020549165164622,
        1.0147253980987931, 1.1125550747513562, 1.2136078973323632,
        1.3177928020706469, 1.4250140581767674, 1.5351710286583267,
        1.6481579188770659, 1.7638635122195154, 1.8821708922207656,
        2.0029571504470800, 2.1260930794074930, 2.2514428497271268,
        2.3788636707756421, 2.5082054339028939, 2.6393103373904170,
        2.7720124921816760, 2.9061375074059868, 3.0415020546605436,
        3.1779134099618966, 3.3151689722224438, 3.4530557570488440,
        3.5913498645975972, 3.7298159201582242, 3.8682064860663332,
        4.0062614434772330, 4.1437073424554409, 4.2802567187562826,
        4.4156073755925416, 4.5494416285916595, 4.6814255120569817,
        4.8112079445499019, 4.9384198517080842, 5.0626732441081232,
        5.1835602478686642, 5.3006520855719330
    }
};

double checkedMaxError(const std::vector<double>& result,
                       const std::vector<double>& truth,
                       const std::vector<double>& reference) {
    const double error = maxError(result, truth);
    if (result.size() != reference.size()) {
        throw std::runtime_error("Incorrect size for the discrete reference.");
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (std::abs(result[i] - reference[i]) > 1e-8) {
            throw std::runtime_error("Numerical solution differs from the discrete reference.");
        }
    }
    return error;
}

int main() {
    const double initial = 0.5;
    const std::vector<double> stepSizes{0.1, 0.05};
    int failures = 0;
    for (std::size_t test = 0; test < stepSizes.size(); ++test) {
        const double h = stepSizes[test];
        const int steps = static_cast<int>(std::round(2.0 / h));
        std::vector<double> times(steps + 1), truth(steps + 1);
        for (int i = 0; i <= steps; ++i) {
            times[i] = i * h;
            truth[i] = exactSolution(times[i]);
        }
        std::cout << "h = " << h << ", grid size = " << times.size() << '\n';
        try {
            std::cout << "Euler maximum absolute error: "
                      << checkedMaxError(euler(times, initial), truth,
                                         eulerReference[test]) << '\n';
        } catch (const std::exception& error) {
            std::cerr << "Euler: " << error.what() << '\n';
            ++failures;
        }
        try {
            std::cout << "Improved Euler maximum absolute error: "
                      << checkedMaxError(improvedEuler(times, initial), truth,
                                         improvedEulerReference[test]) << '\n';
        } catch (const std::exception& error) {
            std::cerr << "Improved Euler: " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}

// The improved Euler method has smaller maximum error than Euler at both step sizes.
// Halving h roughly halves Euler's error and quarters improved Euler's error.
