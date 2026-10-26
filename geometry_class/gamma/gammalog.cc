// gammalog.cc -- Geant4 natural gamma-ray log simulation for PyBorehole.
//
// A centred NaI(Tl) logging tool in a vertical borehole through a 1-D layered
// earth.  The model is axisymmetric, so it is built as a stack of coaxial
// (r, z) tubes: earth layers, borehole fill zones, the steel tool housing and
// the crystal.  The geometry is rebuilt for every tool station.
//
// For every station and every nuclide group (K, U series, Th series) the
// program samples gamma emission points in a sphere around the crystal
// centre, tracks the photons and tallies the energy deposited in the crystal
// (Gaussian-smeared to the NaI resolution).  Emission points are importance
// sampled with density ~ exp(-s/lambda) / s at distance s, and every photon
// carries the weight activity / sampling density, so the tallies are unbiased
// absolute count rates (counts/s) for the given activities.  During transport
// a weight window on the same importance function splits photons that move
// towards the crystal and plays Russian roulette with photons that move away.
// Every split copy is scored as its own pulse (branch); the counting error is
// computed per emitted photon (history), so correlated copies are accounted for.
//
// Usage:  gammalog <input file>
// The input format is written by geometry_class/gamma_geometry.py.

#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserStackingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VUserTrackInformation.hh"
#include "G4TrackingManager.hh"
#include "G4SteppingManager.hh"
#include "G4DynamicParticle.hh"
#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmParameters.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Event.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4ThreeVector.hh"
#include "G4UImanager.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration (all lengths in metres, energies in MeV, activities in
// parent decays / s / m^3)
// ---------------------------------------------------------------------------
struct Region {
  int mat = -1;
  std::vector<double> act;   // one value per nuclide group
};

struct Config {
  long seed = 1;
  long events = 100000;      // tracked emissions per station and group
  double smax = 1.0;         // source sphere radius around the crystal centre
  double lambda = 0.15;      // importance-sampling length
  double margin = 0.5;       // transport medium beyond the source sphere
  double fwhm662 = 0.07;     // NaI FWHM / E at 662 keV
  int nbins = 300;
  double emax = 3.0;
  double threshold = 0.1;    // gross-count threshold

  std::map<int, G4Material*> mats;
  int airMat = -1;
  std::vector<std::string> groups;
  std::vector<std::vector<std::pair<double, double>>> lines;   // (E, yield)
  std::vector<std::string> winNames;
  std::vector<std::pair<double, double>> windows;

  struct Layer { double top, bot; Region reg; };
  struct Zone { double rout; Region reg; };
  std::vector<Layer> layers;   // last layer extends downwards
  std::vector<Zone> zones;     // borehole fill, inside -> out
  double holeLen = 0.0;

  // tool: crystal radius / length, housing inner / outer radius,
  // housing extent below / above the crystal centre
  double rc = 0.019, lc = 0.102, rhi = 0.022, rho = 0.025;
  double below = 0.3, above = 0.5;
  int matCrystal = -1, matHousing = -1, matInner = -1;

  std::vector<double> stations;   // crystal-centre depths (positive down)
  std::string out = "gamma_out.csv", spectra = "gamma_spectra.csv";

  // Importance of a photon at distance d (m) from the crystal centre; it is
  // the inverse of the source weight at that distance, so a photon emitted
  // anywhere reaches the crystal with about the same weight.
  double Importance(double d) const {
    return std::exp(-d / lambda) / std::max(d, std::hypot(rc, 0.5 * lc));
  }

  // Material and radioactive region at radius r, depth d, with the crystal
  // centre at zc.  Returns nullptr where nothing emits (tool, air); mat = -1
  // means vacuum.
  const Region* RegionAt(double r, double d, double zc, int& mat) const {
    if (std::abs(d - zc) <= 0.5 * lc && r < rc) { mat = matCrystal; return nullptr; }
    if (d >= zc - above && d <= zc + below) {
      if (r < rhi) { mat = matInner; return nullptr; }
      if (r < rho) { mat = matHousing; return nullptr; }
    }
    if (d < 0) { mat = airMat; return nullptr; }
    if (d < holeLen && !zones.empty() && r < zones.back().rout) {
      for (const auto& z : zones)
        if (r < z.rout) { mat = z.reg.mat; return &z.reg; }
    }
    for (const auto& l : layers)
      if (d < l.bot || &l == &layers.back()) { mat = l.reg.mat; return &l.reg; }
    mat = -1;
    return nullptr;
  }
};

static void Fail(const std::string& msg) {
  std::cerr << "gammalog: " << msg << std::endl;
  std::exit(1);
}

static Region ReadRegion(std::istringstream& is, size_t ngroups) {
  Region r;
  is >> r.mat;
  r.act.resize(ngroups);
  for (auto& a : r.act) is >> a;
  return r;
}

static Config ReadConfig(const std::string& fname) {
  std::ifstream in(fname);
  if (!in) Fail("cannot open " + fname);
  Config c;
  auto* nist = G4NistManager::Instance();
  std::string line;
  while (std::getline(in, line)) {
    if (auto p = line.find('#'); p != std::string::npos) line.erase(p);
    std::istringstream is(line);
    std::string key;
    if (!(is >> key)) continue;
    if (key == "seed") is >> c.seed;
    else if (key == "events") is >> c.events;
    else if (key == "source") is >> c.smax >> c.lambda >> c.margin;
    else if (key == "resolution") is >> c.fwhm662;
    else if (key == "spectrum") is >> c.nbins >> c.emax >> c.threshold;
    else if (key == "material") {
      int id; std::string kind;
      is >> id >> kind;
      if (kind == "nist") {
        std::string name; is >> name;
        c.mats[id] = nist->FindOrBuildMaterial(name);
        if (!c.mats[id]) Fail("unknown NIST material " + name);
      } else if (kind == "mix") {
        double rho; int n; is >> rho >> n;
        auto* m = new G4Material("mat" + std::to_string(id), rho * g / cm3, n);
        for (int i = 0; i < n; ++i) {
          int Z; double f; is >> Z >> f;
          m->AddElement(nist->FindOrBuildElement(Z), f);
        }
        c.mats[id] = m;
      } else Fail("bad material kind " + kind);
    }
    else if (key == "air") is >> c.airMat;
    else if (key == "group") {
      std::string name; int n; is >> name >> n;
      std::vector<std::pair<double, double>> l(n);
      for (auto& e : l) is >> e.first >> e.second;
      c.groups.push_back(name);
      c.lines.push_back(l);
    }
    else if (key == "window") {
      std::string name; double a, b; is >> name >> a >> b;
      c.winNames.push_back(name); c.windows.push_back({a, b});
    }
    else if (key == "layer") {
      Config::Layer l; is >> l.top >> l.bot;
      l.reg = ReadRegion(is, c.groups.size());
      c.layers.push_back(l);
    }
    else if (key == "zone") {
      Config::Zone z; is >> z.rout;
      z.reg = ReadRegion(is, c.groups.size());
      c.zones.push_back(z);
    }
    else if (key == "hole") is >> c.holeLen;
    else if (key == "tool")
      is >> c.rc >> c.lc >> c.rhi >> c.rho >> c.below >> c.above
         >> c.matCrystal >> c.matHousing >> c.matInner;
    else if (key == "stations") { double z; while (is >> z) c.stations.push_back(z); }
    else if (key == "output") is >> c.out >> c.spectra;
    else Fail("unknown keyword " + key);
  }
  if (c.layers.empty() || c.groups.empty() || c.stations.empty())
    Fail("input needs at least one layer, group and station");
  return c;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const Config& c) : fCfg(c) {}
  void SetStation(double zc) { fZc = zc; }
  G4Material* Crystal() const { return fCfg.mats.at(fCfg.matCrystal); }

  // Local frame: origin at the crystal centre, z up (z = zc - depth).
  G4VPhysicalVolume* Construct() override {
    const Config& c = fCfg;
    const double Rw = c.smax + c.margin, Hw = Rw;
    auto* vac = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
    auto* worldLV = new G4LogicalVolume(
        new G4Box("World", 1.05 * Rw * m, 1.05 * Rw * m, 1.05 * Hw * m), vac, "World");
    auto* world = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);

    std::vector<double> radii = {0.0, c.rc, c.rhi, c.rho};
    for (const auto& z : c.zones) radii.push_back(z.rout);
    radii.push_back(Rw);
    radii = Clean(radii, 0.0, Rw);

    const double d0 = fZc - Hw, d1 = fZc + Hw;
    std::vector<double> zs = {d0, d1, 0.0, c.holeLen, fZc - 0.5 * c.lc, fZc + 0.5 * c.lc,
                              fZc - c.above, fZc + c.below};
    for (const auto& l : c.layers) { zs.push_back(l.top); zs.push_back(l.bot); }
    zs = Clean(zs, d0, d1);

    int n = 0;
    for (size_t i = 0; i + 1 < zs.size(); ++i) {
      const double dm = 0.5 * (zs[i] + zs[i + 1]), hz = 0.5 * (zs[i + 1] - zs[i]);
      for (size_t j = 0; j + 1 < radii.size(); ++j) {
        int mat;
        c.RegionAt(0.5 * (radii[j] + radii[j + 1]), dm, fZc, mat);
        if (mat < 0) continue;
        const std::string name = "cell" + std::to_string(n++);
        auto* s = new G4Tubs(name, radii[j] * m, radii[j + 1] * m, hz * m, 0, twopi);
        auto* lv = new G4LogicalVolume(s, c.mats.at(mat), name);
        new G4PVPlacement(nullptr, G4ThreeVector(0, 0, (fZc - dm) * m), lv, name, worldLV,
                          false, 0);
      }
    }
    return world;
  }

 private:
  static std::vector<double> Clean(std::vector<double> v, double lo, double hi) {
    std::vector<double> out;
    std::sort(v.begin(), v.end());
    for (double x : v)
      if (x >= lo && x <= hi && (out.empty() || x - out.back() > 1e-7)) out.push_back(x);
    return out;
  }
  const Config& fCfg;
  double fZc = 0.0;
};

class PhysicsList : public G4VModularPhysicsList {
 public:
  PhysicsList() {
    SetVerboseLevel(0);
    RegisterPhysics(new G4EmStandardPhysics(0));
    SetDefaultCutValue(1.0 * mm);
  }
};

// ---------------------------------------------------------------------------
// Source: importance-sampled emission points, isotropic gamma lines
// ---------------------------------------------------------------------------
class Source : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Source(const Config& c) : fCfg(c), fGun(1) {
    fGun.SetParticleDefinition(G4Gamma::Definition());
  }
  void Start(int group, double zc) {
    fGroup = group; fZc = zc; fTrials = 0;
    const auto& l = fCfg.lines[group];
    fCum.assign(l.size(), 0.0);
    fYield = 0;
    for (size_t i = 0; i < l.size(); ++i) fCum[i] = (fYield += l[i].second);
    for (auto& x : fCum) x /= fYield;
  }
  long Trials() const { return fTrials; }
  double Weight() const { return fWeight; }
  double Importance() const { return fI0; }   // importance at the emission point

  // Position: the distance s from the crystal centre has pdf
  // p(s) = s exp(-s/L) / (L^2 (1 - (1 + smax/L) exp(-smax/L))) on [0, smax]
  // (a truncated gamma distribution, close to the variance-optimal choice for
  // a formation around a small crystal) and the direction from the centre
  // is isotropic, so the density per volume is q = p(s) / (4 pi s^2).
  // Points with no activity are counted as trials but not tracked.
  // Weight = activity * photons per decay / q  [photons / s].
  void GeneratePrimaries(G4Event* ev) override {
    const double L = fCfg.lambda, a0 = fCfg.smax / L;
    const double norm = L * L * (1.0 - (1.0 + a0) * std::exp(-a0));
    G4ThreeVector x;
    for (;;) {
      ++fTrials;
      double sd;
      do sd = -L * std::log(G4UniformRand() * G4UniformRand()); while (sd > fCfg.smax);
      x = sd * Isotropic();
      int mat;
      const Region* reg = fCfg.RegionAt(x.perp(), fZc - x.z(), fZc, mat);
      const double a = reg ? reg->act[fGroup] : 0.0;
      if (a <= 0) continue;
      const double p = sd * std::exp(-sd / L) / norm;
      fWeight = a * fYield * 2.0 * twopi * sd * sd / p;
      fI0 = fCfg.Importance(sd);
      break;
    }

    const size_t k = std::lower_bound(fCum.begin(), fCum.end(), G4UniformRand()) - fCum.begin();
    fGun.SetParticleEnergy(fCfg.lines[fGroup][std::min(k, fCum.size() - 1)].first * MeV);
    fGun.SetParticlePosition(x * m);
    fGun.SetParticleMomentumDirection(Isotropic());
    fGun.GeneratePrimaryVertex(ev);
  }

 private:
  static G4ThreeVector Isotropic() {
    const double cost = 2.0 * G4UniformRand() - 1.0, sint = std::sqrt(1 - cost * cost);
    const double psi = twopi * G4UniformRand();
    return {sint * std::cos(psi), sint * std::sin(psi), cost};
  }
  const Config& fCfg;
  G4ParticleGun fGun;
  int fGroup = 0;
  double fZc = 0, fYield = 1, fWeight = 0, fI0 = 1;
  long fTrials = 0;
  std::vector<double> fCum;
};

// ---------------------------------------------------------------------------
// Tally: energy deposited in the crystal per branch, smeared and histogrammed
// ---------------------------------------------------------------------------
// A branch is the primary photon or one split copy, with everything it sets in
// motion inside the crystal.  Its weight is fixed once it deposits energy.
struct Branch : public G4VUserTrackInformation {
  Branch(int i, bool c) : id(i), copy(c) {}
  int id;
  bool copy;   // created by splitting (not by a physics process)
};

// Sums of pulse weights (counts/s once divided by the number of trials);
// gross2 sums the squared per-history totals for the statistical error.
struct Tally {
  const Config* cfg = nullptr;
  const Source* src = nullptr;
  G4Material* crystal = nullptr;
  std::map<int, std::pair<double, double>> branches;   // id -> (edep, weight)
  int nextBranch = 1;
  std::vector<double> spec, win;
  double gross = 0, gross2 = 0;
  long hits = 0;               // pulses above the gross threshold
  void Reset() {
    spec.assign(cfg->nbins, 0.0);
    win.assign(cfg->windows.size(), 0.0);
    gross = gross2 = 0;
    hits = 0;
  }
};

static int BranchOf(const G4Track* t) {
  const auto* b = static_cast<const Branch*>(t->GetUserInformation());
  return b ? b->id : 0;
}

class Stepping : public G4UserSteppingAction {
 public:
  explicit Stepping(Tally& t) : fT(t) {}
  void UserSteppingAction(const G4Step* s) override {
    G4Track* trk = s->GetTrack();
    const int br = BranchOf(trk);
    if (s->GetPreStepPoint()->GetMaterial() == fT.crystal) {
      auto& b = fT.branches[br];
      if (b.first == 0) b.second = trk->GetWeight();
      b.first += s->GetTotalEnergyDeposit();
      return;
    }
    // weight window for photons outside the crystal, until their branch scores
    if (trk->GetDefinition() != G4Gamma::Definition() || trk->GetTrackStatus() != fAlive) return;
    if (auto it = fT.branches.find(br); it != fT.branches.end() && it->second.first > 0) return;
    // the track takes its weight from the post-step point at the next step,
    // so both are set
    G4StepPoint* post = s->GetPostStepPoint();
    auto setWeight = [&](double w) { trk->SetWeight(w); post->SetWeight(w); };
    const double target = fT.src->Importance() / fT.cfg->Importance(post->GetPosition().mag() / m);
    const double w = trk->GetWeight();
    if (w > 2.0 * target) {
      const int n = std::min(static_cast<int>(w / target), kMaxSplit);
      setWeight(w / n);
      for (int i = 1; i < n; ++i) {
        auto* c = new G4Track(new G4DynamicParticle(G4Gamma::Definition(), post->GetMomentumDirection(),
                                                    post->GetKineticEnergy()),
                              post->GetGlobalTime(), post->GetPosition());
        c->SetWeight(w / n);
        c->SetPolarization(post->GetPolarization());
        c->SetParentID(trk->GetTrackID());
        c->SetTouchableHandle(trk->GetNextTouchableHandle());
        c->SetUserInformation(new Branch(fT.nextBranch++, true));
        fpSteppingManager->GetfSecondary()->push_back(c);
      }
    } else if (w < 0.5 * target) {
      if (G4UniformRand() * target < w) setWeight(target);
      else trk->SetTrackStatus(fStopAndKill);
    }
  }
 private:
  static constexpr int kMaxSplit = 10;
  Tally& fT;
};

// Secondaries belong to the branch of their parent.
class Tracking : public G4UserTrackingAction {
 public:
  void PostUserTrackingAction(const G4Track* t) override {
    const int br = BranchOf(t);
    for (G4Track* sec : *fpTrackingManager->GimmeSecondaries())
      if (!sec->GetUserInformation()) sec->SetUserInformation(new Branch(br, false));
  }
};

class EventAction : public G4UserEventAction {
 public:
  explicit EventAction(Tally& t) : fT(t) {}
  void BeginOfEventAction(const G4Event*) override {
    fT.branches.clear();
    fT.nextBranch = 1;
  }
  void EndOfEventAction(const G4Event*) override {
    double gross = 0;
    for (const auto& [id, b] : fT.branches) {
      if (b.first <= 0) continue;
      const double e0 = b.first / MeV;
      const double sigma = fT.cfg->fwhm662 / 2.3548 * std::sqrt(0.662 * e0);
      const double e = G4RandGauss::shoot(e0, sigma), w = fT.src->Weight() * b.second;
      if (e >= fT.cfg->threshold) { gross += w; ++fT.hits; }
      for (size_t i = 0; i < fT.cfg->windows.size(); ++i)
        if (e >= fT.cfg->windows[i].first && e < fT.cfg->windows[i].second) fT.win[i] += w;
      const int k = static_cast<int>(e / fT.cfg->emax * fT.cfg->nbins);
      if (k >= 0 && k < fT.cfg->nbins) fT.spec[k] += w;
    }
    fT.gross += gross;
    fT.gross2 += gross * gross;
  }
 private:
  Tally& fT;
};

// Electrons, positrons and photons set in motion outside the crystal by a
// physics process are killed: electrons deposit their energy locally, and the
// positrons, fluorescence and bremsstrahlung photons they would produce add
// well under 1 % to the count rate but would break the one-photon-per-branch
// bookkeeping.  Split copies are kept.
class Stacking : public G4UserStackingAction {
 public:
  explicit Stacking(Tally& t) : fT(t) {}
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    if (t->GetParentID() == 0) return fUrgent;
    const auto* b = static_cast<const Branch*>(t->GetUserInformation());
    const bool inCrystal = t->GetVolume() &&
                           t->GetVolume()->GetLogicalVolume()->GetMaterial() == fT.crystal;
    return (inCrystal || (b && b->copy)) ? fUrgent : fKill;
  }
 private:
  Tally& fT;
};

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 2) Fail("usage: gammalog <input file>");
  Config cfg = ReadConfig(argv[1]);
  G4Random::setTheSeed(cfg.seed);

  auto* rm = new G4RunManager;
  rm->SetVerboseLevel(0);
  auto* det = new Detector(cfg);
  det->SetStation(cfg.stations.front());
  rm->SetUserInitialization(det);
  rm->SetUserInitialization(new PhysicsList);
  G4EmParameters::Instance()->SetVerbose(0);

  Tally tally;
  auto* src = new Source(cfg);
  tally.cfg = &cfg;
  tally.src = src;
  tally.crystal = det->Crystal();
  rm->SetUserAction(src);
  rm->SetUserAction(new EventAction(tally));
  rm->SetUserAction(new Stepping(tally));
  rm->SetUserAction(new Stacking(tally));
  rm->SetUserAction(new Tracking);
  G4UImanager::GetUIpointer()->ApplyCommand("/control/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand("/process/em/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand("/process/had/verbose 0");
  rm->Initialize();

  // All rates in counts/s; cps_err is the 1-sigma Monte Carlo error of cps.
  std::ofstream out(cfg.out), spec(cfg.spectra);
  out << "depth,group,events,trials,hits,cps,cps_err";
  for (const auto& w : cfg.winNames) out << "," << w;
  out << "\n";
  spec << "depth,group";
  for (int b = 0; b < cfg.nbins; ++b) spec << ",b" << b;
  spec << "\n";
  out.precision(10);
  spec.precision(8);

  for (size_t s = 0; s < cfg.stations.size(); ++s) {
    const double zc = cfg.stations[s];
    if (s > 0) {
      det->SetStation(zc);
      rm->ReinitializeGeometry(true);
    }
    for (size_t g = 0; g < cfg.groups.size(); ++g) {
      // Anything of this group radioactive within the source sphere around the
      // crystal? If not, the count rate is zero; the source could never accept
      // an emission point and would loop forever (e.g. a thick unit with
      // K = U = Th = 0).
      const double d0 = zc - cfg.smax, d1 = zc + cfg.smax;
      bool active = false;
      for (size_t i = 0; i < cfg.layers.size(); ++i) {
        const auto& l = cfg.layers[i];
        const double bot = (i + 1 == cfg.layers.size()) ? 1e30 : l.bot;
        if (bot > std::max(d0, 0.0) && l.top < d1) active |= l.reg.act[g] > 0;
      }
      if (cfg.holeLen > std::max(d0, 0.0))
        for (const auto& z : cfg.zones) active |= z.reg.act[g] > 0;
      tally.Reset();
      long trials = 1;
      if (active) {
        src->Start(static_cast<int>(g), zc);
        rm->BeamOn(static_cast<G4int>(cfg.events));
        trials = src->Trials();
      }
      const double n = static_cast<double>(trials);
      // Var(mean) = (sum w^2 / n - (sum w / n)^2) / n
      const double var = std::max(0.0, tally.gross2 / n - std::pow(tally.gross / n, 2)) / n;
      out << zc << "," << cfg.groups[g] << "," << (active ? cfg.events : 0) << "," << trials
          << "," << tally.hits << "," << tally.gross / n << "," << std::sqrt(var);
      for (double w : tally.win) out << "," << w / n;
      out << "\n";
      spec << zc << "," << cfg.groups[g];
      for (double b : tally.spec) spec << "," << b / n;
      spec << "\n";
    }
    out.flush();
    spec.flush();
    std::cout << "station " << s + 1 << "/" << cfg.stations.size() << " depth " << zc << std::endl;
  }
  delete rm;
  return 0;
}
