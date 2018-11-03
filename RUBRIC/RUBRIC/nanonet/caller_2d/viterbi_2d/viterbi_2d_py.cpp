#include <iostream>
#include <viterbi_2d_py.h>


using namespace std;
using ublas::matrix;


Viterbi2D_Py::Viterbi2D_Py(bp::dict& stateInfo, bp::dict& params) {
  bandSize = bp::extract<int>(params["band_size"]);
  int kmerLen = bp::extract<int>(params["kmer_len"]);
  setupKmers(kmerLen);
  bool rc = bp::extract<bool>(params["seq2_is_rc"]);
  useNoise = bp::extract<bool>(params["use_sd"]);
  if (stateInfo.has_key(string("kmers"))) {
    bp::list kmers = bp::extract<bp::list>(stateInfo["kmers"]);
    emission1 = dummyEmission(kmers);
    emission2 = dummyEmission(kmers);
  }
  else {
    bp::dict model1 = bp::extract<bp::dict>(stateInfo["model1"]);
    bp::dict model2 = bp::extract<bp::dict>(stateInfo["model2"]);
    emission1 = makeEmission(model1, false);
    emission2 = makeEmission(model2, rc);
  }
  int numStates = emission1->NumStates();
  int maxNodes = bp::extract<int>(params["max_nodes"]);
  int maxLen = bp::extract<int>(params["max_len"]);
  vector<double> trans(6);
  trans[0] = bp::extract<double>(params["stay1"]);
  trans[1] = bp::extract<double>(params["step1"]);
  trans[2] = bp::extract<double>(params["skip1"]);
  trans[3] = bp::extract<double>(params["stay2"]);
  trans[4] = bp::extract<double>(params["step2"]);
  trans[5] = bp::extract<double>(params["skip2"]);
  viterbi = boost::shared_ptr<Viterbi2D>(new Viterbi2D(maxNodes, maxLen, numStates, trans));
}


bp::list Viterbi2D_Py::GetKmerList() const {
  bp::list kmerList;
  for (size_t i = 0; i < kmers.size(); ++i) {
    kmerList.append(kmers[i]);
  }
  return kmerList;
}


bp::numeric::array Viterbi2D_Py::GetModelLevels1() const {
  const vector<double> levelVec = emission1->GetModelLevels();
  bp::numeric::array result = new_numpy_1d<double>(levelVec.size());
  VecView<double> data = view_1d_array<double>(result);
  for (size_t i = 0; i < levelVec.size(); ++i) {
    data[i] = levelVec[i];
  }
  return result;
}


bp::numeric::array Viterbi2D_Py::GetModelLevels2() const {
  const vector<double> levelVec = emission2->GetModelLevels();
  bp::numeric::array result = new_numpy_1d<double>(levelVec.size());
  VecView<double> data = view_1d_array<double>(result);
  for (size_t i = 0; i < levelVec.size(); ++i) {
    data[i] = levelVec[i];
  }
  return result;
}


bp::dict Viterbi2D_Py::Call(bp::dict& events1, bp::dict& events2, bp::list& alignment, bp::object& prior) {
  vector<double> means1, stdvs1, stwts1, emwts1, means2, stdvs2, stwts2, emwts2;
  getEvents(events1, means1, stdvs1, stwts1, emwts1);
  getEvents(events2, means2, stdvs2, stwts2, emwts2);
  emission1->SetEvents(means1, stdvs1, stwts1, emwts1);
  emission2->SetEvents(means2, stdvs2, stwts2, emwts2);
  Alignment alignIn;
  alignIn = list_to_pair_vector<int32_t>(alignment);
  vector<int32_t> bandStarts, bandEnds;
  makeBands(alignIn, bandStarts, bandEnds);
  vector<int32_t> priorScores(emission1->NumStates());
  if (prior) {
    fill(priorScores.begin(), priorScores.end(), ZERO_PROB_SCORE);
    int state = states[bp::extract<string>(prior)];
    priorScores[state] = 0;
  }
  Alignment alignOut;
  vector<int16_t> statesOut;
  viterbi->Call(*emission1, *emission2, bandStarts, bandEnds, priorScores, alignOut, statesOut);
  return makeResult(alignOut, statesOut);
}


bp::dict Viterbi2D_Py::CallPost(bp::numeric::array& post1, bp::numeric::array& post2,
                                bp::numeric::array& stayWt1, bp::numeric::array& stayWt2,
                                bp::list& alignment, bp::object& prior) {
  MatView<float> probs1 = view_2d_array<float>(post1);
  MatView<float> probs2 = view_2d_array<float>(post2);
  VecView<double> stayWeight1 = view_1d_array<double>(stayWt1);
  VecView<double> stayWeight2 = view_1d_array<double>(stayWt2);
  int numStates = int(probs1.size2());
  emission1->SetNEvents(int(probs1.size1()));
  emission2->SetNEvents(int(probs2.size1()));
  Alignment alignIn;
  alignIn = list_to_pair_vector<int32_t>(alignment);
  vector<int32_t> bandStarts, bandEnds;
  makeBands(alignIn, bandStarts, bandEnds);
  vector<int32_t> priorScores(numStates);
  if (prior) {
    fill(priorScores.begin(), priorScores.end(), ZERO_PROB_SCORE);
    int state = states[bp::extract<string>(prior)];
    priorScores[state] = 0;
  }
  Alignment alignOut;
  vector<int16_t> statesOut;
  viterbi->Call(probs1, probs2, stayWeight1, stayWeight2, bandStarts, bandEnds, priorScores, alignOut, statesOut);
  return makeResult(alignOut, statesOut);
}


void Viterbi2D_Py::setupKmers(int kmerLen) {
  int numKmers = 1 << (kmerLen << 1);
  const char letters[] = "ACGT";
  kmers.resize(numKmers);
  states.clear();
  vector<int> pos(kmerLen);
  for (int i = 0; i < numKmers; ++i) {
    string kmer;
    for (int j = 0; j < kmerLen; ++j) {
      kmer += letters[pos[kmerLen - j - 1]];
    }
    kmers[i] = kmer;
    states[kmer] = i;
    bool flag = true;
    int digit = 0;
    while (flag) {
      ++pos[digit];
      if (pos[digit] == 4) {
        pos[digit] = 0;
        ++digit;
        if (digit == kmerLen) {
          flag = false;
        }
      }
      else {
        flag = false;
      }
    }
  }
}


boost::shared_ptr<Emission> Viterbi2D_Py::makeEmission(bp::dict& model, bool rc) {
  bp::numeric::array levelMean = bp::extract<bp::numeric::array>(model.get("level_mean"));
  bp::numeric::array levelStdv = bp::extract<bp::numeric::array>(model.get("level_stdv"));
  bp::numeric::array sdMean = bp::extract<bp::numeric::array>(model.get("sd_mean"));
  bp::numeric::array sdStdv = bp::extract<bp::numeric::array>(model.get("sd_stdv"));
  bp::list kmer = bp::extract<bp::list>(model.get("kmer"));
  VecView<double> mean = view_1d_array<double>(levelMean);
  VecView<double> sigma = view_1d_array<double>(levelStdv);
  VecView<double> noise = view_1d_array<double>(sdMean);
  VecView<double> noiseSd = view_1d_array<double>(sdStdv);
  int numStates = int(mean.size());
  vector<double> levels(numStates), levelSpreads(numStates), noises(numStates), noiseSpreads(numStates);
  copy(mean.begin(), mean.end(), levels.begin());
  copy(sigma.begin(), sigma.end(), levelSpreads.begin());
  copy(noise.begin(), noise.end(), noises.begin());
  copy(noiseSd.begin(), noiseSd.end(), noiseSpreads.begin());
  vector<string> mdlKmers = list_to_vector<string>(kmer);
  sortModel(levels, levelSpreads, noises, noiseSpreads, mdlKmers, rc);
  return boost::shared_ptr<Emission>(new Emission(levels, levelSpreads, noises, noiseSpreads, useNoise));
}


boost::shared_ptr<Emission> Viterbi2D_Py::dummyEmission(bp::list& kmers) {
  int numStates = bp::len(kmers);
  vector<double> levels(numStates), levelSpreads(numStates), noises(numStates), noiseSpreads(numStates);
  return boost::shared_ptr<Emission>(new Emission(levels, levelSpreads, noises, noiseSpreads, useNoise));
}


void Viterbi2D_Py::sortModel(vector<double>& levels, vector<double>& levelSpreads, vector<double>& noises,
                             vector<double>& noiseSpreads, vector<string>& mdlKmers, bool rc) {
  int numKmers = int(levels.size());
  vector<double> newLvl(numKmers), newLvlSprd(numKmers), newSd(numKmers), newSdSprd(numKmers);
  vector<string> newKmer(numKmers);
  map<char, char> rcMap;
  rcMap['A'] = 'T';
  rcMap['C'] = 'G';
  rcMap['G'] = 'C';
  rcMap['T'] = 'A';
  for (int i = 0; i < numKmers; ++i) {
    string kmer = mdlKmers[i];
    if (rc) {
      reverse(kmer.begin(), kmer.end());
      for (string::iterator p = kmer.begin(); p < kmer.end(); ++p) {
        *p = rcMap[*p];
      }
    }
    int pos = states[kmer];
    newLvl[pos] = levels[i];
    newLvlSprd[pos] = levelSpreads[i];
    newSd[pos] = noises[i];
    newSdSprd[pos] = noiseSpreads[i];
    newKmer[pos] = mdlKmers[i];
  }
  levels.swap(newLvl);
  levelSpreads.swap(newLvlSprd);
  noises.swap(newSd);
  noiseSpreads.swap(newSdSprd);
  mdlKmers.swap(newKmer);
}


void Viterbi2D_Py::getEvents(bp::dict& events, vector<double>& means, vector<double>& stdvs,
                             vector<double>& stayWts, vector<double>& emWts) {
  bp::numeric::array mean = bp::extract<bp::numeric::array>(events.get("mean"));
  bp::numeric::array stdv = bp::extract<bp::numeric::array>(events.get("stdv"));
  bp::numeric::array stayWeight = bp::extract<bp::numeric::array>(events.get("stay_weight"));
  bp::numeric::array emWeight = bp::extract<bp::numeric::array>(events.get("em_weight"));
  VecView<double> meanV = view_1d_array<double>(mean);
  VecView<double> stdvV = view_1d_array<double>(stdv);
  VecView<double> stwtV = view_1d_array<double>(stayWeight);
  VecView<double> emwtV = view_1d_array<double>(emWeight);
  int numEvents = int(meanV.size());
  means.resize(numEvents);
  stdvs.resize(numEvents);
  stayWts.resize(numEvents);
  emWts.resize(numEvents);
  copy(meanV.begin(), meanV.end(), means.begin());
  copy(stdvV.begin(), stdvV.end(), stdvs.begin());
  copy(stwtV.begin(), stwtV.end(), stayWts.begin());
  copy(emwtV.begin(), emwtV.end(), emWts.begin());

}


void Viterbi2D_Py::makeBands(const Alignment& alignIn, vector<int32_t>& bandStarts, vector<int32_t>& bandEnds) {
  int numEvents1 = emission1->NumEvents();
  int numEvents2 = emission2->NumEvents();
  bandStarts.resize(numEvents2);
  bandEnds.resize(numEvents2);
  fill(bandStarts.begin(), bandStarts.end(), int32_t(numEvents1 - 1));
  fill(bandEnds.begin(), bandEnds.end(), int32_t(0));
  int lastX = 0, lastY = numEvents2 - 1;
  int nPos = int(alignIn.size());
  for (int p = 0; p < nPos; ++p) {
    int x = (alignIn[p].first == -1) ? lastX : alignIn[p].first;
    int y = (alignIn[p].second == -1) ? lastY : alignIn[p].second;
    for (int k = y - bandSize; k <= y + bandSize; ++k) {
      if (k < 0 || k >= numEvents2) continue;
      int left = min((int32_t)(x - bandSize), bandStarts[k]);
      int right = max((int32_t)(x + bandSize), bandEnds[k]);
      left = max(0, left);
      right = min(numEvents1 - 1, right);
      bandStarts[k] = left;
      bandEnds[k] = right;
    }
    lastX = x;
    lastY = y;
  }
}


bp::dict Viterbi2D_Py::makeResult(const Alignment& alignOut, const vector<int16_t>& statesOut) {
  bp::list align;
  bp::list kmersOut;
  int count = int(alignOut.size());
  for (int i = 0; i < count; ++i) {
    kmersOut.append(kmers[statesOut[i]]);
    bp::tuple data = bp::make_tuple(alignOut[i].first, alignOut[i].second);
    align.append(data);
  }
  bp::dict results;
  results["alignment"] = align;
  results["kmers"] = kmersOut;
  return results;
}
