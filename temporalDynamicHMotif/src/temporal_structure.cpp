#include "../include/temporal_structure.hpp"
#include "../include/structure.hpp"

TemporalCBSTOperations::TemporalCBSTOperations(const std::string& label,
                                               int payloadCapacity,
                                               int alignment)
  : label_(label), payloadCapacity_(payloadCapacity), alignment_(alignment) {
  slotLabels_[0] = label_ + "-Older";
  slotLabels_[1] = label_ + "-Middle";
  slotLabels_[2] = label_ + "-Newest";
  slots_[0] = std::make_unique<CBSTOperations>(slotLabels_[0].c_str(), payloadCapacity_, alignment_);
  slots_[1] = std::make_unique<CBSTOperations>(slotLabels_[1].c_str(), payloadCapacity_, alignment_);
  slots_[2] = std::make_unique<CBSTOperations>(slotLabels_[2].c_str(), payloadCapacity_, alignment_);
}

TemporalCBSTOperations::~TemporalCBSTOperations() = default;

void TemporalCBSTOperations::rotate() {
  oldestIdx_ = (oldestIdx_ + 1) % 3;
}

void TemporalCBSTOperations::resetLayer(TemporalLayer layer) {
  CBSTContext& ctx = mutableContext(layer);
  ctx.initialPayloadSize = 0;
}

void TemporalCBSTOperations::construct(TemporalLayer layer,
                                       int* cbstKeys,
                                       int* cbstStartOffsets,
                                       int numRecords,
                                       int* flatPayload,
                                       int flatPayloadSize) {
  slot(layer).construct(cbstKeys, cbstStartOffsets, numRecords, flatPayload, flatPayloadSize);
}

void TemporalCBSTOperations::insert(TemporalLayer layer,
                                    const std::vector<int>& keys,
                                    const std::vector<int>& payload,
                                    const std::vector<int>& prefix) {
  slot(layer).insert(keys, payload, prefix);
}

void TemporalCBSTOperations::erase(TemporalLayer layer, const std::vector<int>& keys) {
  slot(layer).erase(keys);
}

void TemporalCBSTOperations::fill(TemporalLayer layer,
                                  const std::vector<int>& keys,
                                  const std::vector<int>& payload,
                                  const std::vector<int>& prefix) {
  fillCBST(keys, payload, prefix, mutableContext(layer));
}

void TemporalCBSTOperations::unfill(TemporalLayer layer,
                                    const std::vector<int>& keys,
                                    const std::vector<int>& valuesToRemove,
                                    const std::vector<int>& removePrefix) {
  unfillCBST(keys, valuesToRemove, removePrefix, mutableContext(layer));
}

const CBSTContext& TemporalCBSTOperations::context(TemporalLayer layer) const {
  return slot(layer).context();
}

CBSTContext& TemporalCBSTOperations::context(TemporalLayer layer) {
  return mutableContext(layer);
}

TemporalLayer TemporalCBSTOperations::oldest() const { return static_cast<TemporalLayer>(oldestIdx_); }
TemporalLayer TemporalCBSTOperations::middle() const { return static_cast<TemporalLayer>((oldestIdx_ + 1) % 3); }
TemporalLayer TemporalCBSTOperations::newest() const { return static_cast<TemporalLayer>((oldestIdx_ + 2) % 3); }

int TemporalCBSTOperations::resolveIndex(TemporalLayer layer) const {
  switch (layer) {
    case TemporalLayer::Older:  return oldestIdx_;
    case TemporalLayer::Middle: return (oldestIdx_ + 1) % 3;
    case TemporalLayer::Newest: return (oldestIdx_ + 2) % 3;
  }
  return oldestIdx_;
}

const CBSTOperations& TemporalCBSTOperations::slot(TemporalLayer layer) const {
  return *slots_[resolveIndex(layer)];
}

CBSTOperations& TemporalCBSTOperations::slot(TemporalLayer layer) {
  return *slots_[resolveIndex(layer)];
}

CBSTContext& TemporalCBSTOperations::mutableContext(TemporalLayer layer) {
  const CBSTContext& c = slot(layer).context();
  return const_cast<CBSTContext&>(c);
}


