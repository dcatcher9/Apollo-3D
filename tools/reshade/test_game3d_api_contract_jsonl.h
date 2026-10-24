// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// One JSON line per API contract matrix case (README, API contract matrix),
// the shape Invoke-Game3DContractMatrix.ps1 collects from the hook layer
// (test_game3d_api_contract_cpu.cpp) and the runtime layer
// (test_game3d_api_contract_runtime.cpp): {layer, family, id, seed,
// dimensions, legal, rule, expected, observed, verdict, rows, evidence}. A
// layer adds what it saw to observed and evidence.
#include "test_contract_matrix_generator.h"
#include "test_game3d_api_contract_tally.h"

#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

namespace game3d_contract::jsonl {
  inline const char *name(refusal value) {
    constexpr const char *names[] {"none", "state_unprovable", "no_copy_point", "lifetime", "not_depth", "malformed", "unknown_abi", "ambiguous_view", "fg_scope"};
    return names[unsigned(value)];
  }

  inline const char *name(known_gaps::verdict value) {
    constexpr const char *names[] {"meets", "violates", "known_failure", "gap_closed", "misattributed", "pending", "overlapping"};
    return names[unsigned(value)];
  }

  template<class Case, unsigned N>
  nlohmann::json line(const char *layer, const generator::family &owner, const generator::row &assignment, const char *seed, bool legal, const outcome &expected, const result &actual, known_gaps::verdict verdict, const known_gaps::row<Case> (&rows)[N], const bool (&matched)[N]) {
    nlohmann::json dimensions = nlohmann::json::object();
    for (std::size_t i = 0; i != owner.dimensions.size(); ++i) {
      dimensions[owner.dimensions[i].name] = owner.dimensions[i].values[assignment[i]];
    }
    nlohmann::json ids = nlohmann::json::array();
    for (unsigned i = 0; i != N; ++i) {
      if (matched[i]) {
        ids.push_back(rows[i].id);
      }
    }
    return {
      {"layer", layer},
      {"family", owner.name},
      {"id", generator::id_text(generator::case_id(owner, assignment))},
      {"seed", seed ? nlohmann::json(seed) : nlohmann::json()},
      {"dimensions", dimensions},
      {"legal", legal},
      {"rule", expected.by ? expected.by->id : ""},
      {"expected", {{"kind", judging::name(expected.kind)}, {"alternative", judging::name(expected.alternative)}, {"why", name(expected.why)}, {"deferred", expected.deferred}, {"basis_rule", expected.basis_by ? expected.basis_by->id : ""}}},
      {"observed", {{"kind", judging::name(actual.kind)}, {"why", name(actual.why)}, {"deferred", actual.deferred}}},
      {"verdict", name(verdict)},
      {"rows", ids},
      {"evidence", nlohmann::json::object()},
    };
  }

  // The file SUNSHINE_CONTRACT_MATRIX_JSONL names, appended to; closed when
  // the variable is unset or empty.
  inline std::ofstream from_environment() {
    std::ofstream out;
    if (const char *path = std::getenv("SUNSHINE_CONTRACT_MATRIX_JSONL"); path && *path) {
      out.open(path, std::ios::binary | std::ios::app);
    }
    return out;
  }
}  // namespace game3d_contract::jsonl
