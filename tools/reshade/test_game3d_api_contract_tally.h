// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// How the API contract matrix's layers (test_game3d_api_contract_cpu.cpp and
// test_game3d_api_contract_device.cpp) count a family's verdicts against its
// known-gap rows (test_game3d_api_contract_known_gaps.h) and describe a cell
// that fails. A family that fails outside its rows, or keeps a row no failing
// cell needs, throws.
#include "test_game3d_api_contract_known_gaps.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace game3d_contract::judging {
  inline const char *name(outcome_kind kind) {
    constexpr const char *names[] {"capture", "refuse", "hold", "continuous", "offered", "ignored"};
    return names[unsigned(kind)];
  }

  inline const char *name(known_gaps::verdict value) {
    constexpr const char *names[] {"meets the oracle", "violates the contract", "recorded gap", "recorded gap now passes", "misattributed", "pending", "overlapping rows"};
    return names[unsigned(value)];
  }

  // One family's verdicts against its known-gap rows (judge). Up to eight
  // failing cells per family are described.
  template<unsigned N>
  struct tally {
    unsigned cells {}, met {}, unmet {}, closed {}, pending {}, known[N] {};

    // The reason to describe a failing cell, or null.
    const char *add(known_gaps::verdict verdict, int index) {
      ++cells;
      switch (verdict) {
        case known_gaps::verdict::meets:
          ++met;
          return nullptr;
        case known_gaps::verdict::known_failure:
          ++known[index];
          return nullptr;
        case known_gaps::verdict::pending:
          ++pending;
          return nullptr;
        case known_gaps::verdict::gap_closed:
          return closed++ < 8 ? "known gap now passes; remove its row" : nullptr;
        case known_gaps::verdict::misattributed:
          return unmet++ < 8 ? "known-gap row matched a cell another rule decides" : nullptr;
        case known_gaps::verdict::overlapping:
          return unmet++ < 8 ? "more than one known-gap row matches" : nullptr;
        default:
          return unmet++ < 8 ? "contract violated" : nullptr;
      }
    }

    template<class Case>
    void finish(const known_gaps::row<Case> (&rows)[N], const char *family) const {
      if (unmet || closed) {
        std::fprintf(stderr, "%s: %u cells violate the contract outside the known-gap table; %u recorded gaps now pass\n", family, unmet, closed);
        throw std::runtime_error("a production table diverged from the API contract matrix");
      }
      for (unsigned i = 0; i != N; ++i) {
        if (rows[i].status == known_gaps::kind::expected_failure && !known[i]) {
          std::fprintf(stderr, "%s: known-gap row %s matches no failing cell\n", family, rows[i].id);
          throw std::runtime_error("a stale known-gap row");
        }
      }
    }

    template<class Case>
    void print(const known_gaps::row<Case> (&rows)[N]) const {
      std::printf("%u meet the oracle, %u are recorded gaps (", met, cells - met - pending);
      for (unsigned i = 0; i != N; ++i) {
        std::printf("%s%s %u", i ? ", " : "", rows[i].id, known[i]);
      }
      std::printf(")");
      if (pending) {
        std::printf(", %u are pending", pending);
      }
    }
  };

  // A family's verdicts against rows judged with judge_each.
  template<unsigned N>
  struct each_tally {
    unsigned cells {}, met {}, unmet {}, closed {}, known {}, pending {};
    unsigned matched[N] {}, alone[N] {}, failing[N] {};

    const char *add(known_gaps::verdict verdict, const bool (&rows)[N], bool satisfied) {
      ++cells;
      unsigned count {}, last {};
      for (unsigned i = 0; i != N; ++i) {
        if (rows[i]) {
          ++count;
          last = i;
          ++matched[i];
          failing[i] += !satisfied;
        }
      }
      switch (verdict) {
        case known_gaps::verdict::meets:
          ++met;
          return nullptr;
        case known_gaps::verdict::pending:
          ++pending;
          return nullptr;
        case known_gaps::verdict::known_failure:
          ++known;
          alone[last] += count == 1;
          return nullptr;
        case known_gaps::verdict::gap_closed:
          return closed++ < 8 ? "known gap now passes; remove or narrow its row" : nullptr;
        case known_gaps::verdict::misattributed:
          return unmet++ < 8 ? "known gap fails another way than its row records, or under another rule" : nullptr;
        default:
          return unmet++ < 8 ? "contract violated" : nullptr;
      }
    }

    // A partial run (one case or one shard) cannot show a row stale.
    template<class Case>
    void finish(const known_gaps::row<Case> (&rows)[N], const char *family, bool complete) const {
      if (unmet || closed) {
        std::fprintf(stderr, "%s: %u cells violate the contract outside the known-gap table; %u recorded gaps now pass\n", family, unmet, closed);
        throw std::runtime_error("an adapter diverged from the API contract matrix");
      }
      for (unsigned i = 0; complete && i != N; ++i) {
        if (rows[i].status == known_gaps::kind::expected_failure && !alone[i]) {
          std::fprintf(stderr, "%s: known-gap row %s is the only row of no failing cell\n", family, rows[i].id);
          throw std::runtime_error("a stale known-gap row");
        }
      }
    }

    template<class Case>
    void print(const known_gaps::row<Case> (&rows)[N]) const {
      std::printf("%u meet the oracle, %u are recorded gaps (", met, known);
      const char *separator = "";
      for (unsigned i = 0; i != N; ++i) {
        if (rows[i].status == known_gaps::kind::expected_failure) {
          std::printf("%s%s %u", separator, rows[i].id, matched[i]);
          separator = ", ";
        }
      }
      std::printf("), %u are pending (", pending);
      separator = "";
      for (unsigned i = 0; i != N; ++i) {
        if (rows[i].status == known_gaps::kind::pending) {
          std::printf("%s%s %u, %u failing", separator, rows[i].id, matched[i], failing[i]);
          separator = "; ";
        }
      }
      std::printf(")");
    }
  };

  inline const char *row_names(const bool *matched, const char *const *ids, unsigned count, std::string &out) {
    out.clear();
    for (unsigned i = 0; i != count; ++i) {
      if (matched[i]) {
        out += out.empty() ? "" : " ";
        out += ids[i];
      }
    }
    return out.empty() ? "none" : out.c_str();
  }

  inline void describe_expected(const outcome &expected) {
    std::fprintf(stderr, "  expected: %s%s%s rule=%s basis_rule=%s deferred=%u basis=%s0x%x/0x%x/0x%x\n", name(expected.kind), either(expected) ? " or " : "", either(expected) ? name(expected.alternative) : "", expected.by ? expected.by->id : "?", expected.basis_by ? expected.basis_by->id : "-", expected.deferred, expected.exact_basis ? "" : "(any) ", expected.basis.enhanced ? expected.basis.layout : expected.basis.state, expected.basis.access, unsigned(expected.basis.enhanced));
  }
}  // namespace game3d_contract::judging
