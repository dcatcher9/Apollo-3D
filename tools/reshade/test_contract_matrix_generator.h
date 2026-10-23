// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The Game 3D API contract matrix's case generator (README, API contract
// matrix). A family names its dimensions, each value list starting with the
// dimension's default, and a legality predicate. From those it builds:
// - product(): every assignment of some dimensions, the rest at a base row;
// - one_factor(): each other value of each dimension alone on a base row,
//   so every value is judged on its own at least once;
// - pairwise(): a covering array in which every legal pair of values of two
//   dimensions appears in at least one row, built by IPOG (Lei et al., "IPOG:
//   A General Strategy for T-Way Software Testing", ECBS 2007) on top of seed
//   rows, honouring the predicate.
// Everything is deterministic: the same family always yields the same rows.
//
// The predicate sees partial rows, where an unassigned dimension holds -1. It
// refuses a row only when the values already assigned make it illegal.
//
// A case id hashes the family and its non-default assignments
// (name=value), never a row index: adding a value, or a dimension whose
// default comes first, renames no existing case.
//
// Like the oracle, this header includes nothing from the add-on
// (reshade_contract_oracle_independence).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace game3d_contract::generator {
  struct dimension {
    const char *name;
    std::vector<const char *> values;  // values[0] is the default.
  };

  // One value index per dimension; -1 is unassigned.
  using row = std::vector<int>;
  using predicate = std::function<bool(const row &)>;

  struct family {
    const char *name;
    std::vector<dimension> dimensions;
    predicate legal;
  };

  inline row defaults(const family &value) {
    return row(value.dimensions.size(), 0);
  }

  inline bool complete(const row &value) {
    return std::none_of(value.begin(), value.end(), [](int index) {
      return index < 0;
    });
  }

  inline int dimension_index(const family &value, const std::string &name) {
    for (std::size_t i = 0; i != value.dimensions.size(); ++i) {
      if (name == value.dimensions[i].name) {
        return int(i);
      }
    }
    return -1;
  }

  inline int value_index(const dimension &value, const std::string &name) {
    for (std::size_t i = 0; i != value.values.size(); ++i) {
      if (name == value.values[i]) {
        return int(i);
      }
    }
    return -1;
  }

  // name=value for every assigned non-default dimension, comma-separated;
  // "defaults" when there is none.
  inline std::string describe(const family &owner, const row &value) {
    std::string out;
    for (std::size_t i = 0; i != value.size(); ++i) {
      if (value[i] <= 0) {
        continue;
      }
      out += out.empty() ? "" : ",";
      out += owner.dimensions[i].name;
      out += '=';
      out += owner.dimensions[i].values[value[i]];
    }
    return out.empty() ? "defaults" : out;
  }

  // 64-bit FNV-1a over "family:" and the non-default assignments.
  inline std::uint64_t case_id(const family &owner, const row &value) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](const char *text) {
      for (; *text; ++text) {
        hash = (hash ^ std::uint8_t(*text)) * 0x100000001b3ull;
      }
    };
    mix(owner.name);
    mix(":");
    for (std::size_t i = 0; i != value.size(); ++i) {
      if (value[i] > 0) {
        mix(owner.dimensions[i].name);
        mix("=");
        mix(owner.dimensions[i].values[value[i]]);
        mix(";");
      }
    }
    return hash;
  }

  inline std::string id_text(std::uint64_t id) {
    char text[17] {};
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(id));
    return text;
  }

  // "name=value,name=value" over the defaults; false names the first unknown
  // dimension or value in error.
  inline bool parse(const family &owner, const std::string &text, row &out, std::string &error) {
    out = defaults(owner);
    std::size_t start = 0;
    while (start < text.size()) {
      auto end = text.find(',', start);
      if (end == std::string::npos) {
        end = text.size();
      }
      const auto item = text.substr(start, end - start);
      const auto equals = item.find('=');
      const auto dimension = equals == std::string::npos ? -1 : dimension_index(owner, item.substr(0, equals));
      const auto index = dimension < 0 ? -1 : value_index(owner.dimensions[dimension], item.substr(equals + 1));
      if (index < 0) {
        error = "unknown " + std::string(dimension < 0 ? "dimension" : "value") + " in '" + item + "'";
        return false;
      }
      out[dimension] = index;
      start = end + 1;
    }
    return true;
  }

  // Every assignment of the listed dimensions (the last varies fastest), the
  // other dimensions as in base. Illegal rows are kept: the caller judges them.
  inline std::vector<row> product(const family &owner, const std::vector<unsigned> &varied, const row &base) {
    std::vector<row> out {base};
    for (const auto dimension : varied) {
      std::vector<row> next;
      for (const auto &value : out) {
        for (std::size_t i = 0; i != owner.dimensions[dimension].values.size(); ++i) {
          auto changed = value;
          changed[dimension] = int(i);
          next.push_back(changed);
        }
      }
      out.swap(next);
    }
    return out;
  }

  // base with one dimension at each of its other values, legal rows only.
  inline std::vector<row> one_factor(const family &owner, const row &base) {
    std::vector<row> out;
    for (std::size_t dimension = 0; dimension != owner.dimensions.size(); ++dimension) {
      for (std::size_t i = 0; i != owner.dimensions[dimension].values.size(); ++i) {
        if (int(i) == base[dimension]) {
          continue;
        }
        auto value = base;
        value[dimension] = int(i);
        if (owner.legal(value)) {
          out.push_back(value);
        }
      }
    }
    return out;
  }

  struct covering {
    std::vector<row> rows;  // The seeds first, then the generated rows.
    std::size_t pairs {}, covered {};  // Legal value pairs, and those some row holds.
    std::size_t unfilled {};  // Generated rows no legal value could complete (dropped).
  };

  namespace detail {
    // Value pairs of dimensions i < j, flattened.
    struct pair_table {
      std::vector<std::size_t> offset;  // [i * n + j]
      std::vector<std::size_t> width;
      std::size_t count {};

      explicit pair_table(const family &owner) {
        const auto n = owner.dimensions.size();
        offset.assign(n * n, 0);
        for (const auto &value : owner.dimensions) {
          width.push_back(value.values.size());
        }
        for (std::size_t i = 0; i != n; ++i) {
          for (std::size_t j = i + 1; j < n; ++j) {
            offset[i * n + j] = count;
            count += width[i] * width[j];
          }
        }
      }

      std::size_t at(std::size_t i, int a, std::size_t j, int b) const {
        if (j < i) {
          std::swap(i, j);
          std::swap(a, b);
        }
        return offset[i * width.size() + j] + std::size_t(a) * width[j] + std::size_t(b);
      }
    };

    // Marks every pair of assigned values of value; returns how many were new.
    inline std::size_t cover(const pair_table &pairs, std::vector<char> &covered, const row &value) {
      std::size_t added {};
      for (std::size_t i = 0; i != value.size(); ++i) {
        for (std::size_t j = i + 1; j < value.size(); ++j) {
          if (value[i] >= 0 && value[j] >= 0) {
            auto &slot = covered[pairs.at(i, value[i], j, value[j])];
            added += !slot;
            slot = 1;
          }
        }
      }
      return added;
    }
  }  // namespace detail

  // IPOG with constraints. Dimensions enter in order of decreasing value
  // count (ties in declaration order). Horizontal growth gives each open row
  // the legal value of the new dimension that covers most new pairs (the
  // lowest index on a tie); vertical growth places each pair still uncovered
  // in the first open row that can take it legally, or in a new row. Open
  // rows are finally completed with the lowest legal value of each
  // unassigned dimension. Seed rows are never changed.
  inline covering pairwise(const family &owner, const std::vector<row> &seeds) {
    const auto n = owner.dimensions.size();
    detail::pair_table pairs(owner);
    std::vector<char> feasible(pairs.count), covered(pairs.count);
    covering out;
    row probe(n, -1);
    for (std::size_t i = 0; i != n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        for (int a = 0; a != int(pairs.width[i]); ++a) {
          for (int b = 0; b != int(pairs.width[j]); ++b) {
            probe[i] = a;
            probe[j] = b;
            feasible[pairs.at(i, a, j, b)] = owner.legal(probe);
          }
        }
        probe[j] = -1;
      }
      probe[i] = -1;
    }
    for (const auto &seed : seeds) {
      if (complete(seed) && owner.legal(seed)) {
        out.rows.push_back(seed);
        detail::cover(pairs, covered, seed);
      }
    }
    const auto first_open = out.rows.size();
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i != n; ++i) {
      order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return pairs.width[a] > pairs.width[b];
    });
    const auto open = [&](std::size_t index) {
      return index >= first_open;
    };
    for (std::size_t step = 1; step < n; ++step) {
      const auto k = order[step];
      // Horizontal growth.
      for (std::size_t r = first_open; r < out.rows.size(); ++r) {
        auto &value = out.rows[r];
        if (value[k] >= 0) {
          continue;
        }
        int best = -1;
        long best_gain = -1;
        for (int candidate = 0; candidate != int(pairs.width[k]); ++candidate) {
          value[k] = candidate;
          if (!owner.legal(value)) {
            continue;
          }
          long gain {};
          for (std::size_t s = 0; s != step; ++s) {
            const auto j = order[s];
            gain += value[j] >= 0 && !covered[pairs.at(j, value[j], k, candidate)];
          }
          if (gain > best_gain) {
            best = candidate;
            best_gain = gain;
          }
        }
        value[k] = best;
        detail::cover(pairs, covered, value);
      }
      // Vertical growth.
      for (std::size_t s = 0; s != step; ++s) {
        const auto j = order[s];
        for (int b = 0; b != int(pairs.width[j]); ++b) {
          for (int a = 0; a != int(pairs.width[k]); ++a) {
            const auto slot = pairs.at(j, b, k, a);
            if (!feasible[slot] || covered[slot]) {
              continue;
            }
            bool placed = false;
            for (std::size_t r = 0; r < out.rows.size() && !placed; ++r) {
              if (!open(r)) {
                continue;
              }
              auto value = out.rows[r];
              if ((value[j] >= 0 && value[j] != b) || (value[k] >= 0 && value[k] != a)) {
                continue;
              }
              value[j] = b;
              value[k] = a;
              if (owner.legal(value)) {
                out.rows[r] = value;
                detail::cover(pairs, covered, value);
                placed = true;
              }
            }
            if (!placed) {
              row value(n, -1);
              value[j] = b;
              value[k] = a;
              out.rows.push_back(value);
              detail::cover(pairs, covered, value);
            }
          }
        }
      }
    }
    // Complete the open rows.
    std::vector<row> rows(out.rows.begin(), out.rows.begin() + std::ptrdiff_t(first_open));
    for (std::size_t r = first_open; r < out.rows.size(); ++r) {
      auto value = out.rows[r];
      bool filled = true;
      for (std::size_t i = 0; i != n && filled; ++i) {
        if (value[i] >= 0) {
          continue;
        }
        filled = false;
        for (int candidate = 0; candidate != int(pairs.width[i]) && !filled; ++candidate) {
          value[i] = candidate;
          filled = owner.legal(value);
        }
      }
      if (filled && owner.legal(value)) {
        rows.push_back(value);
      } else {
        ++out.unfilled;
      }
    }
    out.rows.swap(rows);
    std::fill(covered.begin(), covered.end(), 0);
    for (const auto &value : out.rows) {
      detail::cover(pairs, covered, value);
    }
    for (std::size_t i = 0; i != pairs.count; ++i) {
      out.pairs += feasible[i] != 0;
      out.covered += feasible[i] && covered[i];
    }
    return out;
  }

  // Appends the rows of more whose ids rows does not hold yet.
  inline void append_unique(const family &owner, std::vector<row> &rows, const std::vector<row> &more) {
    std::vector<std::uint64_t> ids;
    for (const auto &value : rows) {
      ids.push_back(case_id(owner, value));
    }
    for (const auto &value : more) {
      const auto id = case_id(owner, value);
      if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
        ids.push_back(id);
        rows.push_back(value);
      }
    }
  }

  // The matrix executables' command line:
  //   --list            print every case (family, id, assignment) and run none
  //   --case <id>       run one case by id
  //   --case <family>:<name=value,...>   run one case by assignment
  //   --shard <i>/<N>   run the cases whose position modulo N is i
  struct options {
    bool list {};
    std::string single;
    unsigned shard {}, shards {1};

    bool partial() const {
      return list || !single.empty() || shards != 1;
    }

    // Whether the case at this position of its family's list runs.
    bool selected(std::size_t position) const {
      return position % shards == shard;
    }
  };

  inline bool parse_options(int argc, char **argv, options &out, std::string &error) {
    out = {};
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--list") {
        out.list = true;
      } else if (argument == "--case" && i + 1 < argc) {
        out.single = argv[++i];
      } else if (argument == "--shard" && i + 1 < argc) {
        char *end {};
        const auto text = argv[++i];
        out.shard = unsigned(std::strtoul(text, &end, 10));
        const bool separated = end && *end == '/';
        out.shards = separated ? unsigned(std::strtoul(end + 1, &end, 10)) : 0;
        if (!separated || !out.shards || out.shard >= out.shards || (end && *end)) {
          error = "--shard takes i/N with i < N";
          return false;
        }
      } else {
        error = "unknown argument '" + argument + "' (--list, --case <id|family:name=value,...>, --shard i/N)";
        return false;
      }
    }
    return true;
  }
}  // namespace game3d_contract::generator
