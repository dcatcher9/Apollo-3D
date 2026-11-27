/**
 * @file src/nvhttp_client_store.cpp
 * @brief The paired-client store: authorized certificates, their names and permissions, and
 * their persistence in the state file.
 */
// macros
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// lib includes
#include <nlohmann/json.hpp>

// local includes
#include "config.h"
#include "crypto.h"
#include "httpcommon.h"
#include "logging.h"
#include "nvhttp.h"
#include "nvhttp_internal.h"
#include "rtsp.h"
#include "system_tray.h"
#include "uuid.h"

using namespace std::literals;

namespace nvhttp {
  namespace fs = std::filesystem;

  struct client_t {
    std::vector<p_named_cert_t> named_devices;
  };

  crypto::cert_chain_t cert_chain;

  // Authorized-client objects are immutable once published. Updates replace the shared object so
  // an in-flight verified request keeps a coherent permission/name snapshot.
  std::shared_mutex client_state_mutex;
  std::mutex state_file_mutex;
  std::atomic<std::uint64_t> client_state_generation {0};
  client_t client_root;

  void rebuild_cert_chain_locked() {
    cert_chain.clear();
    for (auto &client : client_root.named_devices) {
      cert_chain.add(client);
    }
  }

  std::string canonicalize_certificate(std::string_view certificate) {
    auto parsed = crypto::x509(certificate);
    return parsed ? crypto::pem(parsed) : std::string {};
  }

  p_named_cert_t find_authorized_client_locked(std::string_view certificate) {
    const auto client = std::find_if(
      client_root.named_devices.begin(),
      client_root.named_devices.end(),
      [&](const p_named_cert_t &candidate) {
        return candidate->cert == certificate;
      }
    );
    return client == client_root.named_devices.end() ? nullptr : *client;
  }

  p_named_cert_t refresh_authorized_client(const p_named_cert_t &identity) {
    if (!identity) {
      return nullptr;
    }

    std::shared_lock lock(client_state_mutex);
    return find_authorized_client_locked(identity->cert);
  }

  void save_state() {
    nlohmann::json named_cert_nodes = nlohmann::json::array();
    std::uint64_t snapshot_generation = 0;
    {
      std::shared_lock lock(client_state_mutex);
      snapshot_generation = client_state_generation.load(std::memory_order_relaxed);
      std::unordered_set<std::string> unique_certs;
      std::unordered_map<std::string, int> name_counts;

      for (const auto &named_cert_p : client_root.named_devices) {
        // Only add each unique certificate once.
        if (!unique_certs.insert(named_cert_p->cert).second) {
          continue;
        }
        nlohmann::json named_cert_node = nlohmann::json::object();
        std::string base_name = named_cert_p->name;
        // Remove any pending id suffix (e.g., " (2)") if present.
        // Only the generated form is a suffix: digits in parentheses at the very end. A name the
        // user chose, such as "Quest (bedroom)", must survive a restart unchanged.
        const size_t pos = base_name.rfind(" (");
        if (pos != std::string::npos && base_name.size() > pos + 3 && base_name.back() == ')') {
          const std::string_view digits {base_name.data() + pos + 2, base_name.size() - pos - 3};
          if (std::all_of(digits.begin(), digits.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0;
              })) {
            base_name = base_name.substr(0, pos);
          }
        }
        const int count = name_counts[base_name]++;
        std::string final_name = base_name;
        if (count > 0) {
          final_name += " (" + std::to_string(count + 1) + ")";
        }
        named_cert_node["name"] = final_name;
        named_cert_node["cert"] = named_cert_p->cert;
        named_cert_node["uuid"] = named_cert_p->uuid;
        named_cert_node["perm"] = static_cast<uint32_t>(named_cert_p->perm);
        named_cert_nodes.push_back(std::move(named_cert_node));
      }
    }

    // Only serialize disk access. Client verification and UI reads remain available throughout
    // file I/O. If a newer mutation happened before this writer acquired the file lock, its own
    // synchronous save will persist the newer snapshot, so this stale writer can be discarded.
    std::lock_guard file_lock(state_file_mutex);
    if (snapshot_generation != client_state_generation.load(std::memory_order_acquire)) {
      return;
    }

    nlohmann::json root = nlohmann::json::object();
    if (fs::exists(config::nvhttp.file_state)) {
      try {
        std::ifstream in(config::nvhttp.file_state);
        in >> root;
      } catch (std::exception &e) {
        BOOST_LOG(error) << "Couldn't read "sv << config::nvhttp.file_state << ": "sv << e.what();
        return;
      }
    }
    root.erase("root");
    root["root"] = nlohmann::json::object();
    root["root"]["uniqueid"] = http::unique_id;
    root["root"]["named_devices"] = std::move(named_cert_nodes);

    try {
      std::ofstream out(config::nvhttp.file_state);
      out << root.dump(4);  // Pretty-print with an indent of 4 spaces.
    } catch (std::exception &e) {
      BOOST_LOG(error) << "Couldn't write "sv << config::nvhttp.file_state << ": "sv << e.what();
      return;
    }
  }

  void load_state() {
    nlohmann::json tree;
    {
      std::lock_guard file_lock(state_file_mutex);
      if (!fs::exists(config::nvhttp.file_state)) {
        BOOST_LOG(info) << "File "sv << config::nvhttp.file_state << " doesn't exist"sv;
        http::unique_id = uuid_util::uuid_t::generate().string();
        return;
      }

      try {
        std::ifstream in(config::nvhttp.file_state);
        in >> tree;
      } catch (std::exception &e) {
        BOOST_LOG(error) << "Couldn't read "sv << config::nvhttp.file_state << ": "sv << e.what();
        return;
      }
    }

    // Check that the file contains a "root.uniqueid" value.
    if (!tree.contains("root") || !tree["root"].contains("uniqueid")) {
      http::uuid = uuid_util::uuid_t::generate();
      http::unique_id = http::uuid.string();
      return;
    }

    std::string uid = tree["root"]["uniqueid"];
    http::uuid = uuid_util::uuid_t::parse(uid);
    http::unique_id = uid;

    const auto &root = tree["root"];
    client_t client;  // Local client to load into
    std::unordered_set<std::string> loaded_certificates;

    if (root.contains("named_devices")) {
      for (auto &el : root["named_devices"]) {
        auto certificate = canonicalize_certificate(el.value("cert", ""));
        if (certificate.empty() || !loaded_certificates.insert(certificate).second) {
          continue;
        }
        auto named_cert_p = std::make_shared<crypto::named_cert_t>();
        named_cert_p->name = el.value("name", "");
        named_cert_p->cert = std::move(certificate);
        named_cert_p->uuid = el.value("uuid", "");
        // A record without an unsigned permission word, from an old or hand-edited state file,
        // stays paired with no rights until the owner grants them on the devices page.
        const auto perm = el.find("perm");
        named_cert_p->perm = perm != el.end() && perm->is_number_unsigned() ?
                               static_cast<PERM>(perm->get<std::uint32_t>()) & PERM::_all :
                               PERM::_no;
        client.named_devices.emplace_back(named_cert_p);
      }
    }

    {
      std::unique_lock lock(client_state_mutex);
      client_root = std::move(client);
      rebuild_cert_chain_locked();
      client_state_generation.fetch_add(1, std::memory_order_release);
    }
  }

  void add_authorized_client(const p_named_cert_t &named_cert_p) {
    auto certificate = canonicalize_certificate(named_cert_p->cert);
    if (certificate.empty()) {
      BOOST_LOG(warning) << "Refusing to authorize an invalid client certificate";
      return;
    }
    named_cert_p->cert = std::move(certificate);

    rtsp_stream::client_policy_publication_t policy_publication;
    {
      std::unique_lock lock(client_state_mutex);
      // A certificate is the client identity. Re-pairing an already-authorized certificate is
      // idempotent: preserve its UUID, administrator-selected permissions, commands, and name.
      // This also makes simultaneous duplicate pair completions a single atomic insertion.
      if (find_authorized_client_locked(named_cert_p->cert)) {
        return;
      }

      // The first-client privilege grant and insertion are one transaction. Two simultaneous
      // successful pairings can no longer both observe an empty list and receive full control.
      named_cert_p->perm = client_root.named_devices.empty() ? PERM::_all : PERM::_default;
      client_root.named_devices.push_back(named_cert_p);
      auto authorized_client = named_cert_p;
      cert_chain.add(authorized_client);
      const auto generation = client_state_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
      policy_publication = rtsp_stream::stage_client_policy(
        named_cert_p->uuid,
        generation,
        named_cert_p->name,
        named_cert_p->perm,
        false
      );
    }
    rtsp_stream::complete_client_policy(std::move(policy_publication));

    system_tray::update_tray_paired(named_cert_p->name);

    if (!config::sunshine.flags[config::flag::FRESH_STATE]) {
      save_state();
    }
  }

#ifdef SUNSHINE_TESTS
  void add_authorized_client_for_test(std::string name, std::string certificate) {
    auto client = std::make_shared<crypto::named_cert_t>();
    client->name = std::move(name);
    client->cert = std::move(certificate);
    client->uuid = uuid_util::uuid_t::generate().string();
    add_authorized_client(client);
  }

  p_named_cert_t get_authorized_client_for_test(std::string certificate) {
    auto identity = std::make_shared<crypto::named_cert_t>();
    identity->cert = canonicalize_certificate(certificate);
    return refresh_authorized_client(identity);
  }

  p_named_cert_t refresh_authorized_client_for_test(const p_named_cert_t &cached) {
    return refresh_authorized_client(cached);
  }
#endif

  nlohmann::json get_all_clients() {
    nlohmann::json named_cert_nodes = nlohmann::json::array();
    const auto connected_uuid = rtsp_stream::active_session_uuid();
    std::shared_lock lock(client_state_mutex);

    for (const auto &named_cert : client_root.named_devices) {
      nlohmann::json named_cert_node;
      named_cert_node["name"] = named_cert->name;
      named_cert_node["uuid"] = named_cert->uuid;
      named_cert_node["perm"] = static_cast<uint32_t>(named_cert->perm);

      named_cert_node["connected"] = connected_uuid && *connected_uuid == named_cert->uuid;

      named_cert_nodes.push_back(std::move(named_cert_node));
    }

    return named_cert_nodes;
  }

  const char *verify_authorized_client(crypto::x509_t::element_type *x509, p_named_cert_t &named_cert_p) {
    // cert_chain_t reuses one OpenSSL X509_STORE_CTX internally, so verification itself must
    // be exclusive even when the authorized-client list is unchanged.
    std::unique_lock lock(client_state_mutex);
    return cert_chain.verify(x509, named_cert_p);
  }

  void revoke_all_clients() {
    std::vector<std::pair<std::string, std::string>> revoked_clients;
    std::vector<rtsp_stream::client_policy_publication_t> policy_publications;
    {
      std::unique_lock lock(client_state_mutex);
      revoked_clients.reserve(client_root.named_devices.size());
      for (const auto &client : client_root.named_devices) {
        revoked_clients.emplace_back(client->uuid, client->name);
      }
      client_root.named_devices.clear();
      cert_chain.clear();
      const auto generation = client_state_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
      policy_publications.reserve(revoked_clients.size());
      for (const auto &[uuid, name] : revoked_clients) {
        policy_publications.push_back(
          rtsp_stream::stage_client_policy(uuid, generation, name, crypto::PERM::_no, true)
        );
      }
    }
    for (auto &publication : policy_publications) {
      rtsp_stream::complete_client_policy(std::move(publication));
    }
    if (!config::sunshine.flags[config::flag::FRESH_STATE]) {
      save_state();
    }
  }

#ifdef SUNSHINE_TESTS
  bool is_client_certificate_authorized(std::string_view certificate) {
    auto x509 = crypto::x509(certificate);
    if (!x509) {
      return false;
    }
    p_named_cert_t named_client;
    std::unique_lock lock(client_state_mutex);
    return cert_chain.verify(x509.get(), named_client) == nullptr && named_client != nullptr;
  }
#endif

  bool update_device_info(
    const std::string &uuid,
    const std::string &name,
    const crypto::PERM newPerm
  ) {
    bool updated = false;
    rtsp_stream::client_policy_publication_t policy_publication;
    {
      std::unique_lock lock(client_state_mutex);
      for (auto &named_cert_p : client_root.named_devices) {
        if (named_cert_p->uuid != uuid) {
          continue;
        }
        auto replacement = std::make_shared<crypto::named_cert_t>(*named_cert_p);
        replacement->name = name;
        replacement->perm = newPerm;
        named_cert_p = std::move(replacement);
        rebuild_cert_chain_locked();
        const auto generation = client_state_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
        policy_publication = rtsp_stream::stage_client_policy(uuid, generation, name, newPerm, false);
        updated = true;
        break;
      }
    }
    if (updated) {
      rtsp_stream::complete_client_policy(std::move(policy_publication));
    }
    if (updated && !config::sunshine.flags[config::flag::FRESH_STATE]) {
      save_state();
    }
    return updated;
  }

  bool unpair_client(const std::string_view uuid) {
    bool removed = false;
    std::string client_name;
    rtsp_stream::client_policy_publication_t policy_publication;
    {
      std::unique_lock lock(client_state_mutex);
      for (auto it = client_root.named_devices.begin(); it != client_root.named_devices.end();) {
        if ((*it)->uuid == uuid) {
          client_name = (*it)->name;
          it = client_root.named_devices.erase(it);
          removed = true;
        } else {
          ++it;
        }
      }
      if (removed) {
        rebuild_cert_chain_locked();
        const auto generation = client_state_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
        policy_publication = rtsp_stream::stage_client_policy(
          uuid,
          generation,
          client_name,
          crypto::PERM::_no,
          true
        );
      }
    }

    if (removed) {
      rtsp_stream::complete_client_policy(std::move(policy_publication));

      if (!config::sunshine.flags[config::flag::FRESH_STATE]) {
        save_state();
      }
    }

    return removed;
  }
}  // namespace nvhttp
