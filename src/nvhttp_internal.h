/**
 * @file src/nvhttp_internal.h
 * @brief What the GameStream endpoints, pairing and the paired-client store share.
 */
#pragma once

// standard includes
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

// lib includes
#include <Simple-Web-Server/server_http.hpp>

// local includes
#include "crypto.h"
#include "http_request_logging.h"
#include "nvhttp.h"

namespace nvhttp {
  using p_named_cert_t = crypto::p_named_cert_t;
  using PERM = crypto::PERM;
  using args_t = SimpleWeb::CaseInsensitiveMultimap;

  using resp_https_t = std::shared_ptr<typename SimpleWeb::ServerBase<SunshineHTTPS>::Response>;
  using req_https_t = std::shared_ptr<typename SimpleWeb::ServerBase<SunshineHTTPS>::Request>;
  using resp_http_t = std::shared_ptr<typename SimpleWeb::ServerBase<SimpleWeb::HTTP>::Response>;
  using req_http_t = std::shared_ptr<typename SimpleWeb::ServerBase<SimpleWeb::HTTP>::Request>;

  inline std::string get_arg(const args_t &args, const char *name, const char *default_value = nullptr) {
    auto it = args.find(name);
    if (it == std::end(args)) {
      if (default_value != nullptr) {
        return std::string(default_value);
      }

      throw std::out_of_range(name);
    }
    return it->second;
  }

  template<class T>
  struct tunnel;

  template<>
  struct tunnel<SunshineHTTPS> {
    static constexpr std::string_view to_string = "HTTPS";
  };

  template<>
  struct tunnel<SimpleWeb::HTTP> {
    static constexpr std::string_view to_string = "NONE";
  };

  template<class T>
  void print_req(std::shared_ptr<typename SimpleWeb::ServerBase<T>::Request> request) {
    http::log_request_debug(*request, tunnel<T>::to_string);
  }

  // Pairing: nvhttp_pairing.cpp

  /** The /pair endpoint, served over both HTTP and HTTPS. */
  template<class T>
  void pair(std::shared_ptr<typename SimpleWeb::ServerBase<T>::Response> response, std::shared_ptr<typename SimpleWeb::ServerBase<T>::Request> request);
  extern template void pair<SunshineHTTPS>(resp_https_t response, req_https_t request);
  extern template void pair<SimpleWeb::HTTP>(resp_http_t response, req_http_t request);

  /** Forget pending pairings and the one-time PIN. Only after every server handler joined. */
  void clear_pairing_state();

  // The paired-client store: nvhttp_client_store.cpp

  /** Read the state file: the host's unique ID and the authorized clients. */
  void load_state();
  /** Authorize a newly paired client; re-pairing an authorized certificate changes nothing. */
  void add_authorized_client(const p_named_cert_t &named_cert_p);
  /** The current record of a client, or null once it is no longer authorized. */
  p_named_cert_t refresh_authorized_client(const p_named_cert_t &identity);
  /** Verify a TLS peer against the authorized clients. Null on success, else the reason. */
  const char *verify_authorized_client(crypto::x509_t::element_type *x509, p_named_cert_t &named_cert_p);
  /** Revoke every client. The caller has already ended the active session. */
  void revoke_all_clients();
}  // namespace nvhttp
