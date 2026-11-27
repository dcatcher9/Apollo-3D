/**
 * @file src/confighttp_offline_sbs.cpp
 * @brief Web UI routes for Convert Video (offline Host 3D conversion jobs).
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <filesystem>
#include <set>
#include <string>
#include <utility>

// lib includes
#include <nlohmann/json.hpp>

// local includes
#include "confighttp.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"
#include "logging.h"
#include "network.h"
#include "offline_sbs_job.h"

using namespace std::literals;

namespace confighttp {
  namespace fs = std::filesystem;

  bool validateOfflineSbsRequestSize(
    resp_https_t response,
    req_https_t request
  ) {
    const auto content_length = get_single_request_header(
      request,
      "Content-Length"
    );
    if (!content_length.unique) {
      bad_request(
        response,
        request,
        "Offline SBS request has duplicate Content-Length headers"
      );
      return false;
    }

    switch (validate_bounded_content_length(
      content_length.value,
      OFFLINE_SBS_REQUEST_MAX_BYTES
    )) {
      case bounded_content_length_e::invalid:
        bad_request(
          response,
          request,
          "Offline SBS request has an invalid Content-Length"
        );
        return false;
      case bounded_content_length_e::exceeds_limit:
        send_json_response(
          response,
          SimpleWeb::StatusCode::client_error_payload_too_large,
          {
            {"status", false},
            {"error_code", "invalid_request"},
            {"error", "Offline SBS request exceeds the 64 KiB limit"},
          }
        );
        return false;
      case bounded_content_length_e::absent_or_within_limit:
        break;
    }

    // This independently bounds chunked requests and catches a mismatched/omitted
    // Content-Length before copying or parsing the JSON document.
    if (request->content.size() > OFFLINE_SBS_REQUEST_MAX_BYTES) {
      send_json_response(
        response,
        SimpleWeb::StatusCode::client_error_payload_too_large,
        {
          {"status", false},
          {"error_code", "invalid_request"},
          {"error", "Offline SBS request exceeds the 64 KiB limit"},
        }
      );
      return false;
    }
    return true;
  }

  template<class Reply>
  SimpleWeb::StatusCode offline_reply_status(
    const Reply &reply,
    SimpleWeb::StatusCode success
  ) {
    if (reply.ok) {
      return success;
    }
    switch (reply.code) {
      case offline_sbs::error_code_e::invalid_request:
        return SimpleWeb::StatusCode::client_error_bad_request;
      case offline_sbs::error_code_e::not_found:
        return SimpleWeb::StatusCode::client_error_not_found;
      case offline_sbs::error_code_e::busy:
      case offline_sbs::error_code_e::conflict:
        return SimpleWeb::StatusCode::client_error_conflict;
      case offline_sbs::error_code_e::not_initialized:
      case offline_sbs::error_code_e::unavailable:
        return SimpleWeb::StatusCode::server_error_service_unavailable;
      case offline_sbs::error_code_e::io_error:
        return SimpleWeb::StatusCode::server_error_internal_server_error;
      case offline_sbs::error_code_e::none:
        return SimpleWeb::StatusCode::server_error_internal_server_error;
    }
    return SimpleWeb::StatusCode::server_error_internal_server_error;
  }

  void getOfflineSbsCapabilities(
    resp_https_t response,
    req_https_t request
  ) {
    offline_sbs::request_initialization();
    const auto capabilities = offline_sbs::capabilities();
    const bool available = capabilities.value("available", false);
    send_json_response(
      response,
      available ?
        SimpleWeb::StatusCode::success_ok :
        SimpleWeb::StatusCode::server_error_service_unavailable,
      {
        {"status", available},
        {"capabilities", capabilities},
        {"error", available ?
                    nlohmann::json(nullptr) :
                    nlohmann::json(capabilities.value(
                      "error",
                      "offline SBS job manager is unavailable"
                    ))},
      }
    );
  }

  void browseOfflineSbsFilesystem(
    resp_https_t response,
    req_https_t request
  ) {
    if (!validateContentType(response, request, "application/json") ||
        !validateOfflineSbsRequestSize(response, request)) {
      return;
    }
    const auto address = request->remote_endpoint().address();
    if (net::from_address(address) != net::PC) {
      BOOST_LOG(warning)
        << "Web UI: ["sv << net::addr_to_normalized_string(address)
        << "] -- rejected non-loopback offline filesystem browse"sv;
      send_json_response(
        response,
        SimpleWeb::StatusCode::client_error_forbidden,
        offline_sbs::browse_reply_t {
          .code = offline_sbs::error_code_e::unavailable,
          .error =
            "Host filesystem browsing is available only from the Sunshine 3D PC",
        }.json()
      );
      return;
    }

    const auto reject_request = [&](std::string error) {
      send_json_response(
        response,
        SimpleWeb::StatusCode::client_error_bad_request,
        offline_sbs::browse_reply_t {
          .code = offline_sbs::error_code_e::invalid_request,
          .error = std::move(error),
        }.json()
      );
    };
    try {
      const auto input = nlohmann::json::parse(request->content);
      if (!input.is_object()) {
        reject_request("Offline browse request must be an object");
        return;
      }
      static const std::set<std::string> allowed {"path", "type"};
      for (const auto &[name, value] : input.items()) {
        if (!allowed.contains(name)) {
          reject_request("Offline browse request contains an unknown field");
          return;
        }
      }

      const auto path = input.value("path", std::string {});
      const auto requested_type = input.value("type", std::string {"any"});
      offline_sbs::browse_request_t browse;
      if (!path.empty()) {
        if (path.size() > 32ull * 1024ull) {
          reject_request("Offline browse path exceeds the 32 KiB UTF-8 limit");
          return;
        }
        std::u8string utf8_path;
        utf8_path.assign(
          reinterpret_cast<const char8_t *>(path.data()),
          reinterpret_cast<const char8_t *>(path.data() + path.size())
        );
        browse.path = fs::path {utf8_path};
      }
      if (requested_type == "file") {
        browse.type = offline_sbs::browse_type_e::file;
      } else if (requested_type == "directory") {
        browse.type = offline_sbs::browse_type_e::directory;
      } else if (requested_type == "any") {
        browse.type = offline_sbs::browse_type_e::any;
      } else {
        reject_request("Offline browse type must be file, directory, or any");
        return;
      }

      offline_sbs::request_initialization();
      const auto reply = offline_sbs::browse(browse);
      send_json_response(
        response,
        offline_reply_status(reply, SimpleWeb::StatusCode::success_ok),
        reply.json()
      );
    } catch (const std::exception &) {
      reject_request("Offline browse request must contain valid JSON strings");
    }
  }

  void listOfflineSbsJobs(resp_https_t response, req_https_t request) {
    offline_sbs::request_initialization();
    const auto capabilities = offline_sbs::capabilities();
    if (!capabilities.value("available", false)) {
      send_json_response(
        response,
        SimpleWeb::StatusCode::server_error_service_unavailable,
        {
          {"status", false},
          {"jobs", nlohmann::json::array()},
          {"capabilities", capabilities},
          {"error", capabilities.value(
            "error",
            "offline SBS job manager is unavailable"
          )},
        }
      );
      return;
    }

    auto jobs = nlohmann::json::array();
    for (const auto &job : offline_sbs::list()) {
      jobs.push_back(job.json());
    }
    send_json_response(
      response,
      SimpleWeb::StatusCode::success_ok,
      {
        {"status", true},
        {"jobs", std::move(jobs)},
        {"capabilities", capabilities},
        {"error", nullptr},
      }
    );
  }

  void createOfflineSbsJob(resp_https_t response, req_https_t request) {
    if (!validateContentType(response, request, "application/json") ||
        !validateOfflineSbsRequestSize(response, request)) {
      return;
    }
    try {
      const auto input = nlohmann::json::parse(request->content);
      if (!input.is_object()) {
        bad_request(response, request, "Offline conversion request must be an object");
        return;
      }
      static const std::set<std::string> allowed {
        "input_path",
        "operation",
        "output_name",
        "codec",
      };
      for (const auto &[name, value] : input.items()) {
        if (!allowed.contains(name)) {
          bad_request(
            response,
            request,
            "Unknown offline conversion field: " + name
          );
          return;
        }
      }
      if (!input.contains("input_path") || !input["input_path"].is_string()) {
        bad_request(response, request, "input_path must be a string");
        return;
      }

      offline_sbs::create_request_t create;
      {
        // The request is UTF-8. A narrow std::string would be decoded in the ANSI code page,
        // so build the path the same way the browse endpoint does.
        const auto input_path = input["input_path"].get<std::string>();
        create.input_path = fs::path {std::u8string {
          reinterpret_cast<const char8_t *>(input_path.data()),
          reinterpret_cast<const char8_t *>(input_path.data() + input_path.size())
        }};
      }
      const auto operation = input.value("operation", std::string {"convert"});
      if (operation == "convert") {
        create.operation = offline_sbs::operation_e::convert;
      } else if (operation == "evaluate") {
        create.operation = offline_sbs::operation_e::evaluate;
      } else {
        bad_request(response, request, "operation must be convert or evaluate");
        return;
      }
      if (input.contains("output_name")) {
        if (!input["output_name"].is_string()) {
          bad_request(response, request, "output_name must be a string");
          return;
        }
        create.output_name = input["output_name"].get<std::string>();
      }
      if (input.contains("codec")) {
        if (!input["codec"].is_string()) {
          bad_request(response, request, "codec must be a string");
          return;
        }
        create.codec = input["codec"].get<std::string>();
      }

      offline_sbs::request_initialization();
      const auto reply = offline_sbs::create(create);
      send_json_response(
        response,
        offline_reply_status(
          reply,
          SimpleWeb::StatusCode::success_created
        ),
        reply.json()
      );
    } catch (const std::exception &error) {
      bad_request(response, request, error.what());
    }
  }

  void getOfflineSbsJob(resp_https_t response, req_https_t request) {
    offline_sbs::request_initialization();
    const auto reply = offline_sbs::get(request->path_match[1].str());
    send_json_response(
      response,
      offline_reply_status(reply, SimpleWeb::StatusCode::success_ok),
      reply.json()
    );
  }

  void cancelOfflineSbsJob(resp_https_t response, req_https_t request) {
    offline_sbs::request_initialization();
    const auto reply = offline_sbs::cancel(request->path_match[1].str());
    send_json_response(
      response,
      offline_reply_status(reply, SimpleWeb::StatusCode::success_accepted),
      reply.json()
    );
  }

  void clearOfflineSbsJob(resp_https_t response, req_https_t request) {
    offline_sbs::request_initialization();
    const auto reply = offline_sbs::clear(request->path_match[1].str());
    send_json_response(
      response,
      offline_reply_status(reply, SimpleWeb::StatusCode::success_ok),
      reply.json()
    );
  }
}  // namespace confighttp
