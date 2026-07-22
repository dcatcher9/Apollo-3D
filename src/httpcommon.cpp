/**
 * @file src/httpcommon.cpp
 * @brief Definitions for common HTTP.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>
#include <utility>

// lib includes
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/context_base.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <curl/curl.h>
#include <Simple-Web-Server/server_http.hpp>
#include <Simple-Web-Server/server_https.hpp>

// local includes
#include "config.h"
#include "crypto.h"
#include "file_handler.h"
#include "httpcommon.h"
#include "logging.h"
#include "network.h"
#include "nvhttp.h"
#include "platform/common.h"
#include "process.h"
#include "rtsp.h"
#include "utility.h"

namespace http {
  using namespace std::literals;
  namespace fs = std::filesystem;
  namespace pt = boost::property_tree;

  namespace {
    struct web_origin_t {
      std::string host;
      std::uint16_t port;
    };

    bool contains_invalid_header_character(const std::string_view value) {
      if (value.empty()) {
        return true;
      }

      return std::ranges::any_of(value, [](const unsigned char character) {
        return character <= 0x20 || character == 0x7f || character == ',';
      });
    }

    bool has_valid_https_authority_syntax(const std::string_view value) {
      constexpr auto prefix = "https://"sv;
      if (value.size() <= prefix.size() || !boost::iequals(value.substr(0, prefix.size()), prefix)) {
        return false;
      }

      const auto authority_end = value.find_first_of("/?#", prefix.size());
      const auto authority = value.substr(
        prefix.size(),
        authority_end == std::string_view::npos ? std::string_view::npos : authority_end - prefix.size()
      );
      if (authority.empty() || authority.contains('@') || authority.contains('\\')) {
        return false;
      }

      std::string_view port;
      if (authority.front() == '[') {
        const auto bracket = authority.find(']');
        if (bracket == std::string_view::npos || bracket == 1 || authority.find('[', 1) != std::string_view::npos ||
            authority.find(']', bracket + 1) != std::string_view::npos) {
          return false;
        }

        const auto suffix = authority.substr(bracket + 1);
        if (suffix.empty()) {
          return true;
        }
        if (suffix.front() != ':') {
          return false;
        }
        port = suffix.substr(1);
      } else {
        if (authority.contains('[') || authority.contains(']')) {
          return false;
        }
        const auto colon = authority.find(':');
        if (colon == std::string_view::npos) {
          return true;
        }
        if (colon == 0 || authority.find(':', colon + 1) != std::string_view::npos) {
          return false;
        }
        port = authority.substr(colon + 1);
      }

      return !port.empty() &&
             std::ranges::all_of(port, [](const unsigned char character) {
               return std::isdigit(character) != 0;
             });
    }

    std::optional<std::string> curl_url_part(CURLU *url, const CURLUPart part, const unsigned int flags = 0) {
      char *value = nullptr;
      if (curl_url_get(url, part, &value, flags) != CURLUE_OK) {
        return std::nullopt;
      }

      std::string result {value};
      curl_free(value);
      return result;
    }

    std::string normalize_origin_host(std::string host) {
      if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
      }

      boost::system::error_code ec;
      const auto address = boost::asio::ip::make_address(host, ec);
      if (!ec) {
        return address.to_string();
      }

      boost::algorithm::to_lower(host);
      return host;
    }

    std::optional<web_origin_t> parse_https_origin(const std::string_view value, const bool allow_resource) {
      if (contains_invalid_header_character(value) || !has_valid_https_authority_syntax(value)) {
        return std::nullopt;
      }

      auto *url = curl_url();
      if (!url) {
        return std::nullopt;
      }
      const auto cleanup = util::fail_guard([url]() {
        curl_url_cleanup(url);
      });

      const std::string owned_value {value};
      if (curl_url_set(url, CURLUPART_URL, owned_value.c_str(), 0) != CURLUE_OK) {
        return std::nullopt;
      }

      const auto scheme = curl_url_part(url, CURLUPART_SCHEME);
      const auto host = curl_url_part(url, CURLUPART_HOST);
      const auto port = curl_url_part(url, CURLUPART_PORT, CURLU_DEFAULT_PORT);
      if (!scheme || !boost::iequals(*scheme, "https") || !host || host->empty() || !port) {
        return std::nullopt;
      }

      if (curl_url_part(url, CURLUPART_USER) || curl_url_part(url, CURLUPART_PASSWORD)) {
        return std::nullopt;
      }

      const auto path = curl_url_part(url, CURLUPART_PATH);
      if (!allow_resource && path && *path != "/") {
        return std::nullopt;
      }
      if ((!allow_resource && curl_url_part(url, CURLUPART_QUERY)) || curl_url_part(url, CURLUPART_FRAGMENT)) {
        return std::nullopt;
      }

      unsigned int parsed_port = 0;
      const auto [end, error] = std::from_chars(port->data(), port->data() + port->size(), parsed_port);
      if (error != std::errc {} || end != port->data() + port->size() || parsed_port == 0 || parsed_port > 65535) {
        return std::nullopt;
      }

      return web_origin_t {normalize_origin_host(*host), static_cast<std::uint16_t>(parsed_port)};
    }

    bool same_https_origin(const web_origin_t &expected, const std::string_view candidate, const bool allow_resource) {
      const auto parsed = parse_https_origin(candidate, allow_resource);
      return parsed && expected.port == parsed->port && boost::iequals(expected.host, parsed->host);
    }

    template<std::size_t Size>
    bool matches_sensitive_name(const std::string_view name, const std::array<std::string_view, Size> &sensitive_names) {
      return std::ranges::any_of(sensitive_names, [name](const std::string_view sensitive_name) {
        return boost::iequals(name, sensitive_name);
      });
    }
  }  // namespace

  int reload_user_creds(const std::string &file);
  bool user_creds_exist(const std::string &file);

  std::string unique_id;
  uuid_util::uuid_t uuid;
  net::net_e origin_web_ui_allowed;

  int init() {
    bool clean_slate = config::sunshine.flags[config::flag::FRESH_STATE];
    origin_web_ui_allowed = net::from_enum_string(config::nvhttp.origin_web_ui_allowed);

    if (clean_slate) {
      uuid = uuid_util::uuid_t::generate();
      unique_id = uuid.string();
      auto dir = std::filesystem::temp_directory_path() / "Sunshine"sv;
      config::nvhttp.cert = (dir / ("cert-"s + unique_id)).string();
      config::nvhttp.pkey = (dir / ("pkey-"s + unique_id)).string();
    }

    if ((!fs::exists(config::nvhttp.pkey) || !fs::exists(config::nvhttp.cert)) &&
        create_creds(config::nvhttp.pkey, config::nvhttp.cert)) {
      return -1;
    }
    if (!user_creds_exist(config::sunshine.credentials_file)) {
      BOOST_LOG(info) << "The Web UI is ready for passwordless control from this PC or the local network";
    } else if (reload_user_creds(config::sunshine.credentials_file)) {
      return -1;
    }
    return 0;
  }

  int save_user_creds(const std::string &file, const std::string &username, const std::string &password, bool run_our_mouth) {
    nlohmann::json outputTree;

    if (fs::exists(file)) {
      try {
        std::ifstream in(file);
        in >> outputTree;
      } catch (std::exception &e) {
        BOOST_LOG(error) << "Couldn't read user credentials: "sv << e.what();
        return -1;
      }
    }

    auto salt = crypto::rand_alphabet(16);
    outputTree["username"] = username;
    outputTree["salt"] = salt;
    outputTree["password"] = util::hex(crypto::hash(password + salt)).to_string();
    try {
      std::ofstream out(file);
      out << outputTree.dump(4);  // Pretty-print with an indent of 4 spaces.
    } catch (std::exception &e) {
      BOOST_LOG(error) << "error writing to the credentials file, perhaps try this again as an administrator? Details: "sv << e.what();
      return -1;
    }

    BOOST_LOG(info) << "New credentials have been created"sv;
    return 0;
  }

  bool user_creds_exist(const std::string &file) {
    if (!fs::exists(file)) {
      return false;
    }

    pt::ptree inputTree;
    try {
      pt::read_json(file, inputTree);
      return inputTree.find("username") != inputTree.not_found() &&
             inputTree.find("password") != inputTree.not_found() &&
             inputTree.find("salt") != inputTree.not_found();
    } catch (std::exception &e) {
      BOOST_LOG(error) << "validating user credentials: "sv << e.what();
    }

    return false;
  }

  int reload_user_creds(const std::string &file) {
    pt::ptree inputTree;
    try {
      pt::read_json(file, inputTree);
      config::sunshine.username = inputTree.get<std::string>("username");
      config::sunshine.password = inputTree.get<std::string>("password");
      config::sunshine.salt = inputTree.get<std::string>("salt");
    } catch (std::exception &e) {
      BOOST_LOG(error) << "loading user credentials: "sv << e.what();
      return -1;
    }
    return 0;
  }

  int create_creds(const std::string &pkey, const std::string &cert) {
    fs::path pkey_path = pkey;
    fs::path cert_path = cert;

    auto creds = crypto::gen_creds("Sunshine Gamestream Host"sv, 2048);

    auto pkey_dir = pkey_path;
    auto cert_dir = cert_path;
    pkey_dir.remove_filename();
    cert_dir.remove_filename();

    std::error_code err_code {};
    fs::create_directories(pkey_dir, err_code);
    if (err_code) {
      BOOST_LOG(error) << "Couldn't create directory ["sv << pkey_dir << "] :"sv << err_code.message();
      return -1;
    }

    fs::create_directories(cert_dir, err_code);
    if (err_code) {
      BOOST_LOG(error) << "Couldn't create directory ["sv << cert_dir << "] :"sv << err_code.message();
      return -1;
    }

    if (file_handler::write_file(pkey.c_str(), creds.pkey)) {
      BOOST_LOG(error) << "Couldn't open ["sv << config::nvhttp.pkey << ']';
      return -1;
    }

    if (file_handler::write_file(cert.c_str(), creds.x509)) {
      BOOST_LOG(error) << "Couldn't open ["sv << config::nvhttp.cert << ']';
      return -1;
    }

    fs::permissions(pkey_path, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, err_code);

    if (err_code) {
      BOOST_LOG(error) << "Couldn't change permissions of ["sv << config::nvhttp.pkey << "] :"sv << err_code.message();
      return -1;
    }

    fs::permissions(cert_path, fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read | fs::perms::owner_write, fs::perm_options::replace, err_code);

    if (err_code) {
      BOOST_LOG(error) << "Couldn't change permissions of ["sv << config::nvhttp.cert << "] :"sv << err_code.message();
      return -1;
    }

    return 0;
  }

  bool download_file(const std::string &url, const std::string &file, long ssl_version) {
    // sonar complains about weak ssl and tls versions; however sonar cannot detect the fix
    CURL *curl = curl_easy_init();  // NOSONAR
    if (!curl) {
      BOOST_LOG(error) << "Couldn't create CURL instance";
      return false;
    }

    if (std::string file_dir = file_handler::get_parent_directory(file); !file_handler::make_directory(file_dir)) {
      BOOST_LOG(error) << "Couldn't create directory ["sv << file_dir << ']';
      curl_easy_cleanup(curl);
      return false;
    }

    FILE *fp = fopen(file.c_str(), "wb");
    if (!fp) {
      BOOST_LOG(error) << "Couldn't open ["sv << file << ']';
      curl_easy_cleanup(curl);
      return false;
    }

    curl_easy_setopt(curl, CURLOPT_SSLVERSION, ssl_version);  // NOSONAR
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
#ifdef _WIN32
    curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
#endif
    CURLcode result = curl_easy_perform(curl);
    if (result != CURLE_OK) {
      BOOST_LOG(error) << "Couldn't download ["sv << url << ", code:" << result << ']';
    }
    curl_easy_cleanup(curl);
    fclose(fp);
    return result == CURLE_OK;
  }

  std::string url_escape(const std::string &url) {
    char *string = curl_easy_escape(nullptr, url.c_str(), static_cast<int>(url.length()));
    std::string result(string);
    curl_free(string);
    return result;
  }

  std::string url_get_host(const std::string &url) {
    CURLU *curlu = curl_url();
    curl_url_set(curlu, CURLUPART_URL, url.c_str(), static_cast<unsigned int>(url.length()));
    char *host;
    if (curl_url_get(curlu, CURLUPART_HOST, &host, 0) != CURLUE_OK) {
      curl_url_cleanup(curlu);
      return "";
    }
    std::string result(host);
    curl_free(host);
    curl_url_cleanup(curlu);
    return result;
  }

  bool web_ui_origin_allowed(
    const std::string_view host,
    const std::optional<std::string_view> origin,
    const std::optional<std::string_view> referer,
    const std::optional<std::string_view> sec_fetch_site
  ) {
    const auto expected = parse_https_origin("https://"s + std::string {host}, false);
    if (!expected) {
      return false;
    }

    bool has_source_metadata = false;
    if (sec_fetch_site) {
      has_source_metadata = true;
      if (!boost::iequals(*sec_fetch_site, "same-origin")) {
        return false;
      }
    }

    if (origin) {
      has_source_metadata = true;
      if (boost::iequals(*origin, "null") || !same_https_origin(*expected, *origin, false)) {
        return false;
      }
    }

    if (referer) {
      has_source_metadata = true;
      if (!same_https_origin(*expected, *referer, true)) {
        return false;
      }
    }

    return has_source_metadata;
  }

  std::string_view redact_request_header(const std::string_view name, const std::string_view value) {
    static constexpr std::array sensitive_names {
      "authorization"sv,
      "proxy-authorization"sv,
      "cookie"sv,
      "set-cookie"sv,
      "x-csrf-token"sv,
    };
    return matches_sensitive_name(name, sensitive_names) ? "CREDENTIALS REDACTED"sv : value;
  }

  std::string_view redact_query_parameter(const std::string_view name, const std::string_view value) {
    static constexpr std::array sensitive_names {
      "rikey"sv,
      "rikeyid"sv,
      "salt"sv,
      "otpauth"sv,
      "clientchallenge"sv,
      "serverchallengeresp"sv,
      "clientpairingsecret"sv,
      "clientcert"sv,
      "csrf_token"sv,
    };
    return matches_sensitive_name(name, sensitive_names) ? "CREDENTIALS REDACTED"sv : value;
  }
}  // namespace http
