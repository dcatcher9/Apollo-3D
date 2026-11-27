/**
 * @file src/nvhttp.cpp
 * @brief Definitions for the nvhttp (GameStream) server.
 *
 * The GameStream endpoints and their servers. Pairing is in nvhttp_pairing.cpp and the
 * paired-client store in nvhttp_client_store.cpp.
 */
// macros
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>

// lib includes
#include <boost/algorithm/string/predicate.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/context_base.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <Simple-Web-Server/server_http.hpp>

// local includes
#include "config.h"
#include "file_handler.h"
#include "globals.h"
#include "host_sbs_resolution.h"
#include "http_request_logging.h"
#include "httpcommon.h"
#include "logging.h"
#include "network.h"
#include "nvhttp.h"
#include "nvhttp_internal.h"
#include "platform/common.h"
#include "process.h"
#include "rtsp.h"
#include "stream.h"
#include "system_tray.h"
#include "utility.h"
#include "uuid.h"
#include "video.h"

#ifdef SUNSHINE_TESTS
  #include "nvhttp_launch_test.h"
  #include <stdexcept>
#endif

#include "platform/windows/ar_glasses.h"
#include "platform/windows/virtual_display.h"

using namespace std::literals;

namespace nvhttp {

  static constexpr std::string_view EMPTY_PROPERTY_TREE_ERROR_MSG = "Property tree is empty. Control flow was probably interrupted by an unexpected C++ exception. This is a Sunshine 3D bug; the client will report malformed XML (missing root element)."sv;

  namespace pt = boost::property_tree;

  namespace detail {
    bool app_identity_matches(
      std::optional<int> requested_app_id,
      std::string_view requested_app_uuid,
      int candidate_app_id,
      std::string_view candidate_app_uuid
    ) {
      if (!requested_app_uuid.empty() && !boost::iequals(requested_app_uuid, candidate_app_uuid)) {
        return false;
      }
      if (requested_app_id && *requested_app_id > 0 && *requested_app_id != candidate_app_id) {
        return false;
      }
      return !requested_app_uuid.empty() || (requested_app_id && *requested_app_id > 0);
    }
  }  // namespace detail

  class SunshineHTTPSServer: public SimpleWeb::ServerBase<SunshineHTTPS> {
  public:
    SunshineHTTPSServer(const std::string &certification_file, const std::string &private_key_file):
        ServerBase<SunshineHTTPS>::ServerBase(443),
        context(boost::asio::ssl::context::tls_server) {
      // Disabling TLS 1.0 and 1.1 (see RFC 8996)
      context.set_options(boost::asio::ssl::context::no_tlsv1);
      context.set_options(boost::asio::ssl::context::no_tlsv1_1);
      context.use_certificate_chain_file(certification_file);
      context.use_private_key_file(private_key_file, boost::asio::ssl::context::pem);
    }

    std::function<bool(std::shared_ptr<Request>, SSL *)> verify;
    std::function<void(std::shared_ptr<Response>, std::shared_ptr<Request>)> on_verify_failed;

  protected:
    boost::asio::ssl::context context;

    void after_bind() override {
      if (verify) {
        context.set_verify_mode(boost::asio::ssl::verify_peer | boost::asio::ssl::verify_fail_if_no_peer_cert | boost::asio::ssl::verify_client_once);
        context.set_verify_callback([](int verified, boost::asio::ssl::verify_context &ctx) {
          // To respond with an error message, a connection must be established
          return 1;
        });
      }
    }

    // This is Server<HTTPS>::accept() with SSL validation support added
    void accept() override {
      auto connection = create_connection(*io_service, context);

      acceptor->async_accept(connection->socket->lowest_layer(), [this, connection](const SimpleWeb::error_code &ec) {
        auto lock = connection->handler_runner->continue_lock();
        if (!lock) {
          return;
        }

        if (ec != SimpleWeb::error::operation_aborted) {
          this->accept();
        }

        auto session = std::make_shared<Session>(config.max_request_streambuf_size, connection);

        if (!ec) {
          boost::asio::ip::tcp::no_delay option(true);
          SimpleWeb::error_code ec;
          session->connection->socket->lowest_layer().set_option(option, ec);

          session->connection->set_timeout(config.timeout_request);
          session->connection->socket->async_handshake(boost::asio::ssl::stream_base::server, [this, session](const SimpleWeb::error_code &ec) {
            session->connection->cancel_timeout();
            auto lock = session->connection->handler_runner->continue_lock();
            if (!lock) {
              return;
            }
            if (!ec) {
              if (verify && !verify(session->request, session->connection->socket->native_handle())) {
                this->write(session, on_verify_failed);
              } else {
                this->read(session);
              }
            } else if (this->on_error) {
              this->on_error(session->request, ec);
            }
          });
        } else if (this->on_error) {
          this->on_error(session->request, ec);
        }
      });
    }
  };

  using https_server_t = SunshineHTTPSServer;
  using http_server_t = SimpleWeb::Server<SimpleWeb::HTTP>;

  std::mutex launch_request_mutex;

  namespace {
    void terminate_active_session_locked() {
      rtsp_stream::terminate_session();

      if (proc::proc.running() > 0) {
        proc::proc.terminate();
      }

      stream::session::flush_platform_state();
    }
  }  // namespace

  void terminate_active_session() {
    std::lock_guard launch_lock(launch_request_mutex);
    terminate_active_session_locked();
  }

  void reload_apps(const std::string &file_name) {
    std::lock_guard launch_lock(launch_request_mutex);
    terminate_active_session_locked();
    proc::refresh(file_name, false, false);
  }

  std::atomic<uint32_t> session_id_counter;

#ifdef SUNSHINE_TESTS
  namespace {
    thread_local const testing::launch_native_hooks_t *launch_native_hooks = nullptr;
  }
#endif

  static std::optional<std::string_view> find_arg(const args_t &args, const char *name) {
    const auto it = args.find(name);
    return it == std::end(args) ? std::nullopt : std::optional<std::string_view> {it->second};
  }

  std::optional<launch_mode_t> parse_launch_mode(std::string_view mode) {
    const auto first_separator = mode.find('x');
    if (first_separator == std::string_view::npos) {
      return std::nullopt;
    }
    const auto second_separator = mode.find('x', first_separator + 1);
    if (second_separator == std::string_view::npos || mode.find('x', second_separator + 1) != std::string_view::npos) {
      return std::nullopt;
    }

    const auto width_text = mode.substr(0, first_separator);
    const auto height_text = mode.substr(first_separator + 1, second_separator - first_separator - 1);
    const auto fps_text = mode.substr(second_separator + 1);
    const auto width = util::from_view_checked<int>(width_text);
    const auto height = util::from_view_checked<int>(height_text);
    if (!width || !height || *width < 1 || *width > 16384 || *height < 1 || *height > 16384 || fps_text.empty()) {
      return std::nullopt;
    }

    bool decimal_point_seen = false;
    for (const auto ch : fps_text) {
      if (ch == '.' && !decimal_point_seen) {
        decimal_point_seen = true;
      } else if (!std::isdigit(static_cast<unsigned char>(ch))) {
        return std::nullopt;
      }
    }

    double fps_value {};
    const auto [fps_end, fps_error] = std::from_chars(fps_text.data(), fps_text.data() + fps_text.size(), fps_value);
    if (fps_error != std::errc {} || fps_end != fps_text.data() + fps_text.size() || !std::isfinite(fps_value) || fps_value < 1.0) {
      return std::nullopt;
    }
    if (fps_value <= 1000.0) {
      fps_value *= 1000.0;
    }
    if (fps_value > 1000000.0) {
      return std::nullopt;
    }

    const auto fps = std::lround(fps_value);
    if (fps < 1000 || fps > 1000000) {
      return std::nullopt;
    }

    return launch_mode_t {*width, *height, static_cast<int>(fps)};
  }

  std::optional<int> parse_launch_int(launch_int_field field, std::string_view value) {
    const auto parsed = util::from_view_checked<int>(value);
    if (!parsed) {
      return std::nullopt;
    }

    bool valid = false;
    switch (field) {
      case launch_int_field::core_version:
        valid = *parsed >= 1;
        break;
      case launch_int_field::app_id:
        valid = *parsed >= 0;
        break;
      case launch_int_field::surround_info:
        {
          const auto channels = *parsed & 0xFFFF;
          valid = *parsed >= 0 && (channels == 2 || channels == 6 || channels == 8);
          break;
        }
      case launch_int_field::binary_option:
        valid = *parsed == 0 || *parsed == 1;
        break;
      case launch_int_field::scale_factor:
        valid = *parsed >= 20 && *parsed <= 200;
        break;
      case launch_int_field::sbs_mode:
        // Game always starts mono. Packed output requires a negotiated live transaction.
        valid = *parsed >= video::SBS_OFF && *parsed <= video::SBS_GAME_MONO;
        break;
      case launch_int_field::stream_gamma:
        valid = *parsed >= 0 && *parsed <= 2;
        break;
    }

    return valid ? parsed : std::nullopt;
  }

  std::optional<launch_display_options_t> parse_launch_display_options(
    std::optional<std::string_view> virtual_display,
    std::optional<std::string_view> scale_factor,
    std::optional<std::string_view> sbs_mode,
    std::optional<std::string_view> virtual_display_only
  ) {
    if (virtual_display_only && *virtual_display_only != "0" && *virtual_display_only != "1") {
      return std::nullopt;
    }
    const launch_display_options_t defaults;
    const auto parse_or_default = [](launch_int_field field, std::optional<std::string_view> value, int fallback) {
      return value ? parse_launch_int(field, *value) : std::optional<int> {fallback};
    };

    const auto parsed_virtual_display =
      parse_or_default(launch_int_field::binary_option, virtual_display, defaults.virtual_display);
    const auto parsed_virtual_display_only =
      parse_or_default(launch_int_field::binary_option, virtual_display_only, defaults.virtual_display_only);
    const auto parsed_scale_factor =
      parse_or_default(launch_int_field::scale_factor, scale_factor, defaults.scale_factor);
    const auto parsed_sbs_mode =
      parse_or_default(launch_int_field::sbs_mode, sbs_mode, defaults.sbs_mode);
    if (!parsed_virtual_display || !parsed_virtual_display_only || !parsed_scale_factor || !parsed_sbs_mode) {
      return std::nullopt;
    }

    return launch_display_options_t {
      static_cast<bool>(*parsed_virtual_display),
      static_cast<bool>(*parsed_virtual_display_only),
      static_cast<std::uint32_t>(*parsed_scale_factor),
      *parsed_sbs_mode,
    };
  }

  std::optional<crypto::aes_t> parse_remote_input_key(std::string_view key) {
    if (key.size() != 32) {
      return std::nullopt;
    }

    const auto decode_nibble = [](char ch) -> std::optional<std::uint8_t> {
      if (ch >= '0' && ch <= '9') {
        return static_cast<std::uint8_t>(ch - '0');
      }
      if (ch >= 'a' && ch <= 'f') {
        return static_cast<std::uint8_t>(ch - 'a' + 10);
      }
      if (ch >= 'A' && ch <= 'F') {
        return static_cast<std::uint8_t>(ch - 'A' + 10);
      }
      return std::nullopt;
    };

    crypto::aes_t result(16);
    for (std::size_t index = 0; index < result.size(); ++index) {
      const auto high = decode_nibble(key[index * 2]);
      const auto low = decode_nibble(key[index * 2 + 1]);
      if (!high || !low) {
        return std::nullopt;
      }
      result[index] = static_cast<std::uint8_t>((*high << 4) | *low);
    }

    return result;
  }

  std::optional<std::uint32_t> parse_remote_input_key_id(std::string_view key_id) {
    if (key_id.starts_with('-')) {
      const auto parsed = util::from_view_checked<std::int32_t>(key_id);
      return parsed ? std::optional<std::uint32_t> {static_cast<std::uint32_t>(*parsed)} : std::nullopt;
    }
    return util::from_view_checked<std::uint32_t>(key_id);
  }

  std::optional<std::uint64_t> parse_host_session_id(std::string_view host_session_id) {
    const auto parsed = util::from_view_checked<std::uint64_t>(host_session_id);
    return parsed && *parsed != 0 ? parsed : std::nullopt;
  }

  bool detail::cancel_admission(
    std::optional<std::string_view> presented_host_session_id,
    std::uint64_t retained_host_session_id
  ) {
    const auto parsed = presented_host_session_id ? parse_host_session_id(*presented_host_session_id) : std::nullopt;
    return parsed && host_session_matches(retained_host_session_id, *parsed);
  }

  bool detail::resolve_resume_virtual_display_only(
    bool requested,
    bool retained,
    std::string_view retained_client_uuid,
    std::string_view requesting_client_uuid,
    PERM requesting_permissions
  ) {
    const bool retained_owner = !retained_client_uuid.empty() &&
                                retained_client_uuid == requesting_client_uuid;
    const bool launch_authorized = !!(requesting_permissions & PERM::launch);
    return retained_owner || launch_authorized ? requested : retained;
  }

  static std::shared_ptr<rtsp_stream::launch_session_t> make_launch_session(const args_t &args, const crypto::named_cert_t *named_cert_p) {
    static constexpr std::array required_args {
      "corever",
      "localAudioPlayMode",
      "mode",
      "rikey",
      "rikeyid",
      "sops",
      "surroundAudioInfo",
    };
    for (const auto name : required_args) {
      if (args.find(name) == args.end()) {
        BOOST_LOG(warning) << "Rejecting client missing required modern launch parameter ["sv << name << "] for ["sv << named_cert_p->name << ']';
        return nullptr;
      }
    }

    auto launch_session = std::make_shared<rtsp_stream::launch_session_t>();

    do {
      launch_session->id = ++session_id_counter;
    } while (launch_session->id == 0);

    const auto rikey = parse_remote_input_key(get_arg(args, "rikey"));
    const auto rikey_id = parse_remote_input_key_id(get_arg(args, "rikeyid"));
    if (!rikey || !rikey_id) {
      BOOST_LOG(warning) << "Rejecting invalid remote-input encryption parameters for client ["sv << named_cert_p->name << ']';
      return nullptr;
    }
    launch_session->gcm_key = *rikey;

    // Artemis always negotiates encrypted RTSP. Reject obsolete clients before any
    // display or application state is prepared.
    const auto core_version = parse_launch_int(launch_int_field::core_version, get_arg(args, "corever"));
    if (!core_version) {
      BOOST_LOG(warning) << "Rejecting client without encrypted RTSP support ["sv << named_cert_p->name << ']';
      return nullptr;
    }
    launch_session->rtsp_cipher = crypto::cipher::gcm_t {
      launch_session->gcm_key,
      false
    };

    // Generate the unique identifiers for this connection that we will send later during RTSP handshake
    unsigned char raw_payload[8];
    if (RAND_bytes(raw_payload, sizeof(raw_payload)) != 1 || RAND_bytes(reinterpret_cast<unsigned char *>(&launch_session->control_connect_data), sizeof(launch_session->control_connect_data)) != 1) {
      BOOST_LOG(error) << "Failed to generate secure streaming-session identifiers."sv;
      return nullptr;
    }
    launch_session->av_ping_payload = util::hex_vec(raw_payload);

    launch_session->iv.resize(16);
    uint32_t prepend_iv = util::endian::big<uint32_t>(*rikey_id);
    auto prepend_iv_p = (uint8_t *) &prepend_iv;
    std::copy(prepend_iv_p, prepend_iv_p + sizeof(prepend_iv), std::begin(launch_session->iv));

    const std::string mode = get_arg(args, "mode");
    BOOST_LOG(info) << "Display mode for client ["sv << named_cert_p->name << "] requested as ["sv << mode << ']';

    const auto parsed_mode = parse_launch_mode(mode);
    if (!parsed_mode) {
      BOOST_LOG(warning) << "Rejecting invalid display mode for client ["sv << named_cert_p->name << ']';
      return nullptr;
    }
    launch_session->display_context->width = parsed_mode->width;
    launch_session->display_context->height = parsed_mode->height;
    launch_session->display_context->fps = parsed_mode->fps;

    launch_session->display_context->device_name = named_cert_p->name.empty() ? "ApolloDisplay"s : named_cert_p->name;
    launch_session->display_context->unique_id = named_cert_p->uuid;
    launch_session->perm = named_cert_p->perm;
    const auto enable_sops = parse_launch_int(launch_int_field::binary_option, get_arg(args, "sops"));
    const auto host_audio = parse_launch_int(launch_int_field::binary_option, get_arg(args, "localAudioPlayMode"));
    const auto surround_info = parse_launch_int(launch_int_field::surround_info, get_arg(args, "surroundAudioInfo"));
    const auto enable_hdr = parse_launch_int(launch_int_field::binary_option, get_arg(args, "hdrMode", "0"));
    const auto stream_gamma = parse_launch_int(launch_int_field::stream_gamma, get_arg(args, "streamGamma", "0"));
    // These Apollo display extensions are optional for encrypted clients that implement the base
    // GameStream launch contract. Missing values use conservative defaults, while explicit invalid
    // values are still rejected.
    const auto display_options = parse_launch_display_options(
      find_arg(args, "virtualDisplay"),
      find_arg(args, "scaleFactor"),
      find_arg(args, "sbsMode"),
      find_arg(args, "virtualDisplayOnly")
    );
    if (!enable_sops || !host_audio || !surround_info || !enable_hdr || !display_options || !stream_gamma) {
      BOOST_LOG(warning) << "Rejecting invalid launch options for client ["sv << named_cert_p->name << ']';
      return nullptr;
    }
    const auto host_sbs_rejection =
      models::host_sbs_v2_source_resolution_rejection_reason(
        static_cast<std::uint32_t>(parsed_mode->width),
        static_cast<std::uint32_t>(parsed_mode->height)
      );
    if (display_options->sbs_mode == video::SBS_AI && !host_sbs_rejection.empty()) {
      const auto fitted = models::fit_host_sbs_v2_depth_tensor_shape(
        static_cast<std::uint32_t>(parsed_mode->width),
        static_cast<std::uint32_t>(parsed_mode->height)
      );
      BOOST_LOG(warning) << "Rejecting Host SBS V2 launch at "sv
                         << parsed_mode->width << 'x' << parsed_mode->height
                         << " for client ["sv << named_cert_p->name
                         << "]: "sv << host_sbs_rejection << "; fitted depth tensor "sv
                         << fitted.width << 'x' << fitted.height << '.';
      return nullptr;
    }

    launch_session->host_audio = *host_audio;
    launch_session->surround_info = *surround_info;
    launch_session->display_context->enable_hdr = *enable_hdr;
    launch_session->display_context->virtual_display = display_options->virtual_display;
    launch_session->display_context->virtual_display_only = display_options->virtual_display_only;
    launch_session->display_context->scale_factor = display_options->scale_factor;
    launch_session->display_context->sbs_mode = display_options->sbs_mode;
    launch_session->stream_gamma = static_cast<video::stream_gamma_mode_e>(*stream_gamma);

    if (!proc::calculate_render_size(launch_session->display_context->width, launch_session->display_context->height, launch_session->display_context->scale_factor)) {
      BOOST_LOG(warning) << "Rejecting a launch whose scaled display dimensions are invalid."sv;
      return nullptr;
    }

    return launch_session;
  }

  inline p_named_cert_t get_verified_cert(const req_https_t &request) {
    auto cached = std::static_pointer_cast<crypto::named_cert_t>(request->userp);
    auto current = refresh_authorized_client(cached);

    // Simple-Web-Server carries userp across keep-alive requests. Refresh it from the
    // authoritative client store so permission edits take effect on the next request and a
    // revoked certificate cannot retain access through an already-established TLS connection.
    request->userp = current;
    return current;
  }

  template<class T>
  void not_found(std::shared_ptr<typename SimpleWeb::ServerBase<T>::Response> response, std::shared_ptr<typename SimpleWeb::ServerBase<T>::Request> request) {
    print_req<T>(request);

    pt::ptree tree;
    tree.put("root.<xmlattr>.status_code", 404);

    std::ostringstream data;

    pt::write_xml(data, tree);
    response->write(SimpleWeb::StatusCode::client_error_not_found, data.str());
    response->close_connection_after_response = true;
  }

  template<class T>
  void serverinfo(std::shared_ptr<typename SimpleWeb::ServerBase<T>::Response> response, std::shared_ptr<typename SimpleWeb::ServerBase<T>::Request> request) {
    print_req<T>(request);

    int pair_status = 0;
    if constexpr (std::is_same_v<SunshineHTTPS, T>) {
      auto args = request->parse_query_string();
      auto clientID = args.find("uniqueid"s);

      if (clientID != std::end(args)) {
        pair_status = 1;
      }
    }

    auto local_endpoint = request->local_endpoint();

    pt::ptree tree;

    tree.put("root.<xmlattr>.status_code", 200);
    tree.put("root.hostname", config::nvhttp.sunshine_name);

    tree.put("root.appversion", VERSION);
    tree.put("root.GfeVersion", GFE_VERSION);
    tree.put("root.uniqueid", http::unique_id);
    tree.put("root.HttpsPort", net::map_port(PORT_HTTPS));
    tree.put("root.ExternalPort", net::map_port(PORT_HTTP));
    const auto codec_capabilities = video::nvenc_capabilities_snapshot();
    tree.put("root.MaxLumaPixelsHEVC", codec_capabilities.hevc ? "1869449984" : "0");

    // Only include the MAC address for requests sent from paired clients over HTTPS.
    // For HTTP requests, use a placeholder MAC address that Artemis knows to ignore.
    if constexpr (std::is_same_v<SunshineHTTPS, T>) {
      tree.put("root.mac", platf::get_mac_address(net::addr_to_normalized_string(local_endpoint.address())));

      auto named_cert_p = get_verified_cert(request);
      if (!named_cert_p) {
        tree.put("root.<xmlattr>.status_code", 401);
        tree.put("root.<xmlattr>.status_message", "The client is no longer authorized");
        std::ostringstream data;
        pt::write_xml(data, tree);
        response->write(data.str());
        response->close_connection_after_response = true;
        return;
      }
      tree.put("root.Permission", std::to_string((uint32_t) named_cert_p->perm));

      tree.put("root.VirtualDisplayCapable", true);
      tree.put("root.VirtualDisplayOnlySupported", 1);
      tree.put("root.GameProviderV1Supported", 1);
      tree.put("root.StreamGammaV1Supported", 1);
      if (!!(named_cert_p->perm & PERM::_all_actions)) {
        tree.put("root.VirtualDisplayDriverReady", proc::vDisplayDriverStatus == VDISPLAY::DRIVER_STATUS::OK);
      } else {
        tree.put("root.VirtualDisplayDriverReady", true);
      }
    } else {
      tree.put("root.mac", "00:00:00:00:00:00");
      tree.put("root.Permission", "0");
    }

    // Artemis tracks LAN IPv6 addresses separately from LocalIP, which is expected to
    // always be an IPv4 address. If we return that same IPv6 address here, it will clobber the
    // stored LAN IPv4 address. To avoid this, we need to return an IPv4 address in this field
    // when we get a request over IPv6.
    //
    // HACK: We should return the IPv4 address of local interface here, but we don't currently
    // have that implemented. For now, we will emulate the behavior of GFE+GS-IPv6-Forwarder,
    // which returns 127.0.0.1 as LocalIP for IPv6 connections. Artemis builds with IPv6
    // support know to ignore this bogus address.
    if (local_endpoint.address().is_v6() && !local_endpoint.address().to_v6().is_v4_mapped()) {
      tree.put("root.LocalIP", "127.0.0.1");
    } else {
      tree.put("root.LocalIP", net::addr_to_normalized_string(local_endpoint.address()));
    }

    uint32_t codec_mode_flags = SCM_H264;
    if (codec_capabilities.hevc) {
      codec_mode_flags |= SCM_HEVC;
    }
    if (codec_capabilities.hevc_hdr) {
      codec_mode_flags |= SCM_HEVC_MAIN10;
    }
    if (codec_capabilities.av1) {
      codec_mode_flags |= SCM_AV1_MAIN8;
    }
    if (codec_capabilities.av1_hdr) {
      codec_mode_flags |= SCM_AV1_MAIN10;
    }
    tree.put("root.ServerCodecModeSupport", codec_mode_flags);

    tree.put("root.PairStatus", pair_status);

    if constexpr (std::is_same_v<SunshineHTTPS, T>) {
      const auto process_status = proc::proc.get_status();
      const int current_appid = process_status.app_id;
      tree.put("root.currentgame", current_appid);
      tree.put("root.currentgameuuid", process_status.app_uuid);
      tree.put("root.hostsessionid", process_status.host_session_id);
      tree.put("root.state", current_appid > 0 ? "SUNSHINE_SERVER_BUSY" : "SUNSHINE_SERVER_FREE");
    } else {
      tree.put("root.currentgame", 0);
      tree.put("root.currentgameuuid", "");
      tree.put("root.hostsessionid", 0);
      tree.put("root.state", "SUNSHINE_SERVER_FREE");
    }

    std::ostringstream data;

    pt::write_xml(data, tree);
    response->write(data.str());
    response->close_connection_after_response = true;
  }

  void applist(resp_https_t response, req_https_t request) {
    print_req<SunshineHTTPS>(request);

    pt::ptree tree;

    auto g = util::fail_guard([&]() {
      std::ostringstream data;

      pt::write_xml(data, tree);
      response->write(data.str());
      response->close_connection_after_response = true;
    });

    auto &apps = tree.add_child("root", pt::ptree {});

    apps.put("<xmlattr>.status_code", 200);

    auto named_cert_p = get_verified_cert(request);
    if (!named_cert_p) {
      apps.put("<xmlattr>.status_code", 401);
      apps.put("<xmlattr>.status_message", "The client is no longer authorized");
      return;
    }
    if (!!(named_cert_p->perm & PERM::_all_actions)) {
      auto app_list = proc::proc.get_app_list();
      const auto codec_capabilities = video::nvenc_capabilities_snapshot();

      for (const auto &app : app_list) {
        pt::ptree app_node;

        app_node.put("IsHdrSupported"s, (codec_capabilities.hevc_hdr || codec_capabilities.av1_hdr) ? 1 : 0);
        app_node.put("AppTitle"s, app.name);
        app_node.put("UUID", app.uuid);
        app_node.put("IDX", app.idx);
        app_node.put("ID", app.id);

        apps.push_back(std::make_pair("App", std::move(app_node)));
      }
    } else {
      BOOST_LOG(debug) << "Permission ListApp denied for [" << named_cert_p->name << "] (" << (uint32_t) named_cert_p->perm << ")";

      pt::ptree app_node;

      app_node.put("IsHdrSupported"s, 0);
      app_node.put("AppTitle"s, "Permission Denied");
      app_node.put("UUID", "");
      app_node.put("IDX", "0");
      app_node.put("ID", "114514");

      apps.push_back(std::make_pair("App", std::move(app_node)));

      return;
    }
  }

  namespace {
    /** Own one HTTP launch/resume attempt until RTSP accepts its reservation. */
    class remote_launch_attempt_t {
    public:
      enum class kind_e {
        fresh,
        resume,
      };

      enum class failure_e {
        new_session_unavailable,
        process_start,
        encoder_probe,
        remote_display_reservation,
        retained_reconfiguration,
      };

      struct error_t {
        failure_e failure;
        int status;
      };

      remote_launch_attempt_t(kind_e kind, std::shared_ptr<rtsp_stream::launch_session_t> launch):
          kind_(kind),
          launch_(std::move(launch)),
          platform_(stream::session::guard_platform_launch()) {
      }

      ~remote_launch_attempt_t() {
        if (phase_ != phase_e::published) {
          if (kind_ == kind_e::fresh && phase_ == phase_e::prepared) {
            // execute() owns its own failed preparation. Only a successfully created process
            // needs this request's rollback if RTSP never accepted it.
#ifdef SUNSHINE_TESTS
            if (launch_native_hooks) {
              launch_native_hooks->terminate();
            } else
#endif
              proc::proc.terminate();
          } else if (kind_ == kind_e::resume && (phase_ == phase_e::resume_reconfiguring || phase_ == phase_e::prepared)) {
            // Keep the retained app and its coherent updated contract. Undo promotion without
            // renewing grace; the platform guard arranges retry if restoration is incomplete.
            platform_->restore_primary_display();
          }
          if (phase_ != phase_e::validating) {
            // Preserve the explicit admission boundary even if transport/socket references
            // retain an unpublished live-GPU reservation.
            launch_->release_unpublished_live_gpu();
          }
        }

        // The former resume guard was declared before platform_launch. Preserve its exact
        // unwind order: primary restoration and GPU release under the lifecycle lock, then
        // unlock, and only then release the temporary remote-display reservation.
        platform_.reset();
        if (phase_ != phase_e::published && remote_display_reserved_) {
#ifdef SUNSHINE_TESTS
          if (launch_native_hooks) {
            launch_native_hooks->remote_ended(launch_->id);
          } else
#endif
            ar_glasses::remote_virtual_display_ended(launch_->id);
        }
      }

      remote_launch_attempt_t(const remote_launch_attempt_t &) = delete;
      remote_launch_attempt_t &operator=(const remote_launch_attempt_t &) = delete;
      remote_launch_attempt_t(remote_launch_attempt_t &&) = delete;
      remote_launch_attempt_t &operator=(remote_launch_attempt_t &&) = delete;

      bool idle() const {
        return platform_->idle();
      }

      bool reserve_gpu() {
        if (!launch_->reserve_live_gpu()) {
          return false;
        }
        phase_ = phase_e::gpu_reserved;
        return true;
      }

      std::optional<error_t> prepare_new(proc::ctx_t &&app) {
        // Fresh launch may replace an idle retained app; resume never uses this policy.
        if (!platform_->prepare_new_session()) {
          return error_t {failure_e::new_session_unavailable, 503};
        }
        const auto error =
#ifdef SUNSHINE_TESTS
          launch_native_hooks ? launch_native_hooks->execute(std::move(app), launch_, true) :
#endif
                                proc::proc.execute(
                                  std::move(app),
                                  launch_->display_context,
                                  proc::launch_options_t {
                                    .launch_session_id = launch_->id,
                                    .host_audio = launch_->host_audio,
                                    .surround_info = launch_->surround_info,
                                  },
                                  true);
        if (error) {
          return error_t {failure_e::process_start, error};
        }
        phase_ = phase_e::prepared;
        return std::nullopt;
      }

      std::optional<error_t> prepare_resume(bool retained_virtual_display) {
        // Branch on the guarded process snapshot, not the client's requested display option.
        // Probe a physical source before mutating the retained contract, including before a
        // physical-to-virtual request can be refused by reconfigure_retained_session().
        if (!retained_virtual_display) {
          if (
#ifdef SUNSHINE_TESTS
            launch_native_hooks ? launch_native_hooks->probe_encoders() :
#endif
                                  video::probe_encoders()
          ) {
            return error_t {failure_e::encoder_probe, 503};
          }
        }
        if (retained_virtual_display) {
          if (
#ifdef SUNSHINE_TESTS
            launch_native_hooks ? !launch_native_hooks->remote_starting(
                                    launch_->id,
                                    config::stream.ping_timeout,
                                    true
                                  ) :
#endif
                                  !ar_glasses::remote_virtual_display_starting(
                                    launch_->id,
                                    config::stream.ping_timeout,
                                    true
                                  )
          ) {
            return error_t {failure_e::remote_display_reservation, 503};
          }
          remote_display_reserved_ = true;
        }
        // Arm primary rollback before entering the fallible retained-session operation. An
        // inner failure may already restore or terminate; the guarded retry policy stays intact.
        phase_ = phase_e::resume_reconfiguring;
        if (const auto error =
#ifdef SUNSHINE_TESTS
              launch_native_hooks ? launch_native_hooks->reconfigure(launch_) :
#endif
                                    proc::proc.reconfigure_retained_session(launch_->display_context, launch_->id)) {
          return error_t {failure_e::retained_reconfiguration, error};
        }
        if (retained_virtual_display) {
          // Reconfiguration can outlast the ordinary connect reservation. Start the fresh
          // client-connect window only after the retained display is ready for capture.
#ifdef SUNSHINE_TESTS
          if (launch_native_hooks) {
            launch_native_hooks->remote_awaiting(launch_->id, config::stream.ping_timeout);
          } else
#endif
            ar_glasses::remote_virtual_display_awaiting_client(
              launch_->id,
              config::stream.ping_timeout
            );
        }
        phase_ = phase_e::prepared;
        return std::nullopt;
      }

      bool publish() {
        if (phase_ != phase_e::prepared || !rtsp_stream::launch_session_raise(launch_)) {
          return false;
        }
        // RTSP now owns handshake timeout and lease transfer. Disarm before commit(), just
        // as the old endpoint guards did, so a commit exception cannot undo published work.
        phase_ = phase_e::published;
        platform_->commit();
        return true;
      }

    private:
      enum class phase_e {
        validating,
        gpu_reserved,
        resume_reconfiguring,
        prepared,
        published,
      };

      kind_e kind_;
      std::shared_ptr<rtsp_stream::launch_session_t> launch_;
      std::optional<stream::session::platform_launch_guard_t> platform_;
      phase_e phase_ = phase_e::validating;
      bool remote_display_reserved_ = false;
    };
  }  // namespace

  void launch(resp_https_t response, req_https_t request) {
    print_req<SunshineHTTPS>(request);

    pt::ptree tree;
    auto g = util::fail_guard([&]() {
      std::ostringstream data;

      if (tree.empty()) {
        BOOST_LOG(error) << EMPTY_PROPERTY_TREE_ERROR_MSG;
      }

      pt::write_xml(data, tree);
      response->write(data.str());
      response->close_connection_after_response = true;
    });
    std::lock_guard launch_lock(launch_request_mutex);

    auto args = request->parse_query_string();

    const bool appid_supplied = args.find("appid"s) != std::end(args);
    const bool appuuid_supplied = args.find("appuuid"s) != std::end(args);
    auto appid_str = get_arg(args, "appid", "");
    auto appuuid_str = get_arg(args, "appuuid", "");
    auto named_cert_p = get_verified_cert(request);
    if (!named_cert_p) {
      tree.put("root.<xmlattr>.status_code", 401);
      tree.put("root.<xmlattr>.status_message", "The client is no longer authorized");
      return;
    }

    std::optional<int> requested_appid;
    if (appid_supplied && !appid_str.empty()) {
      const auto parsed_appid = parse_launch_int(launch_int_field::app_id, appid_str);
      if (!parsed_appid) {
        tree.put("root.resume", 0);
        tree.put("root.<xmlattr>.status_code", 400);
        tree.put("root.<xmlattr>.status_message", "Invalid application ID");
        return;
      }
      if (*parsed_appid > 0) {
        requested_appid = *parsed_appid;
      }
    }
    BOOST_LOG(verbose) << "Launching app [" << appid_str << "] with UUID [" << appuuid_str << "]";
    // BOOST_LOG(verbose) << "QS: " << request->query_string;

    if ((!appid_supplied && !appuuid_supplied) || (!requested_appid && appuuid_str.empty())) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Missing a required launch parameter");

      return;
    }

    auto launch_session = make_launch_session(args, named_cert_p.get());
    if (!launch_session) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Invalid launch parameters");
      return;
    }

    auto selected_app = proc::proc.resolve_app(requested_appid, appuuid_str);
    if (selected_app.status == proc::app_lookup_status_e::not_found) {
      tree.put("root.<xmlattr>.status_code", 404);
      tree.put("root.<xmlattr>.status_message", "Cannot find requested application");
      tree.put("root.gamesession", 0);
      return;
    }
    if (selected_app.status == proc::app_lookup_status_e::identity_mismatch) {
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Application ID and UUID identify different applications");
      tree.put("root.gamesession", 0);
      return;
    }
    // From this point through RTSP publication, grace expiry and sole-session startup are
    // serialized with process/display mutation. Rejected paths release the guard without
    // changing the existing grace deadline.
    remote_launch_attempt_t attempt {remote_launch_attempt_t::kind_e::fresh, launch_session};

    if (!(named_cert_p->perm & PERM::launch)) {
      BOOST_LOG(debug) << "Permission LaunchApp denied for [" << named_cert_p->name << "] (" << (uint32_t) named_cert_p->perm << ")";
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 403);
      tree.put("root.<xmlattr>.status_message", "Permission denied");
      return;
    }

    if (!attempt.idle()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Another streaming session is already active");
      return;
    }

    // Reject an occupied HTTP-to-RTSP slot before mutating display or process state.
    // launch_request_mutex prevents another HTTP launch from filling the slot before publication.
    if (!rtsp_stream::launch_session_available()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Another streaming handshake is already pending");
      return;
    }

    // Reserve live GPU ownership before proc.execute() performs encoder probing or mutates the
    // virtual display. RTSP transfers this exact lease into its active slot, so offline admission
    // cannot race into the HTTP-to-RTSP handshake window.
    if (!attempt.reserve_gpu()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Offline SBS conversion is currently using the GPU");
      return;
    }

    // /launch is an explicit new session; /resume alone preserves an exact retained token.
    // Validate permissions, app, parameters and resource availability before ending the old app.
    if (const auto error = attempt.prepare_new(std::move(*selected_app.app))) {
      if (error->failure == remote_launch_attempt_t::failure_e::new_session_unavailable) {
        tree.put("root.resume", 0);
        tree.put("root.<xmlattr>.status_code", 503);
        tree.put("root.<xmlattr>.status_message", "Another streaming session is active or connecting");
      } else {
        tree.put("root.<xmlattr>.status_code", error->status);
        tree.put(
          "root.<xmlattr>.status_message",
          error->status == 503 ? "Failed to initialize video capture/encoding. Is a display connected and turned on?" : "Failed to start the specified application"
        );
        tree.put("root.gamesession", 0);
      }
      return;
    }

    if (!attempt.publish()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Another streaming handshake is already pending");
      return;
    }

    tree.put("root.<xmlattr>.status_code", 200);
    tree.put(
      "root.sessionUrl0",
      std::format(
        "{}{}:{}",
        "rtspenc://",
        net::addr_to_url_escaped_string(request->local_endpoint().address()),
        static_cast<int>(net::map_port(rtsp_stream::RTSP_SETUP_PORT))
      )
    );
    tree.put("root.gamesession", 1);
    tree.put("root.hostsessionid", proc::proc.get_host_session_id());
  }

  void resume(resp_https_t response, req_https_t request) {
    print_req<SunshineHTTPS>(request);

    pt::ptree tree;
    auto g = util::fail_guard([&]() {
      std::ostringstream data;

      if (tree.empty()) {
        BOOST_LOG(error) << EMPTY_PROPERTY_TREE_ERROR_MSG;
        tree.put("root.resume", 0);
        tree.put("root.<xmlattr>.status_code", 500);
        tree.put("root.<xmlattr>.status_message", "Internal server error");
      }

      pt::write_xml(data, tree);
      response->write(data.str());
      response->close_connection_after_response = true;
    });
    std::lock_guard launch_lock(launch_request_mutex);

    auto named_cert_p = get_verified_cert(request);
    if (!named_cert_p) {
      tree.put("root.<xmlattr>.status_code", 401);
      tree.put("root.<xmlattr>.status_message", "The client is no longer authorized");
      return;
    }
    if (!(named_cert_p->perm & PERM::_allow_view)) {
      BOOST_LOG(debug) << "Permission ViewApp denied for [" << named_cert_p->name << "] (" << (uint32_t) named_cert_p->perm << ")";

      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 403);
      tree.put("root.<xmlattr>.status_message", "Permission denied");

      return;
    }

    auto args = request->parse_query_string();
    const auto host_session_id_arg = find_arg(args, "hostSessionId");
    const auto presented_host_session_id = host_session_id_arg ? parse_host_session_id(*host_session_id_arg) : std::nullopt;
    if (!presented_host_session_id || !detail::host_session_matches(proc::proc.get_host_session_id(), *presented_host_session_id)) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 409);
      tree.put("root.<xmlattr>.status_message", "Missing or stale host session ID");
      return;
    }
    const bool appid_supplied = args.find("appid"s) != std::end(args);
    const auto appid_text = get_arg(args, "appid", "");
    const auto appuuid = get_arg(args, "appuuid", "");
    std::optional<int> requested_appid;
    if (appid_supplied && !appid_text.empty()) {
      const auto parsed_appid = parse_launch_int(launch_int_field::app_id, appid_text);
      if (!parsed_appid) {
        tree.put("root.resume", 0);
        tree.put("root.<xmlattr>.status_code", 400);
        tree.put("root.<xmlattr>.status_message", "Invalid application ID");
        return;
      }
      if (*parsed_appid > 0) {
        requested_appid = *parsed_appid;
      }
    }
    if (!requested_appid && appuuid.empty()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Missing retained application identity");
      return;
    }
    // Parse every client-controlled option before acquiring the lifecycle guard. Malformed
    // retries must not postpone an already pending grace expiry.
    auto launch_session = make_launch_session(args, named_cert_p.get());
    if (!launch_session) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Invalid resume parameters");
      return;
    }
    remote_launch_attempt_t attempt {remote_launch_attempt_t::kind_e::resume, launch_session};
    if (!attempt.idle()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Another streaming session is already active");
      return;
    }
    const auto process_status = proc::proc.get_status();
    auto current_appid = process_status.app_id;
    if (current_appid <= 0) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "No running app to resume");
      return;
    }
    if (!detail::host_session_matches(process_status.host_session_id, *presented_host_session_id)) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 409);
      tree.put("root.<xmlattr>.status_message", "The retained host session changed before resume");
      return;
    }
    if (!detail::app_identity_matches(
          requested_appid,
          appuuid,
          current_appid,
          process_status.app_uuid
        )) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 409);
      tree.put("root.<xmlattr>.status_message", "The requested application does not match the retained host session");
      return;
    }

    const bool requested_virtual_display_only = launch_session->display_context->virtual_display_only;
    launch_session->display_context->virtual_display_only = detail::resolve_resume_virtual_display_only(
      requested_virtual_display_only,
      process_status.virtual_display_only,
      process_status.client_uuid,
      named_cert_p->uuid,
      named_cert_p->perm
    );
    if (launch_session->display_context->virtual_display_only != requested_virtual_display_only) {
      BOOST_LOG(warning) << "Ignoring a virtual-display-only policy change from view-only client ["
                         << named_cert_p->name << "]; retaining the current session policy."sv;
    }

    if (!rtsp_stream::launch_session_available()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Another streaming handshake is already pending");
      return;
    }

    // Claim GPU ownership before the retained-session probe/reconfiguration below. The same lease
    // survives publication and is moved into RTSP's active slot without an admission gap.
    if (!attempt.reserve_gpu()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Offline SBS conversion is currently using the GPU");
      return;
    }
    if (const auto error = attempt.prepare_resume(process_status.virtual_display)) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", error->status);
      if (error->failure == remote_launch_attempt_t::failure_e::encoder_probe) {
        tree.put("root.<xmlattr>.status_message", "Failed to initialize video capture/encoding. Is a display connected and turned on?");
      } else if (error->failure == remote_launch_attempt_t::failure_e::remote_display_reservation) {
        tree.put("root.<xmlattr>.status_message", "Local AR display ownership could not be released");
      } else {
        tree.put("root.<xmlattr>.status_message", "Failed to reconfigure the retained streaming session");
      }
      return;
    }

    if (!attempt.publish()) {
      tree.put("root.resume", 0);
      tree.put("root.<xmlattr>.status_code", 503);
      tree.put("root.<xmlattr>.status_message", "Another streaming handshake is already pending");
      return;
    }

    tree.put("root.<xmlattr>.status_code", 200);
    tree.put(
      "root.sessionUrl0",
      std::format(
        "{}{}:{}",
        "rtspenc://",
        net::addr_to_url_escaped_string(request->local_endpoint().address()),
        static_cast<int>(net::map_port(rtsp_stream::RTSP_SETUP_PORT))
      )
    );
    tree.put("root.resume", 1);
    tree.put("root.hostsessionid", process_status.host_session_id);

    system_tray::update_tray_client_connected(named_cert_p->name);
  }

#ifdef SUNSHINE_TESTS
  void testing::invoke_launch_handler_for_test(
    bool resuming,
    response_t response,
    request_t request,
    launch_native_hooks_t &hooks
  ) {
    if (!hooks.execute || !hooks.terminate || !hooks.probe_encoders || !hooks.reconfigure || !hooks.remote_starting || !hooks.remote_awaiting || !hooks.remote_ended) {
      throw std::invalid_argument("The inert launch handler requires every native operation");
    }
    const auto *previous = std::exchange(launch_native_hooks, &hooks);
    auto restore = util::fail_guard([&]() {
      launch_native_hooks = previous;
    });
    if (resuming) {
      resume(std::move(response), std::move(request));
    } else {
      launch(std::move(response), std::move(request));
    }
  }
#endif

  void cancel(resp_https_t response, req_https_t request) {
    print_req<SunshineHTTPS>(request);

    pt::ptree tree;
    auto g = util::fail_guard([&]() {
      std::ostringstream data;

      if (tree.empty()) {
        BOOST_LOG(error) << EMPTY_PROPERTY_TREE_ERROR_MSG;
        tree.put("root.cancel", 0);
        tree.put("root.<xmlattr>.status_code", 500);
        tree.put("root.<xmlattr>.status_message", "Internal server error");
      }

      pt::write_xml(data, tree);
      response->write(data.str());
      response->close_connection_after_response = true;
    });
    auto named_cert_p = get_verified_cert(request);
    if (!named_cert_p) {
      tree.put("root.<xmlattr>.status_code", 401);
      tree.put("root.<xmlattr>.status_message", "The client is no longer authorized");
      return;
    }
    if (!(named_cert_p->perm & PERM::launch)) {
      BOOST_LOG(debug) << "Permission CancelApp denied for [" << named_cert_p->name << "] (" << (uint32_t) named_cert_p->perm << ")";

      tree.put("root.cancel", 0);
      tree.put("root.<xmlattr>.status_code", 403);
      tree.put("root.<xmlattr>.status_message", "Permission denied");

      return;
    }

    std::lock_guard launch_lock(launch_request_mutex);
    const auto args = request->parse_query_string();
    const auto process_status = proc::proc.get_status();
    const auto presented_host_session_id = find_arg(args, "hostSessionId");
    if (!detail::cancel_admission(presented_host_session_id, process_status.host_session_id)) {
      BOOST_LOG(warning) << "Rejecting cancel request from client [" << named_cert_p->name
                         << "]: hostSessionId is absent, malformed, or stale";
      tree.put("root.cancel", 0);
      tree.put("root.<xmlattr>.status_code", 409);
      tree.put("root.<xmlattr>.status_message", "Missing or stale host session ID");
      return;
    }
    terminate_active_session_locked();

    tree.put("root.cancel", 1);
    tree.put("root.<xmlattr>.status_code", 200);
  }

  void appasset(resp_https_t response, req_https_t request) {
    print_req<SunshineHTTPS>(request);

    auto fg = util::fail_guard([&]() {
      response->write(SimpleWeb::StatusCode::server_error_internal_server_error);
      response->close_connection_after_response = true;
    });

    auto named_cert_p = get_verified_cert(request);
    if (!named_cert_p) {
      fg.disable();
      response->write(SimpleWeb::StatusCode::client_error_unauthorized);
      response->close_connection_after_response = true;
      return;
    }

    if (!(named_cert_p->perm & PERM::_all_actions)) {
      BOOST_LOG(debug) << "Permission Get AppAsset denied for [" << named_cert_p->name << "] (" << (uint32_t) named_cert_p->perm << ")";

      fg.disable();
      response->write(SimpleWeb::StatusCode::client_error_unauthorized);
      response->close_connection_after_response = true;
      return;
    }

    auto args = request->parse_query_string();
    auto app_image = proc::proc.get_app_image(util::from_view(get_arg(args, "appid")));

    fg.disable();

    std::ifstream in(app_image, std::ios::binary);
    SimpleWeb::CaseInsensitiveMultimap headers;
    headers.emplace("Content-Type", "image/png");
    response->write(SimpleWeb::StatusCode::success_ok, in, headers);
    response->close_connection_after_response = true;
  }

  void getClipboard(resp_https_t response, req_https_t request) {
    print_req<SunshineHTTPS>(request);

    auto named_cert_p = get_verified_cert(request);
    if (!named_cert_p) {
      response->write(SimpleWeb::StatusCode::client_error_unauthorized);
      response->close_connection_after_response = true;
      return;
    }

    if (
      !(named_cert_p->perm & PERM::_allow_view) || !(named_cert_p->perm & PERM::clipboard_read)
    ) {
      BOOST_LOG(debug) << "Permission Read Clipboard denied for [" << named_cert_p->name << "] (" << (uint32_t) named_cert_p->perm << ")";

      response->write(SimpleWeb::StatusCode::client_error_unauthorized);
      response->close_connection_after_response = true;
      return;
    }

    auto args = request->parse_query_string();
    auto clipboard_type = get_arg(args, "type");
    if (clipboard_type != "text"sv) {
      BOOST_LOG(debug) << "Clipboard type [" << clipboard_type << "] is not supported!";

      response->write(SimpleWeb::StatusCode::client_error_bad_request);
      response->close_connection_after_response = true;
      return;
    }

    const auto connected_uuid = rtsp_stream::active_session_uuid();
    const bool found = connected_uuid && *connected_uuid == named_cert_p->uuid;

    if (!found) {
      BOOST_LOG(debug) << "Client [" << named_cert_p->name << "] trying to get clipboard is not connected to a stream";

      response->write(SimpleWeb::StatusCode::client_error_forbidden);
      response->close_connection_after_response = true;
      return;
    }

    std::string content = platf::get_clipboard();
    response->write(content);
    return;
  }

  void
    setClipboard(resp_https_t response, req_https_t request) {
    print_req<SunshineHTTPS>(request);

    auto named_cert_p = get_verified_cert(request);
    if (!named_cert_p) {
      response->write(SimpleWeb::StatusCode::client_error_unauthorized);
      response->close_connection_after_response = true;
      return;
    }

    if (
      !(named_cert_p->perm & PERM::_allow_view) || !(named_cert_p->perm & PERM::clipboard_set)
    ) {
      BOOST_LOG(debug) << "Permission Write Clipboard denied for [" << named_cert_p->name << "] (" << (uint32_t) named_cert_p->perm << ")";

      response->write(SimpleWeb::StatusCode::client_error_unauthorized);
      response->close_connection_after_response = true;
      return;
    }

    auto args = request->parse_query_string();
    auto clipboard_type = get_arg(args, "type");
    if (clipboard_type != "text"sv) {
      BOOST_LOG(debug) << "Clipboard type [" << clipboard_type << "] is not supported!";

      response->write(SimpleWeb::StatusCode::client_error_bad_request);
      response->close_connection_after_response = true;
      return;
    }

    const auto connected_uuid = rtsp_stream::active_session_uuid();
    const bool found = connected_uuid && *connected_uuid == named_cert_p->uuid;

    if (!found) {
      BOOST_LOG(debug) << "Client [" << named_cert_p->name << "] trying to set clipboard is not connected to a stream";

      response->write(SimpleWeb::StatusCode::client_error_forbidden);
      response->close_connection_after_response = true;
      return;
    }

    std::string content = request->content.string();

    bool success = platf::set_clipboard(content);

    if (!success) {
      BOOST_LOG(debug) << "Setting clipboard failed!";

      response->write(SimpleWeb::StatusCode::server_error_internal_server_error);
      response->close_connection_after_response = true;
      return;
    }

    response->write();
    return;
  }

  void start() {
    auto shutdown_event = mail::man->event<bool>(mail::shutdown);

    auto port_http = net::map_port(PORT_HTTP);
    auto port_https = net::map_port(PORT_HTTPS);
    auto address_family = net::af_from_enum_string(config::sunshine.address_family);

    bool clean_slate = config::sunshine.flags[config::flag::FRESH_STATE];

    if (!clean_slate) {
      load_state();
    }

    auto pkey = file_handler::read_file(config::nvhttp.pkey.c_str());
    auto cert = file_handler::read_file(config::nvhttp.cert.c_str());
    setup(pkey, cert);

    https_server_t https_server {config::nvhttp.cert, config::nvhttp.pkey};
    http_server_t http_server;

    // Verify certificates after establishing connection
    https_server.verify = [](req_https_t req, SSL *ssl) {
      crypto::x509_t x509 {
        SSL_get1_peer_certificate(ssl)
      };
      if (!x509) {
        BOOST_LOG(info) << "unknown -- denied"sv;
        return false;
      }

      bool verified = false;
      p_named_cert_t named_cert_p;

      auto fg = util::fail_guard([&]() {
        char subject_name[256];

        X509_NAME_oneline(X509_get_subject_name(x509.get()), subject_name, sizeof(subject_name));

        if (verified) {
          BOOST_LOG(debug) << subject_name << " -- "sv << "verified, device name: "sv << named_cert_p->name;
        } else {
          BOOST_LOG(debug) << subject_name << " -- "sv << "denied"sv;
        }
      });

      const char *err_str = verify_authorized_client(x509.get(), named_cert_p);
      if (err_str) {
        BOOST_LOG(warning) << "SSL Verification error :: "sv << err_str;
        return verified;
      }

      verified = true;
      req->userp = named_cert_p;

      return true;
    };

    https_server.on_verify_failed = [](resp_https_t resp, req_https_t req) {
      pt::ptree tree;
      auto g = util::fail_guard([&]() {
        std::ostringstream data;

        pt::write_xml(data, tree);
        resp->write(data.str());
        resp->close_connection_after_response = true;
      });

      tree.put("root.<xmlattr>.status_code"s, 401);
      tree.put("root.<xmlattr>.query"s, req->path);
      tree.put("root.<xmlattr>.status_message"s, "The client is not authorized. Certificate verification failed."s);
    };

    https_server.default_resource["GET"] = not_found<SunshineHTTPS>;
    https_server.resource["^/serverinfo$"]["GET"] = serverinfo<SunshineHTTPS>;
    https_server.resource["^/pair$"]["GET"] = pair<SunshineHTTPS>;
    https_server.resource["^/applist$"]["GET"] = applist;
    https_server.resource["^/appasset$"]["GET"] = appasset;
    https_server.resource["^/launch$"]["GET"] = [](auto resp, auto req) {
      launch(resp, req);
    };
    https_server.resource["^/resume$"]["GET"] = [](auto resp, auto req) {
      resume(resp, req);
    };
    https_server.resource["^/cancel$"]["GET"] = cancel;
    https_server.resource["^/actions/clipboard$"]["GET"] = getClipboard;
    https_server.resource["^/actions/clipboard$"]["POST"] = setClipboard;

    https_server.config.reuse_address = true;
    // The library's default accepts a request of any size into memory. Only paired clients
    // get past the TLS handshake here; their largest request is a clipboard text.
    https_server.config.max_request_streambuf_size = 8u * 1024u * 1024u;
    const auto bind_address = net::get_bind_address(address_family);
    if (!bind_address) {
      BOOST_LOG(fatal) << "GameStream servers refused invalid bind_address ["sv << config::sunshine.bind_address << ']';
      shutdown_event->raise(true);
      return;
    }
    https_server.config.address = *bind_address;
    https_server.config.port = port_https;

    http_server.default_resource["GET"] = not_found<SimpleWeb::HTTP>;
    http_server.resource["^/serverinfo$"]["GET"] = serverinfo<SimpleWeb::HTTP>;
    http_server.resource["^/pair$"]["GET"] = pair<SimpleWeb::HTTP>;

    http_server.config.reuse_address = true;
    // Anyone on the network reaches this server. It serves only GET requests whose largest
    // query carries a client certificate.
    http_server.config.max_request_streambuf_size = 64u * 1024u;
    http_server.config.address = *bind_address;
    http_server.config.port = port_http;

    auto accept_and_run = [&](auto *http_server) {
      try {
        http_server->start();
      } catch (boost::system::system_error &err) {
        // It's possible the exception gets thrown after calling http_server->stop() from a different thread
        if (shutdown_event->peek()) {
          return;
        }

        BOOST_LOG(fatal) << "Couldn't start http server on ports ["sv << port_https << ", "sv << port_https << "]: "sv << err.what();
        shutdown_event->raise(true);
        return;
      }
    };
    std::thread ssl {accept_and_run, &https_server};
    std::thread tcp {accept_and_run, &http_server};

    // Wait for any event
    shutdown_event->view();

    // Stop accepting work and join every server handler before clearing pairing state. Otherwise
    // an in-flight getservercert request can publish a session (and retain its Response) after the
    // clear, leaving it alive beyond server teardown.
    https_server.stop();
    http_server.stop();

    ssl.join();
    tcp.join();

    clear_pairing_state();
  }

  void
    erase_all_clients() {
    std::lock_guard launch_lock(launch_request_mutex);
    terminate_active_session_locked();
    revoke_all_clients();
  }

  void stop_session(stream::session_t &session, bool graceful) {
    if (graceful) {
      stream::session::graceful_stop(session);
    } else {
      stream::session::stop(session);
    }
  }

  bool find_and_stop_session(const std::string &uuid, bool graceful) {
    const auto session = rtsp_stream::find_session(uuid);
    if (!session) {
      return false;
    }
    stop_session(*session, graceful);
    return true;
  }

}  // namespace nvhttp
