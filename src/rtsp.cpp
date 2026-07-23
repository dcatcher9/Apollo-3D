/**
 * @file src/rtsp.cpp
 * @brief Definitions for RTSP streaming.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

extern "C" {
#include <moonlight-common-c/src/Limelight-internal.h>
#include <moonlight-common-c/src/Rtsp.h>
}

// standard includes
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// lib includes
#include <boost/asio.hpp>
#include <boost/bind.hpp>

// local includes
#include "config.h"
#include "globals.h"
#include "input.h"
#include "logging.h"
#include "network.h"
#include "rtsp.h"
#include "stream.h"
#include "sync.h"
#include "video.h"

namespace asio = boost::asio;

using asio::ip::tcp;
using asio::ip::udp;

using namespace std::literals;

namespace rtsp_stream {
  namespace detail {
    std::optional<int> parse_announce_int(announce_int_field field, std::string_view value) {
      const auto parsed = util::from_view_checked<int>(value);
      if (!parsed) {
        return std::nullopt;
      }

      const auto in_range = [&](int minimum, int maximum) {
        return *parsed >= minimum && *parsed <= maximum;
      };

      bool valid = false;
      switch (field) {
        case announce_int_field::audio_channels:
          valid = *parsed == 2 || *parsed == 6 || *parsed == 8;
          break;
        case announce_int_field::audio_channel_mask:
          valid = in_range(0, 65535);
          break;
        case announce_int_field::audio_packet_duration:
          valid = *parsed == 5 || *parsed == 10;
          break;
        case announce_int_field::audio_quality:
        case announce_int_field::binary_option:
          valid = in_range(0, 1);
          break;
        case announce_int_field::control_protocol:
          valid = *parsed == 13;
          break;
        case announce_int_field::feature_flags:
          valid = *parsed >= 0;
          break;
        case announce_int_field::audio_qos:
          valid = *parsed == 0 || *parsed == 4;
          break;
        case announce_int_field::video_qos:
          valid = *parsed == 0 || *parsed == 5;
          break;
        case announce_int_field::encryption_flags:
          valid = in_range(0, SS_ENC_VIDEO | SS_ENC_AUDIO | SS_ENC_CONTROL_V2);
          break;
        case announce_int_field::viewport_dimension:
          valid = in_range(1, 16384);
          break;
        case announce_int_field::max_fps:
          valid = in_range(1, 1000000);
          break;
        case announce_int_field::client_refresh_x100:
          valid = in_range(0, 100000);
          break;
        case announce_int_field::bitrate_kbps:
          valid = in_range(1, 1000000);
          break;
        case announce_int_field::configured_bitrate_kbps:
          valid = in_range(0, 1000000);
          break;
        case announce_int_field::slices_per_frame:
          valid = in_range(1, 255);
          break;
        case announce_int_field::reference_frames:
          valid = in_range(0, 16);
          break;
        case announce_int_field::encoder_csc_mode:
          valid = in_range(0, 5);
          break;
        case announce_int_field::video_format:
          valid = in_range(0, 2);
          break;
      }

      return valid ? parsed : std::nullopt;
    }

    int validated_client_refresh_x100(int announced_fps, int client_refresh_x100) {
      if (client_refresh_x100 <= 0 || client_refresh_x100 > 100000) {
        return 0;
      }

      double announced_hz {};
      if (announced_fps >= 1 && announced_fps <= 1000) {
        announced_hz = static_cast<double>(announced_fps);
      } else if (announced_fps > 4000 && announced_fps <= 1000000) {
        announced_hz = static_cast<double>(announced_fps) / 1000.0;
      } else {
        // Values in this gap are ambiguous: they could mean an implausibly high integral rate or
        // a low fractional rate encoded in millihertz. Do not let optional metadata reinterpret it.
        return 0;
      }

      const auto exact_hz = static_cast<double>(client_refresh_x100) / 100.0;
      const auto ratio = exact_hz / announced_hz;
      // Exact display metadata may refine a rounded request downward (60 -> 59.94), but must not
      // undo a deliberate client-side under-refresh cap such as 119 fps on a 120 Hz display.
      return ratio >= 0.99 && ratio <= 1.0 ? client_refresh_x100 : 0;
    }

    int calculate_warp_bitrate_factor(int announced_fps, int session_fps) {
      if (announced_fps <= 0 || announced_fps > 1000000 || session_fps <= 0 || session_fps > 1000000) {
        return 1;
      }

      const auto numerator = static_cast<std::int64_t>(announced_fps) * 1000 + session_fps / 2;
      const auto factor = numerator / session_fps;
      return factor >= 2 && factor <= 4 ? static_cast<int>(factor) : 1;
    }

    bool is_safe_encoder_bitrate(std::int64_t bitrate_kbps) {
      return bitrate_kbps > 0 && bitrate_kbps <= std::numeric_limits<int>::max() / 1000;
    }

    std::int64_t reserve_video_bitrate_for_fec(std::int64_t total_bitrate_kbps, int fec_percentage) {
      if (total_bitrate_kbps <= 0 || fec_percentage <= 0) {
        return total_bitrate_kbps;
      }

      // Video FEC parity is expressed as a percentage of data shards, so total wire payload is
      // data * (1 + F). Reserve data bitrate with B / (1 + F), not B * (1 - F).
      const auto bounded_fec_percentage = std::clamp(fec_percentage, 0, 255);
      return total_bitrate_kbps * 100 / (100 + bounded_fec_percentage);
    }

    std::int64_t calculate_video_bitrate_budget(
      std::int64_t total_bitrate_kbps,
      int fec_percentage,
      std::int64_t audio_bitrate_kbps
    ) {
      if (total_bitrate_kbps <= 0) {
        return total_bitrate_kbps;
      }

      // Audio and fixed transport/control overhead consume the total wire budget and do not
      // themselves receive video FEC. Deduct them first, then reserve parity from what remains.
      auto video_wire_budget_kbps = total_bitrate_kbps;
      video_wire_budget_kbps -= std::min(
        std::max<std::int64_t>(0, audio_bitrate_kbps),
        video_wire_budget_kbps / 5
      );
      video_wire_budget_kbps -= std::min<std::int64_t>(500, video_wire_budget_kbps / 10);
      return reserve_video_bitrate_for_fec(video_wire_budget_kbps, fec_percentage);
    }

    bool is_video_mode_supported(
      int video_format,
      int dynamic_range,
      bool hevc_sdr,
      bool hevc_hdr,
      bool av1_sdr,
      bool av1_hdr
    ) {
      switch (video_format) {
        case 0:
          return dynamic_range == 0;
        case 1:
          return dynamic_range == 0 ? hevc_sdr : dynamic_range == 1 && hevc_hdr;
        case 2:
          return dynamic_range == 0 ? av1_sdr : dynamic_range == 1 && av1_hdr;
        default:
          return false;
      }
    }

    int apply_packet_size_limit(int client_packet_size, int configured_limit) {
      if (!stream::is_valid_video_packet_size(client_packet_size) || configured_limit == 0 || !stream::is_valid_video_packet_size(configured_limit)) {
        return client_packet_size;
      }
      return std::min(client_packet_size, configured_limit);
    }

    std::optional<std::string_view> parse_setup_stream_type(std::string_view target) {
      const auto separator = target.find('=');
      if (separator == std::string_view::npos || separator + 1 >= target.size()) {
        return std::nullopt;
      }

      const auto begin = separator + 1;
      const auto slash = target.find('/', begin);
      const auto type = target.substr(begin, slash == std::string_view::npos ? slash : slash - begin);
      return type.empty() ? std::nullopt : std::optional<std::string_view> {type};
    }

    std::unordered_map<std::string_view, std::string_view> parse_announce_attributes(std::string_view payload) {
      std::unordered_map<std::string_view, std::string_view> args;

      std::size_t begin = 0;
      while (begin < payload.size()) {
        const auto end = payload.find_first_of("\r\n", begin);
        const auto line = payload.substr(begin, end == std::string_view::npos ? end : end - begin);

        if (line.starts_with("a="sv)) {
          const auto separator = line.find(':', 2);
          if (separator != std::string_view::npos && separator > 2) {
            const auto name = line.substr(2, separator - 2);
            auto value = line.substr(separator + 1);
            while (!value.empty() && value.back() == ' ') {
              value.remove_suffix(1);
            }
            args.emplace(name, value);
          }
        }

        if (end == std::string_view::npos) {
          break;
        }
        begin = payload.find_first_not_of("\r\n", end);
        if (begin == std::string_view::npos) {
          break;
        }
      }

      return args;
    }
  }  // namespace detail

  void free_msg(PRTSP_MESSAGE msg) {
    freeMessage(msg);

    delete msg;
  }

#pragma pack(push, 1)

  struct encrypted_rtsp_header_t {
    // We set the MSB in encrypted RTSP messages to allow format-agnostic
    // parsing code to be able to tell encrypted from plaintext messages.
    static constexpr std::uint32_t ENCRYPTED_MESSAGE_TYPE_BIT = 0x80000000;

    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }

    std::uint32_t payload_length() {
      return util::endian::big<std::uint32_t>(typeAndLength) & ~ENCRYPTED_MESSAGE_TYPE_BIT;
    }

    bool is_encrypted() {
      return !!(util::endian::big<std::uint32_t>(typeAndLength) & ENCRYPTED_MESSAGE_TYPE_BIT);
    }

    // This field is the length of the payload + ENCRYPTED_MESSAGE_TYPE_BIT in big-endian
    std::uint32_t typeAndLength;

    // This field is the number used to initialize the bottom 4 bytes of the AES IV in big-endian
    std::uint32_t sequenceNumber;

    // This field is the AES GCM authentication tag
    std::uint8_t tag[16];
  };

#pragma pack(pop)

  class rtsp_server_t;

  using msg_t = util::safe_ptr<RTSP_MESSAGE, free_msg>;
  using cmd_func_t = std::function<void(rtsp_server_t *server, tcp::socket &, launch_session_t &, msg_t &&)>;

  void print_msg(PRTSP_MESSAGE msg);
  void cmd_not_found(tcp::socket &sock, launch_session_t &, msg_t &&req);
  void respond(tcp::socket &sock, launch_session_t &session, POPTION_ITEM options, int statuscode, const char *status_msg, int seqn, const std::string_view &payload);

  class socket_t: public std::enable_shared_from_this<socket_t> {
  public:
    socket_t(boost::asio::io_context &io_context, std::function<void(tcp::socket &sock, launch_session_t &, msg_t &&)> &&handle_data_fn):
        handle_data_fn {std::move(handle_data_fn)},
        sock {io_context} {
    }

    /**
     * @brief Queue an asynchronous read to begin the next message.
     */
    void read() {
      if (begin == std::end(msg_buf) || begin + sizeof(encrypted_rtsp_header_t) >= std::end(msg_buf)) {
        BOOST_LOG(error) << "RTSP: read(): Exceeded maximum rtsp packet size: "sv << msg_buf.size();

        respond(sock, *session, nullptr, 400, "BAD REQUEST", 0, {});

        boost::system::error_code ec;
        sock.close(ec);

        return;
      }

      if (!session->rtsp_cipher) {
        BOOST_LOG(error) << "RTSP: refusing a launch without encrypted RTSP"sv;
        boost::system::error_code ec;
        sock.close(ec);
        return;
      }

      boost::asio::async_read(sock, boost::asio::buffer(begin, sizeof(encrypted_rtsp_header_t)), boost::bind(&socket_t::handle_read_encrypted_header, shared_from_this(), boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
    }

    /**
     * @brief Handle the initial read of the header of an encrypted message.
     * @param socket The socket the message was received on.
     * @param ec The error code of the read operation.
     * @param bytes The number of bytes read.
     */
    static void handle_read_encrypted_header(std::shared_ptr<socket_t> &socket, const boost::system::error_code &ec, std::size_t bytes) {
      BOOST_LOG(debug) << "handle_read_encrypted_header(): Handle read of size: "sv << bytes << " bytes"sv;

      auto sock_close = util::fail_guard([&socket]() {
        boost::system::error_code ec;
        socket->sock.close(ec);

        if (ec) {
          BOOST_LOG(error) << "RTSP: handle_read_encrypted_header(): Couldn't close tcp socket: "sv << ec.message();
        }
      });

      if (ec || bytes < sizeof(encrypted_rtsp_header_t)) {
        BOOST_LOG(error) << "RTSP: handle_read_encrypted_header(): Couldn't read from tcp socket: "sv << ec.message();

        respond(socket->sock, *socket->session, nullptr, 400, "BAD REQUEST", 0, {});
        return;
      }

      auto header = (encrypted_rtsp_header_t *) socket->begin;
      if (!header->is_encrypted()) {
        BOOST_LOG(error) << "RTSP: handle_read_encrypted_header(): Rejecting unencrypted RTSP message"sv;

        respond(socket->sock, *socket->session, nullptr, 400, "BAD REQUEST", 0, {});
        return;
      }

      auto payload_length = header->payload_length();

      // Check if we have enough space to read this message
      const auto remaining_capacity = static_cast<std::size_t>(std::end(socket->msg_buf) - socket->begin);
      if (payload_length >= remaining_capacity - sizeof(*header)) {
        BOOST_LOG(error) << "RTSP: handle_read_encrypted_header(): Exceeded maximum rtsp packet size: "sv << socket->msg_buf.size();

        respond(socket->sock, *socket->session, nullptr, 400, "BAD REQUEST", 0, {});
        return;
      }

      sock_close.disable();

      // Read the remainder of the header and full encrypted payload
      boost::asio::async_read(socket->sock, boost::asio::buffer(socket->begin + bytes, payload_length), boost::bind(&socket_t::handle_read_encrypted_message, socket->shared_from_this(), boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
    }

    /**
     * @brief Handle the final read of the content of an encrypted message.
     * @param socket The socket the message was received on.
     * @param ec The error code of the read operation.
     * @param bytes The number of bytes read.
     */
    static void handle_read_encrypted_message(std::shared_ptr<socket_t> &socket, const boost::system::error_code &ec, std::size_t bytes) {
      BOOST_LOG(debug) << "handle_read_encrypted(): Handle read of size: "sv << bytes << " bytes"sv;

      auto sock_close = util::fail_guard([&socket]() {
        boost::system::error_code ec;
        socket->sock.close(ec);

        if (ec) {
          BOOST_LOG(error) << "RTSP: handle_read_encrypted_message(): Couldn't close tcp socket: "sv << ec.message();
        }
      });

      auto header = (encrypted_rtsp_header_t *) socket->begin;
      auto payload_length = header->payload_length();
      auto seq = util::endian::big<std::uint32_t>(header->sequenceNumber);

      if (ec || bytes < payload_length) {
        BOOST_LOG(error) << "RTSP: handle_read_encrypted(): Couldn't read from tcp socket: "sv << ec.message();

        respond(socket->sock, *socket->session, nullptr, 400, "BAD REQUEST", 0, {});
        return;
      }

      // We use the deterministic IV construction algorithm specified in NIST SP 800-38D
      // Section 8.2.1. The sequence number is our "invocation" field and the 'RC' in the
      // high bytes is the "fixed" field. Because each client provides their own unique
      // key, our values in the fixed field need only uniquely identify each independent
      // use of the client's key with AES-GCM in our code.
      //
      // The sequence number is 32 bits long which allows for 2^32 RTSP messages to be
      // received from each client before the IV repeats.
      crypto::aes_t iv(12);
      std::copy_n((uint8_t *) &seq, sizeof(seq), std::begin(iv));
      iv[10] = 'C';  // Client originated
      iv[11] = 'R';  // RTSP

      std::vector<uint8_t> plaintext;
      if (socket->session->rtsp_cipher->decrypt(std::string_view {(const char *) header->tag, sizeof(header->tag) + bytes}, plaintext, &iv)) {
        BOOST_LOG(error) << "Failed to verify RTSP message tag"sv;

        respond(socket->sock, *socket->session, nullptr, 400, "BAD REQUEST", 0, {});
        return;
      }

      msg_t req {new msg_t::element_type {}};
      if (auto status = parseRtspMessage(req.get(), (char *) plaintext.data(), plaintext.size())) {
        BOOST_LOG(error) << "Malformed RTSP message: ["sv << status << ']';

        respond(socket->sock, *socket->session, nullptr, 400, "BAD REQUEST", 0, {});
        return;
      }

      sock_close.disable();

      print_msg(req.get());

      socket->handle_data(std::move(req));
    }

    void handle_data(msg_t &&req) {
      handle_data_fn(sock, *session, std::move(req));
    }

    std::function<void(tcp::socket &sock, launch_session_t &, msg_t &&)> handle_data_fn;

    tcp::socket sock;

    std::array<char, 2048> msg_buf;

    char *begin = msg_buf.data();

    std::shared_ptr<launch_session_t> session;
  };

  class rtsp_server_t {
    struct client_policy_t {
      std::uint64_t generation;
      std::string name;
      crypto::PERM permissions;
      bool revoked;
    };

  public:
    ~rtsp_server_t() {
      clear();
    }

    int bind(net::af_e af, std::uint16_t port, boost::system::error_code &ec) {
      const auto bind_address_string = net::get_bind_address(af);
      if (!bind_address_string) {
        BOOST_LOG(error) << "RTSP refused invalid bind_address ["sv << config::sunshine.bind_address << ']';
        ec = boost::asio::error::invalid_argument;
        return -1;
      }
      const auto bind_address = boost::asio::ip::make_address(*bind_address_string, ec);
      if (ec) {
        BOOST_LOG(error) << "Invalid RTSP bind address ["sv << *bind_address_string << "]: "sv << ec.message();
        return -1;
      }

      acceptor.open(bind_address.is_v4() ? tcp::v4() : tcp::v6(), ec);
      if (ec) {
        return -1;
      }

      acceptor.set_option(boost::asio::socket_base::reuse_address {true});

      acceptor.bind(tcp::endpoint(bind_address, port), ec);
      if (ec) {
        return -1;
      }

      acceptor.listen(4096, ec);
      if (ec) {
        return -1;
      }

      next_socket = std::make_shared<socket_t>(io_context, [this](tcp::socket &sock, launch_session_t &session, msg_t &&msg) {
        handle_msg(sock, session, std::move(msg));
      });

      acceptor.async_accept(next_socket->sock, [this](const auto &ec) {
        handle_accept(ec);
      });

      return 0;
    }

    void handle_msg(tcp::socket &sock, launch_session_t &session, msg_t &&req) {
      if (session.reservation() == launch_reservation_state_e::revoked) {
        BOOST_LOG(debug) << "Rejecting RTSP request for a revoked launch reservation."sv;
        return;
      }

      auto func = _map_cmd_cb.find(req->message.request.command);
      if (func != std::end(_map_cmd_cb)) {
        func->second(this, sock, session, std::move(req));
      } else {
        cmd_not_found(sock, session, std::move(req));
      }

      boost::system::error_code ec;
      sock.shutdown(boost::asio::socket_base::shutdown_type::shutdown_both, ec);
    }

    void handle_accept(const boost::system::error_code &ec) {
      if (ec) {
        BOOST_LOG(error) << "Couldn't accept incoming connections: "sv << ec.message();

        // Stop server
        clear();
        return;
      }

      auto socket = std::move(next_socket);

      auto launch_session {launch_event.view(0s)};
      if (launch_session && launch_session->reservation() != launch_reservation_state_e::revoked) {
        // Associate the current RTSP session with this socket and start reading
        socket->session = launch_session;
        socket->read();
      } else {
        // This can happen due to normal things like port scanning, so let's not make these visible by default
        BOOST_LOG(debug) << "No pending session for incoming RTSP connection"sv;

        // If there is no session pending, close the connection immediately
        boost::system::error_code ec;
        socket->sock.close(ec);
      }

      // Queue another asynchronous accept for the next incoming connection
      next_socket = std::make_shared<socket_t>(io_context, [this](tcp::socket &sock, launch_session_t &session, msg_t &&msg) {
        handle_msg(sock, session, std::move(msg));
      });
      acceptor.async_accept(next_socket->sock, [this](const auto &ec) {
        handle_accept(ec);
      });
    }

    void map(const std::string_view &type, cmd_func_t cb) {
      _map_cmd_cb.emplace(type, std::move(cb));
    }

    /**
     * @brief Launch a new streaming session.
     * @note If the client does not begin streaming within the ping_timeout,
     *       the session will be discarded.
     * @param launch_session Streaming session information.
     */
    bool session_raise(std::shared_ptr<launch_session_t> launch_session) {
      std::lock_guard lock(_launch_mutex);
      // Reservation and publication are one operation. A caller must never return a successful
      // launch response for a session that was silently discarded behind an existing handshake.
      if (!launch_session || !launch_session->rtsp_cipher || launch_session->av_ping_payload.size() != sizeof(SS_PING::payload) || _claimed_launch_session || launch_session->reservation() != launch_reservation_state_e::pending) {
        return false;
      }
      const auto launch_session_id = launch_session->id;
      if (!launch_event.try_raise(std::move(launch_session))) {
        return false;
      }

      // Arm the timer to expire this launch session if the client times out
      raised_timer.expires_after(config::stream.ping_timeout);
      raised_timer.async_wait([this, launch_session_id](const boost::system::error_code &ec) {
        if (!ec) {
          std::lock_guard lock(_launch_mutex);
          auto pending = launch_event.view(0s);
          if (pending && pending->id == launch_session_id) {
            auto discarded = launch_event.pop(0s);
            discarded->revoke_reservation();
            BOOST_LOG(debug) << "Event timeout: "sv << discarded->unique_id;
          }
        }
      });
      return true;
    }

    /**
     * @brief Clear state for the oldest launch session.
     * @param launch_session_id The ID of the session to clear.
     */
    void session_clear(uint32_t launch_session_id) {
      std::lock_guard lock(_launch_mutex);
      // We currently only support a single pending RTSP session,
      // so the ID should always match the one for that session.
      auto launch_session = launch_event.view(0s);
      if (launch_session) {
        if (launch_session->id != launch_session_id) {
          BOOST_LOG(error) << "Attempted to clear unexpected session: "sv << launch_session_id << " vs "sv << launch_session->id;
        } else {
          raised_timer.cancel();
          auto cleared = launch_event.pop();
          // A normal control connection may clear an already claimed launch. Preserve that state
          // for subsequent PLAY, but ensure an unclaimed accepted socket cannot start later.
          cleared->revoke_pending_reservation();
        }
      }
    }

    void clear_pending_launch_session() {
      std::lock_guard lock(_launch_mutex);
      raised_timer.cancel();
      if (launch_event.view(0s)) {
        auto cleared = launch_event.pop(0s);
        cleared->revoke_reservation();
      }
      if (_claimed_launch_session) {
        _claimed_launch_session->revoke_reservation();
      }
    }

    bool launch_session_available() {
      std::lock_guard lock(_launch_mutex);
      return !launch_event.peek() && !_claimed_launch_session;
    }

    bool claim_launch_session(launch_session_t &launch_session) {
      std::lock_guard lock(_launch_mutex);
      auto pending = launch_event.view(0s);
      if (!pending || pending.get() != &launch_session || _claimed_launch_session || !launch_session.try_claim_reservation()) {
        return false;
      }
      _claimed_launch_session = std::move(pending);
      return true;
    }

    void finish_launch_session(launch_session_t &launch_session, bool started) {
      std::lock_guard lock(_launch_mutex);
      if (_claimed_launch_session.get() != &launch_session) {
        return;
      }

      if (!started) {
        launch_session.revoke_reservation();
        auto pending = launch_event.view(0s);
        if (pending && pending.get() == &launch_session) {
          raised_timer.cancel();
          launch_event.pop(0s);
        }
      }
      _claimed_launch_session.reset();
    }

    bool has_active_session() {
      auto lg = _active_session.lock();
      return static_cast<bool>(*_active_session);
    }

    safe::event_t<std::shared_ptr<launch_session_t>> launch_event;

    /**
     * @brief Clear the active stream.
     * @param force If true, clear the stream unconditionally. Otherwise, clear it only after stop.
     * @examples
     * clear(false);
     * @examples_end
     */
    void clear(bool force = true) {
      auto lg = _active_session.lock();
      auto &session = *_active_session;
      if (session && (force || stream::session::state(*session) == stream::session::state_e::STOPPING)) {
        stream::session::stop(*session);
        stream::session::join(*session);
        session.reset();
      }
    }

#ifdef SUNSHINE_TESTS
    /** Test-only active-slot mutation used by authorization concurrency tests. */
    void remove(const std::shared_ptr<stream::session_t> &session) {
      auto lg = _active_session.lock();
      if (*_active_session == session) {
        _active_session->reset();
      }
    }

    bool insert(const std::shared_ptr<stream::session_t> &session) {
      // The policy lock serializes authorization publication with insertion. A pending RTSP
      // launch therefore cannot slip through after its client was revoked or keep stale rights.
      std::lock_guard policy_lock(_client_policy_mutex);
      auto session_lock = _active_session.lock();
      if (*_active_session) {
        return false;
      }
      if (!apply_current_policy_locked(*session)) {
        return false;
      }

      *_active_session = session;
      BOOST_LOG(info) << "New streaming session started"sv;
      return true;
    }
#endif

    /**
     * @brief Starts and registers a session atomically with authorization publication.
     * @return 0 on success, 403 when authorization was revoked, 409 when another stream is active,
     *         454 when the launch reservation was revoked, or 500 when startup failed.
     */
    int start_and_insert(
      const std::shared_ptr<stream::session_t> &session,
      launch_session_t &launch_session
    ) {
      // Keep the policy lock through startup. Otherwise a revocation can observe the registered
      // session while it is still STOPPED (where graceful_stop is intentionally a no-op), after
      // which the RTSP thread could incorrectly transition it to RUNNING.
      std::lock_guard policy_lock(_client_policy_mutex);
      // Explicit teardown revokes the shared launch object before waiting on the active-slot lock.
      // Therefore either teardown wins here, or it waits until the started session is inserted
      // and then removes it before returning.
      auto session_lock = _active_session.lock();
      if (launch_session.reservation() != launch_reservation_state_e::claimed) {
        finish_launch_session(launch_session, false);
        return 454;
      }
      if (*_active_session) {
        finish_launch_session(launch_session, false);
        return 409;
      }
      if (!apply_current_policy_locked(*session)) {
        finish_launch_session(launch_session, false);
        return 403;
      }
      if (stream::session::start(*session)) {
        finish_launch_session(launch_session, false);
        return 500;
      }
      if (launch_session.reservation() == launch_reservation_state_e::revoked) {
        stream::session::stop(*session);
        stream::session::join(*session);
        finish_launch_session(launch_session, false);
        return 454;
      }

      *_active_session = session;
      BOOST_LOG(info) << "New streaming session started"sv;
      finish_launch_session(launch_session, true);
      return 0;
    }

  private:
    bool apply_current_policy_locked(stream::session_t &session) {
      const auto policy = _client_policies.find(stream::session::uuid(session));
      if (policy == _client_policies.end()) {
        return static_cast<bool>(stream::session::permissions(session) & crypto::PERM::_allow_view);
      }
      if (policy->second.revoked || !(policy->second.permissions & crypto::PERM::_allow_view)) {
        return false;
      }
      stream::session::update_client_policy(
        session,
        policy->second.generation,
        policy->second.name,
        policy->second.permissions,
        false
      );
      return true;
    }

  public:
    /**
     * @brief Runs an iteration of the RTSP server loop
     */
    void iterate() {
      // If we have a session, we will return to the server loop every
      // 500ms to allow session cleanup to happen.
      if (has_active_session()) {
        io_context.run_one_for(500ms);
      } else {
        io_context.run_one();
      }
    }

    /**
     * @brief Stop the RTSP server.
     */
    void stop() {
      acceptor.close();
      io_context.stop();
      clear();
    }

    std::shared_ptr<stream::session_t> find_session(std::string_view uuid) {
      auto lg = _active_session.lock();
      const auto &session = *_active_session;
      return session && stream::session::uuid_match(*session, uuid) ? session : nullptr;
    }

    client_policy_publication_t stage_client_policy(
      std::string_view uuid,
      std::uint64_t generation,
      std::string name,
      crypto::PERM permissions,
      bool revoked
    ) {
      client_policy_publication_t publication;
      std::shared_ptr<stream::session_t> session;
      client_policy_t published_policy {
        .generation = generation,
        .name = std::move(name),
        .permissions = permissions,
        .revoked = revoked,
      };
      {
        // All publication and insertion paths take this lock first, then the active-slot lock.
        // Session mutation and shutdown happen only after both locks are released.
        std::lock_guard policy_lock(_client_policy_mutex);
        const auto current = _client_policies.find(std::string(uuid));
        if (current != _client_policies.end() && generation <= current->second.generation) {
          return publication;
        }

        _client_policies.insert_or_assign(
          std::string(uuid),
          published_policy
        );

        auto session_lock = _active_session.lock();
        if (*_active_session && stream::session::uuid_match(**_active_session, uuid)) {
          session = *_active_session;
        }
      }

      if (session) {
        const auto result = stream::session::update_client_policy(
          *session,
          published_policy.generation,
          published_policy.name,
          published_policy.permissions,
          published_policy.revoked
        );
        if (result == stream::session::client_policy_result_e::disconnect) {
          publication.stop = client_policy_stop_t {
            .session = std::move(session),
            .generation = published_policy.generation,
          };
        }
      }
      publication.accepted = true;
      return publication;
    }

    std::optional<std::string> active_session_uuid() {
      auto lg = _active_session.lock();
      return *_active_session ? std::optional {stream::session::uuid(**_active_session)} : std::nullopt;
    }

  private:
    std::unordered_map<std::string_view, cmd_func_t> _map_cmd_cb;

    sync_util::sync_t<std::shared_ptr<stream::session_t>> _active_session;
    std::mutex _client_policy_mutex;
    std::mutex _launch_mutex;
    std::shared_ptr<launch_session_t> _claimed_launch_session;
    std::unordered_map<std::string, client_policy_t> _client_policies;

    boost::asio::io_context io_context;
    tcp::acceptor acceptor {io_context};
    boost::asio::steady_timer raised_timer {io_context};

    std::shared_ptr<socket_t> next_socket;
  };

  rtsp_server_t server {};

  bool launch_session_raise(std::shared_ptr<launch_session_t> launch_session) {
    return server.session_raise(std::move(launch_session));
  }

  bool launch_session_available() {
    return server.launch_session_available();
  }

  void launch_session_clear(uint32_t launch_session_id) {
    server.session_clear(launch_session_id);
  }

  void clear_pending_launch_session() {
    server.clear_pending_launch_session();
  }

  std::shared_ptr<stream::session_t> find_session(std::string_view uuid) {
    return server.find_session(uuid);
  }

  client_policy_publication_t stage_client_policy(
    std::string_view uuid,
    std::uint64_t generation,
    std::string name,
    crypto::PERM permissions,
    bool revoked
  ) {
    return server.stage_client_policy(uuid, generation, std::move(name), permissions, revoked);
  }

  void complete_client_policy(client_policy_publication_t publication, bool graceful) {
    if (publication.stop) {
      stream::session::stop_if_client_policy_current(
        *publication.stop->session,
        publication.stop->generation,
        graceful
      );
    }
  }

  bool publish_client_policy(
    std::string_view uuid,
    std::uint64_t generation,
    std::string name,
    crypto::PERM permissions,
    bool revoked
  ) {
    auto publication = stage_client_policy(uuid, generation, std::move(name), permissions, revoked);
    const bool accepted = publication.accepted;
    complete_client_policy(std::move(publication));
    return accepted;
  }

  std::optional<std::string> active_session_uuid() {
    return server.active_session_uuid();
  }

#ifdef SUNSHINE_TESTS
  bool insert_session_for_test(const std::shared_ptr<stream::session_t> &session) {
    return server.insert(session);
  }

  void remove_session_for_test(const std::shared_ptr<stream::session_t> &session) {
    server.remove(session);
  }

  bool claim_launch_session_for_test(launch_session_t &launch_session) {
    return server.claim_launch_session(launch_session);
  }

  void finish_launch_session_for_test(launch_session_t &launch_session, bool started) {
    server.finish_launch_session(launch_session, started);
  }
#endif

  void terminate_session() {
    server.clear_pending_launch_session();
    server.clear(true);
  }

  int send(tcp::socket &sock, const std::string_view &sv) {
    std::size_t bytes_send = 0;

    while (bytes_send != sv.size()) {
      boost::system::error_code ec;
      bytes_send += sock.send(boost::asio::buffer(sv.substr(bytes_send)), 0, ec);

      if (ec) {
        BOOST_LOG(error) << "RTSP: Couldn't send data over tcp socket: "sv << ec.message();
        return -1;
      }
    }

    return 0;
  }

  void respond(tcp::socket &sock, launch_session_t &session, msg_t &resp) {
    auto payload = std::make_pair(resp->payload, resp->payloadLength);

    // Restore response message for proper destruction
    auto lg = util::fail_guard([&]() {
      resp->payload = payload.first;
      resp->payloadLength = payload.second;
    });

    resp->payload = nullptr;
    resp->payloadLength = 0;

    int serialized_len;
    util::c_ptr<char> raw_resp {serializeRtspMessage(resp.get(), &serialized_len)};
    BOOST_LOG(debug)
      << "---Begin Response---"sv << std::endl
      << std::string_view {raw_resp.get(), (std::size_t) serialized_len} << std::endl
      << std::string_view {payload.first, (std::size_t) payload.second} << std::endl
      << "---End Response---"sv << std::endl;

    if (!session.rtsp_cipher) {
      BOOST_LOG(error) << "RTSP: refusing to send an unencrypted response"sv;
      return;
    }

    // We use the deterministic IV construction algorithm specified in NIST SP 800-38D
    // Section 8.2.1. The sequence number is our "invocation" field and the 'RH' in the
    // high bytes is the "fixed" field. Because each client provides their own unique
    // key, our values in the fixed field need only uniquely identify each independent
    // use of the client's key with AES-GCM in our code.
    //
    // The sequence number is 32 bits long which allows for 2^32 RTSP messages to be
    // sent to each client before the IV repeats.
    crypto::aes_t iv(12);
    session.rtsp_iv_counter++;
    std::copy_n((uint8_t *) &session.rtsp_iv_counter, sizeof(session.rtsp_iv_counter), std::begin(iv));
    iv[10] = 'H';  // Host originated
    iv[11] = 'R';  // RTSP

    // Allocate the message with an empty header and reserved space for the payload
    auto payload_length = serialized_len + payload.second;
    std::vector<uint8_t> message(sizeof(encrypted_rtsp_header_t));
    message.reserve(message.size() + payload_length);

    // Copy the complete plaintext into the message
    std::copy_n(raw_resp.get(), serialized_len, std::back_inserter(message));
    std::copy_n(payload.first, payload.second, std::back_inserter(message));

    // Initialize the message header
    auto header = (encrypted_rtsp_header_t *) message.data();
    header->typeAndLength = util::endian::big<std::uint32_t>(encrypted_rtsp_header_t::ENCRYPTED_MESSAGE_TYPE_BIT + payload_length);
    header->sequenceNumber = util::endian::big<std::uint32_t>(session.rtsp_iv_counter);

    // Encrypt the RTSP message in place
    session.rtsp_cipher->encrypt(std::string_view {(const char *) header->payload(), (std::size_t) payload_length}, header->tag, &iv);

    // Send the full encrypted message
    send(sock, std::string_view {(char *) message.data(), message.size()});
  }

  void respond(tcp::socket &sock, launch_session_t &session, POPTION_ITEM options, int statuscode, const char *status_msg, int seqn, const std::string_view &payload) {
    msg_t resp {new msg_t::element_type};
    createRtspResponse(resp.get(), nullptr, 0, const_cast<char *>("RTSP/1.0"), statuscode, const_cast<char *>(status_msg), seqn, options, const_cast<char *>(payload.data()), (int) payload.size());

    respond(sock, session, resp);
  }

  void cmd_not_found(tcp::socket &sock, launch_session_t &session, msg_t &&req) {
    respond(sock, session, nullptr, 404, "NOT FOUND", req->sequenceNumber, {});
  }

  void cmd_option(rtsp_server_t *server, tcp::socket &sock, launch_session_t &session, msg_t &&req) {
    OPTION_ITEM option {};

    // I know these string literals will not be modified
    option.option = const_cast<char *>("CSeq");

    auto seqn_str = std::to_string(req->sequenceNumber);
    option.content = const_cast<char *>(seqn_str.c_str());

    respond(sock, session, &option, 200, "OK", req->sequenceNumber, {});
  }

  void cmd_describe(rtsp_server_t *server, tcp::socket &sock, launch_session_t &session, msg_t &&req) {
    OPTION_ITEM option {};

    // I know these string literals will not be modified
    option.option = const_cast<char *>("CSeq");

    auto seqn_str = std::to_string(req->sequenceNumber);
    option.content = const_cast<char *>(seqn_str.c_str());

    std::stringstream ss;

    // Tell the client about our supported features
    ss << "a=x-ss-general.featureFlags:" << (uint32_t) platf::get_capabilities() << std::endl;

    // Modern Artemis encrypts control, audio, and video. Apollo has no plaintext media mode.
    constexpr uint32_t encryption_flags_supported = SS_ENC_CONTROL_V2 | SS_ENC_AUDIO | SS_ENC_VIDEO;
    constexpr uint32_t encryption_flags_requested = encryption_flags_supported;

    // Report supported and required encryption flags
    ss << "a=x-ss-general.encryptionSupported:" << encryption_flags_supported << std::endl;
    ss << "a=x-ss-general.encryptionRequested:" << encryption_flags_requested << std::endl;

    // Artemis uses reference-picture invalidation to recover packet loss without a full IDR.
    // The native NVENC session falls back to an IDR if the active codec cannot honor a request.
    ss << "a=x-nv-video[0].refPicInvalidation:1"sv << std::endl;

    const auto codec_capabilities = video::nvenc_capabilities_snapshot();
    if (codec_capabilities.hevc) {
      ss << "sprop-parameter-sets=AAAAAU"sv << std::endl;
    }

    if (codec_capabilities.av1) {
      ss << "a=rtpmap:98 AV1/90000"sv << std::endl;
    }

    for (int x = 0; x < audio::MAX_STREAM_CONFIG; ++x) {
      auto &stream_config = audio::stream_configs[x];
      std::uint8_t mapping[platf::speaker::MAX_SPEAKERS];

      auto mapping_p = stream_config.mapping;

      /**
       * GFE advertises incorrect mapping for normal quality configurations,
       * as a result, Artemis rotates all channels from index '3' to the right
       * To work around this, rotate channels to the left from index '3'
       */
      if (x == audio::SURROUND51 || x == audio::SURROUND71) {
        std::copy_n(mapping_p, stream_config.channelCount, mapping);
        std::rotate(mapping + 3, mapping + 4, mapping + audio::MAX_STREAM_CONFIG);

        mapping_p = mapping;
      }

      ss << "a=fmtp:97 surround-params="sv << stream_config.channelCount << stream_config.streams << stream_config.coupledStreams;

      std::for_each_n(mapping_p, stream_config.channelCount, [&ss](std::uint8_t digit) {
        ss << (char) (digit + '0');
      });

      ss << std::endl;
    }

    respond(sock, session, &option, 200, "OK", req->sequenceNumber, ss.str());
  }

  void cmd_setup(rtsp_server_t *server, tcp::socket &sock, launch_session_t &session, msg_t &&req) {
    OPTION_ITEM options[4] {};

    auto &seqn = options[0];
    auto &session_option = options[1];
    auto &port_option = options[2];
    auto &payload_option = options[3];

    seqn.option = const_cast<char *>("CSeq");

    auto seqn_str = std::to_string(req->sequenceNumber);
    seqn.content = const_cast<char *>(seqn_str.c_str());

    std::string_view target {req->message.request.target};
    const auto parsed_type = detail::parse_setup_stream_type(target);
    if (!parsed_type) {
      BOOST_LOG(warning) << "Rejecting malformed RTSP SETUP target ["sv << target << ']';
      respond(sock, session, nullptr, 400, "BAD REQUEST", req->sequenceNumber, {});
      return;
    }
    const auto type = *parsed_type;

    std::uint16_t port;
    if (type == "audio"sv) {
      port = net::map_port(stream::AUDIO_STREAM_PORT);
    } else if (type == "video"sv) {
      port = net::map_port(stream::VIDEO_STREAM_PORT);
    } else if (type == "control"sv) {
      port = net::map_port(stream::CONTROL_PORT);
    } else {
      cmd_not_found(sock, session, std::move(req));

      return;
    }

    seqn.next = &session_option;

    session_option.option = const_cast<char *>("Session");
    session_option.content = const_cast<char *>("DEADBEEFCAFE;timeout = 90");

    session_option.next = &port_option;

    // Artemis merely requires 'server_port=<port>'
    auto port_value = std::format("server_port={}", static_cast<int>(port));

    port_option.option = const_cast<char *>("Transport");
    port_option.content = port_value.data();

    // Send identifiers that will be echoed in the other connections
    auto connect_data = std::to_string(session.control_connect_data);
    if (type == "control"sv) {
      payload_option.option = const_cast<char *>("X-SS-Connect-Data");
      payload_option.content = connect_data.data();
    } else {
      payload_option.option = const_cast<char *>("X-SS-Ping-Payload");
      payload_option.content = session.av_ping_payload.data();
    }

    port_option.next = &payload_option;

    respond(sock, session, &seqn, 200, "OK", req->sequenceNumber, {});
  }

  void cmd_announce(rtsp_server_t *server, tcp::socket &sock, launch_session_t &session, msg_t &&req) {
    OPTION_ITEM option {};

    // I know these string literals will not be modified
    option.option = const_cast<char *>("CSeq");

    auto seqn_str = std::to_string(req->sequenceNumber);
    option.content = const_cast<char *>(seqn_str.c_str());

    std::string_view payload {req->payload, (size_t) req->payloadLength};

    auto args = detail::parse_announce_attributes(payload);

    stream::config_t config;

    std::int64_t configuredBitrateKbps;
    config.audio.flags[audio::config_t::HOST_AUDIO] = session.host_audio;
    try {
      const auto required_int = [&](detail::announce_int_field field, std::string_view name) {
        const auto value = args.at(name);
        const auto parsed = detail::parse_announce_int(field, value);
        if (!parsed) {
          throw std::invalid_argument(std::string {name});
        }
        return *parsed;
      };

      config.audio.channels = required_int(detail::announce_int_field::audio_channels, "x-nv-audio.surround.numChannels"sv);
      config.audio.mask = required_int(detail::announce_int_field::audio_channel_mask, "x-nv-audio.surround.channelMask"sv);
      config.audio.packetDuration = required_int(detail::announce_int_field::audio_packet_duration, "x-nv-aqos.packetDuration"sv);

      config.audio.flags[audio::config_t::HIGH_QUALITY] =
        required_int(detail::announce_int_field::audio_quality, "x-nv-audio.surround.AudioQuality"sv);

      required_int(detail::announce_int_field::control_protocol, "x-nv-general.useReliableUdp"sv);
      const auto packet_size_arg = args.at("x-nv-video[0].packetSize"sv);
      const auto min_fec_packets_arg = args.at("x-nv-vqos[0].fec.minRequiredFecPackets"sv);
      const auto packet_size = util::from_view_checked<int>(packet_size_arg);
      const auto min_fec_packets = util::from_view_checked<int>(min_fec_packets_arg);
      if (!packet_size || !min_fec_packets || !stream::is_valid_video_transport_config(*packet_size, *min_fec_packets)) {
        BOOST_LOG(warning) << "Rejecting invalid RTSP video transport parameters: packetSize=["sv << packet_size_arg
                           << "], minRequiredFecPackets=["sv << min_fec_packets_arg << ']';
        respond(sock, session, &option, 400, "BAD REQUEST", req->sequenceNumber, {});
        return;
      }
      config.packetsize = detail::apply_packet_size_limit(*packet_size, ::config::stream.packet_size_limit);
      if (config.packetsize != *packet_size) {
        if (config.packetsize < 500) {
          BOOST_LOG(info) << "Configured packetsize limit is small; reduce bitrate if the stream becomes unstable."sv;
        }
        BOOST_LOG(info) << "Applying video packetsize limit: "sv << *packet_size << " -> "sv
                        << config.packetsize << " bytes"sv;
      }
      config.minRequiredFecPackets = *min_fec_packets;
      const auto client_features = required_int(detail::announce_int_field::feature_flags, "x-ml-general.featureFlags"sv);
      if (!(client_features & ML_FF_SESSION_ID_V1)) {
        throw std::invalid_argument("x-ml-general.featureFlags requires SESSION_ID_V1");
      }
      config.audioQosType = required_int(detail::announce_int_field::audio_qos, "x-nv-aqos.qosTrafficType"sv);
      config.videoQosType = required_int(detail::announce_int_field::video_qos, "x-nv-vqos[0].qosTrafficType"sv);
      const auto encryption_flags = required_int(
        detail::announce_int_field::encryption_flags,
        "x-ss-general.encryptionEnabled"sv
      );
      constexpr auto required_encryption = SS_ENC_CONTROL_V2 | SS_ENC_VIDEO | SS_ENC_AUDIO;
      if ((encryption_flags & required_encryption) != required_encryption) {
        throw std::invalid_argument("x-ss-general.encryptionEnabled requires CONTROL_V2, VIDEO, and AUDIO");
      }

      config.monitor.height = required_int(detail::announce_int_field::viewport_dimension, "x-nv-video[0].clientViewportHt"sv);
      config.monitor.width = required_int(detail::announce_int_field::viewport_dimension, "x-nv-video[0].clientViewportWd"sv);
      config.monitor.framerate = required_int(detail::announce_int_field::max_fps, "x-nv-video[0].maxFPS"sv);
      const auto client_refresh_x100 = required_int(
        detail::announce_int_field::client_refresh_x100,
        "x-nv-video[0].clientRefreshRateX100"sv
      );
      config.monitor.framerateX100 = detail::validated_client_refresh_x100(
        config.monitor.framerate,
        client_refresh_x100
      );
      if (client_refresh_x100 > 0 && config.monitor.framerateX100 == 0) {
        BOOST_LOG(debug) << "Ignoring client display refresh ["sv << (client_refresh_x100 / 100.0)
                         << "Hz] because it does not match the requested stream rate."sv;
      }
      config.monitor.bitrate = required_int(detail::announce_int_field::bitrate_kbps, "x-nv-vqos[0].bw.maximumBitrateKbps"sv);
      config.monitor.slicesPerFrame = required_int(detail::announce_int_field::slices_per_frame, "x-nv-video[0].videoEncoderSlicesPerFrame"sv);
      config.monitor.numRefFrames = required_int(detail::announce_int_field::reference_frames, "x-nv-video[0].maxNumReferenceFrames"sv);
      config.monitor.encoderCscMode = required_int(detail::announce_int_field::encoder_csc_mode, "x-nv-video[0].encoderCscMode"sv);
      config.monitor.videoFormat = required_int(detail::announce_int_field::video_format, "x-nv-vqos[0].bitStreamFormat"sv);
      config.monitor.dynamicRange = required_int(detail::announce_int_field::binary_option, "x-nv-video[0].dynamicRangeMode"sv);
      if (required_int(detail::announce_int_field::binary_option, "x-ss-video[0].chromaSamplingType"sv) != 0) {
        throw std::invalid_argument("x-ss-video[0].chromaSamplingType requires 4:2:0");
      }
      if (session.fps <= 0 || session.fps > 1000000) {
        throw std::invalid_argument("launch session frame rate");
      }

      if (config.monitor.framerateX100 > 0) {
        config.monitor.encodingFramerate = config.monitor.framerateX100 * 10;
      } else {
        config.monitor.encodingFramerate = session.fps;
      }

      // When fractional refresh rate requested from client side, it should be well above 1000fps
      // 4000fps is when Warp2 Mode is enabled on the client, requested framerate can be actual * 4
      if (config.monitor.framerate > 4000) {
        config.monitor.framerate = std::round((float) config.monitor.framerate / 1000);
      }

      config.monitor.sbs_mode = session.sbs_mode;

      configuredBitrateKbps = required_int(detail::announce_int_field::configured_bitrate_kbps, "x-ml-video.configuredBitrateKbps"sv);

      if (!configuredBitrateKbps) {
        configuredBitrateKbps = config.monitor.bitrate;
      }

      BOOST_LOG(info) << "Client Requested bitrate is [" << configuredBitrateKbps << "kbps]";

      if (config::video.max_bitrate > 0) {
        if (config::video.max_bitrate < configuredBitrateKbps) {
          configuredBitrateKbps = config::video.max_bitrate;
        }
      }

      BOOST_LOG(info) << "Host Streaming bitrate is [" << configuredBitrateKbps << "kbps]";

      // Preserve the requested bitrate budget when Artemis uses a Warp frame-rate multiplier.
      const auto warp_factor = detail::calculate_warp_bitrate_factor(config.monitor.framerate, session.fps);
      if (warp_factor >= 2) {
        configuredBitrateKbps *= warp_factor;
        BOOST_LOG(info) << "Warp factor [" << warp_factor << "] engaged";
      }

      if (!detail::is_safe_encoder_bitrate(configuredBitrateKbps)) {
        throw std::invalid_argument("encoding bitrate");
      }

    } catch (std::out_of_range &) {
      respond(sock, session, &option, 400, "BAD REQUEST", req->sequenceNumber, {});
      return;
    } catch (const std::invalid_argument &error) {
      BOOST_LOG(warning) << "Rejecting invalid RTSP ANNOUNCE parameter: "sv << error.what();
      respond(sock, session, &option, 400, "BAD REQUEST", req->sequenceNumber, {});
      return;
    }

    // If the client sent a configured bitrate, we will choose the actual bitrate ourselves
    // by using FEC percentage and audio quality settings. If the calculated bitrate ends up
    // too low, we'll allow it to exceed the limits rather than reducing the encoding bitrate
    // down to nearly nothing.
    if (configuredBitrateKbps) {
      BOOST_LOG(debug) << "Client configured bitrate is "sv << configuredBitrateKbps << " Kbps"sv;

      // The bitrate request is a total wire budget. Audio/control are not protected by video FEC,
      // so deduct them before reserving the remaining budget for video data plus parity.
      const auto audioBitrateKbps =
        (config.audio.flags[audio::config_t::HIGH_QUALITY] ? 256 : 96) *
        config.audio.channels;
      configuredBitrateKbps = detail::calculate_video_bitrate_budget(
        configuredBitrateKbps,
        config::stream.fec_percentage,
        audioBitrateKbps
      );

      BOOST_LOG(debug) << "Final adjusted video encoding bitrate is "sv << configuredBitrateKbps << " Kbps"sv;
      config.monitor.bitrate = configuredBitrateKbps;
    }

    const auto codec_capabilities = video::nvenc_capabilities_snapshot();
    if (!detail::is_video_mode_supported(
          config.monitor.videoFormat,
          config.monitor.dynamicRange,
          codec_capabilities.hevc,
          codec_capabilities.hevc_hdr,
          codec_capabilities.av1,
          codec_capabilities.av1_hdr
        )) {
      BOOST_LOG(warning) << "Rejecting unsupported codec/bit-depth request: format="sv
                         << config.monitor.videoFormat << ", dynamicRange="sv
                         << config.monitor.dynamicRange;
      respond(sock, session, &option, 400, "BAD REQUEST", req->sequenceNumber, {});
      return;
    }

    // Claim only after the ANNOUNCE payload, codec, encryption, and authorization checks pass.
    // A copied socket whose pending reservation was cleared cannot claim and resurrect teardown.
    if (!server->claim_launch_session(session)) {
      BOOST_LOG(warning) << "Rejecting stale or duplicate RTSP ANNOUNCE reservation."sv;
      respond(sock, session, &option, 454, "Session Not Found", req->sequenceNumber, {});
      return;
    }

    auto stream_session = stream::session::alloc(config, session);
    const auto start_result = server->start_and_insert(stream_session, session);
    if (start_result == 403) {
      BOOST_LOG(warning) << "Rejecting streaming session for a client with revoked view permission"sv;
      respond(sock, session, &option, 403, "Forbidden", req->sequenceNumber, {});
      return;
    }
    if (start_result == 409) {
      BOOST_LOG(warning) << "Rejecting streaming session because another stream is already active"sv;
      respond(sock, session, &option, 503, "Service Unavailable", req->sequenceNumber, {});
      return;
    }
    if (start_result != 0) {
      if (start_result == 454) {
        BOOST_LOG(warning) << "Rejecting RTSP session whose launch reservation was revoked during startup."sv;
        respond(sock, session, &option, 454, "Session Not Found", req->sequenceNumber, {});
        return;
      }
      BOOST_LOG(error) << "Failed to start a streaming session"sv;
      respond(sock, session, &option, 500, "Internal Server Error", req->sequenceNumber, {});
      return;
    }

    respond(sock, session, &option, 200, "OK", req->sequenceNumber, {});
  }

  void cmd_play(rtsp_server_t *server, tcp::socket &sock, launch_session_t &session, msg_t &&req) {
    OPTION_ITEM option {};

    // I know these string literals will not be modified
    option.option = const_cast<char *>("CSeq");

    auto seqn_str = std::to_string(req->sequenceNumber);
    option.content = const_cast<char *>(seqn_str.c_str());

    respond(sock, session, &option, 200, "OK", req->sequenceNumber, {});
  }

  void start() {
    auto shutdown_event = mail::man->event<bool>(mail::shutdown);

    server.map("OPTIONS"sv, &cmd_option);
    server.map("DESCRIBE"sv, &cmd_describe);
    server.map("SETUP"sv, &cmd_setup);
    server.map("ANNOUNCE"sv, &cmd_announce);
    server.map("PLAY"sv, &cmd_play);

    boost::system::error_code ec;
    if (server.bind(net::af_from_enum_string(config::sunshine.address_family), net::map_port(rtsp_stream::RTSP_SETUP_PORT), ec)) {
      BOOST_LOG(fatal) << "Couldn't bind RTSP server to port ["sv << net::map_port(rtsp_stream::RTSP_SETUP_PORT) << "], " << ec.message();
      shutdown_event->raise(true);

      return;
    }

    std::thread rtsp_thread {[&shutdown_event] {
      auto broadcast_shutdown_event = mail::man->event<bool>(mail::broadcast_shutdown);

      while (!shutdown_event->peek()) {
        server.iterate();

        if (broadcast_shutdown_event->peek()) {
          server.clear();
        } else {
          // Clean up the stopped session.
          server.clear(false);
        }
      }

      server.clear();
    }};

    // Wait for shutdown
    shutdown_event->view();

    // Stop the server and join the server thread
    server.stop();
    rtsp_thread.join();
  }

  void print_msg(PRTSP_MESSAGE msg) {
    std::string_view type = msg->type == TYPE_RESPONSE ? "RESPONSE"sv : "REQUEST"sv;

    std::string_view payload {msg->payload, (size_t) msg->payloadLength};
    std::string_view protocol {msg->protocol};
    auto seqnm = msg->sequenceNumber;
    std::string_view messageBuffer {msg->messageBuffer};

    BOOST_LOG(debug) << "type ["sv << type << ']';
    BOOST_LOG(debug) << "sequence number ["sv << seqnm << ']';
    BOOST_LOG(debug) << "protocol :: "sv << protocol;
    BOOST_LOG(debug) << "payload :: "sv << payload;

    if (msg->type == TYPE_RESPONSE) {
      auto &resp = msg->message.response;

      auto statuscode = resp.statusCode;
      std::string_view status {resp.statusString};

      BOOST_LOG(debug) << "statuscode :: "sv << statuscode;
      BOOST_LOG(debug) << "status :: "sv << status;
    } else {
      auto &req = msg->message.request;

      std::string_view command {req.command};
      std::string_view target {req.target};

      BOOST_LOG(debug) << "command :: "sv << command;
      BOOST_LOG(debug) << "target :: "sv << target;
    }

    for (auto option = msg->options; option != nullptr; option = option->next) {
      std::string_view content {option->content};
      std::string_view name {option->option};

      BOOST_LOG(debug) << name << " :: "sv << content;
    }

    BOOST_LOG(debug) << "---Begin MessageBuffer---"sv << std::endl
                     << messageBuffer << std::endl
                     << "---End MessageBuffer---"sv << std::endl;
  }
}  // namespace rtsp_stream
