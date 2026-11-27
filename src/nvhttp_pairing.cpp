/**
 * @file src/nvhttp_pairing.cpp
 * @brief GameStream pairing: the PIN and OTP pairing protocol and its pending sessions.
 */
// macros
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

// lib includes
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <Simple-Web-Server/server_http.hpp>

// local includes
#include "config.h"
#include "crypto.h"
#include "logging.h"
#include "nvhttp.h"
#include "nvhttp_internal.h"
#include "system_tray.h"
#include "utility.h"
#include "uuid.h"

using namespace std::literals;

namespace nvhttp {
  namespace pt = boost::property_tree;

  namespace detail {
    pairing_admission_e pairing_admission(
      std::size_t active_sessions,
      std::size_t maximum_sessions,
      bool replacing_existing,
      bool manual_pin,
      bool another_manual_pin_pending
    ) {
      if (!replacing_existing && active_sessions >= maximum_sessions) {
        return pairing_admission_e::capacity_reached;
      }
      if (manual_pin && another_manual_pin_pending) {
        return pairing_admission_e::manual_pin_conflict;
      }
      return pairing_admission_e::allowed;
    }
  }  // namespace detail

  static std::string one_time_pin;
  static std::string otp_passphrase;
  static std::string otp_device_name;
  static std::chrono::time_point<std::chrono::steady_clock> otp_creation_time;

  struct conf_intern_t {
    std::string servercert;
    std::string pkey;
  } conf_intern;

  // Pairing requests are accepted by independent HTTP and HTTPS server threads, while PIN/OTP
  // operations arrive from the configuration server. Map entries therefore have shared ownership:
  // replacing or erasing a unique ID cannot invalidate a phase already in progress.
  std::mutex pairing_state_mutex;
  std::unordered_map<std::string, std::shared_ptr<pair_session_t>> map_id_sess;
  constexpr std::size_t MAX_PAIRING_SESSIONS = 16;
  std::uint64_t pairing_sequence;

  void prune_pairing_sessions_locked(std::chrono::steady_clock::time_point now) {
    std::erase_if(map_id_sess, [now](const auto &entry) {
      return now - entry.second->created_at > OTP_EXPIRE_DURATION;
    });
  }

  enum class otp_auth_status {
    unavailable,
    invalid,
    valid,
  };

  struct otp_auth_result_t {
    otp_auth_status status = otp_auth_status::unavailable;
    std::string pin;
    std::string device_name;
  };

  void clear_otp_locked() {
    one_time_pin.clear();
    otp_passphrase.clear();
    otp_device_name.clear();
  }

  otp_auth_result_t authenticate_otp(std::string_view authentication, std::string_view salt) {
    std::lock_guard lock(pairing_state_mutex);
    const auto now = std::chrono::steady_clock::now();
    if (one_time_pin.empty() || now - otp_creation_time > OTP_EXPIRE_DURATION) {
      // Expired credentials cannot become valid again and may be discarded immediately.
      clear_otp_locked();
      return {};
    }

    const auto expected = util::hex(crypto::hash(one_time_pin + std::string(salt) + otp_passphrase), true);
    if (expected.to_string_view() != authentication) {
      // An unauthenticated request must not invalidate a legitimate user's still-live OTP.
      return {.status = otp_auth_status::invalid};
    }

    otp_auth_result_t result {
      .status = otp_auth_status::valid,
      .pin = one_time_pin,
      .device_name = otp_device_name,
    };
    clear_otp_locked();
    return result;
  }

#ifdef SUNSHINE_TESTS
  bool authenticate_otp_for_test(std::string authentication, std::string salt) {
    return authenticate_otp(authentication, salt).status == otp_auth_status::valid;
  }
#endif

  void remove_session(const pair_session_t &sess) {
    std::lock_guard lock(pairing_state_mutex);
    const auto session = std::find_if(map_id_sess.begin(), map_id_sess.end(), [&](const auto &entry) {
      return entry.second.get() == &sess;
    });
    if (session != map_id_sess.end()) {
      map_id_sess.erase(session);
    }
  }

  void fail_pair(pair_session_t &sess, pt::ptree &tree, const std::string status_msg) {
    tree.put("root.paired", 0);
    tree.put("root.<xmlattr>.status_code", 400);
    tree.put("root.<xmlattr>.status_message", status_msg);
    remove_session(sess);  // Security measure, delete the session when something went wrong and force a re-pair
    BOOST_LOG(warning) << "Pair attempt failed due to " << status_msg;
  }

  void getservercert(pair_session_t &sess, pt::ptree &tree, const std::string &pin) {
    std::lock_guard session_lock(*sess.state_mutex);
    if (sess.last_phase != PAIR_PHASE::NONE) {
      fail_pair(sess, tree, "Out of order call to getservercert");
      return;
    }
    sess.last_phase = PAIR_PHASE::GETSERVERCERT;

    if (sess.async_insert_pin.salt.size() < 32) {
      fail_pair(sess, tree, "Salt too short");
      return;
    }

    std::string_view salt_view {sess.async_insert_pin.salt.data(), 32};

    auto salt = util::from_hex<std::array<uint8_t, 16>>(salt_view, true);

    auto key = crypto::gen_aes_key(salt, pin);
    sess.cipher_key = std::make_unique<crypto::aes_t>(key);

    tree.put("root.paired", 1);
    tree.put("root.plaincert", util::hex_vec(conf_intern.servercert, true));
    tree.put("root.<xmlattr>.status_code", 200);
  }

  void clientchallenge(pair_session_t &sess, pt::ptree &tree, const std::string &challenge) {
    std::lock_guard session_lock(*sess.state_mutex);
    if (sess.last_phase != PAIR_PHASE::GETSERVERCERT) {
      fail_pair(sess, tree, "Out of order call to clientchallenge");
      return;
    }
    sess.last_phase = PAIR_PHASE::CLIENTCHALLENGE;

    if (!sess.cipher_key) {
      fail_pair(sess, tree, "Cipher key not set");
      return;
    }
    crypto::cipher::ecb_t cipher(*sess.cipher_key, false);

    std::vector<uint8_t> decrypted;
    cipher.decrypt(challenge, decrypted);

    auto x509 = crypto::x509(conf_intern.servercert);
    auto sign = crypto::signature(x509);
    auto serversecret = crypto::rand(16);

    decrypted.insert(std::end(decrypted), std::begin(sign), std::end(sign));
    decrypted.insert(std::end(decrypted), std::begin(serversecret), std::end(serversecret));

    auto hash = crypto::hash({(char *) decrypted.data(), decrypted.size()});
    auto serverchallenge = crypto::rand(16);

    std::string plaintext;
    plaintext.reserve(hash.size() + serverchallenge.size());

    plaintext.insert(std::end(plaintext), std::begin(hash), std::end(hash));
    plaintext.insert(std::end(plaintext), std::begin(serverchallenge), std::end(serverchallenge));

    std::vector<uint8_t> encrypted;
    cipher.encrypt(plaintext, encrypted);

    sess.serversecret = std::move(serversecret);
    sess.serverchallenge = std::move(serverchallenge);

    tree.put("root.paired", 1);
    tree.put("root.challengeresponse", util::hex_vec(encrypted, true));
    tree.put("root.<xmlattr>.status_code", 200);
  }

  void serverchallengeresp(pair_session_t &sess, pt::ptree &tree, const std::string &encrypted_response) {
    std::lock_guard session_lock(*sess.state_mutex);
    if (sess.last_phase != PAIR_PHASE::CLIENTCHALLENGE) {
      fail_pair(sess, tree, "Out of order call to serverchallengeresp");
      return;
    }
    sess.last_phase = PAIR_PHASE::SERVERCHALLENGERESP;

    if (!sess.cipher_key || sess.serversecret.empty()) {
      fail_pair(sess, tree, "Cipher key or serversecret not set");
      return;
    }

    std::vector<uint8_t> decrypted;
    crypto::cipher::ecb_t cipher(*sess.cipher_key, false);

    cipher.decrypt(encrypted_response, decrypted);

    sess.clienthash = std::move(decrypted);

    auto serversecret = sess.serversecret;
    auto sign = crypto::sign256(crypto::pkey(conf_intern.pkey), serversecret);

    serversecret.insert(std::end(serversecret), std::begin(sign), std::end(sign));

    tree.put("root.pairingsecret", util::hex_vec(serversecret, true));
    tree.put("root.paired", 1);
    tree.put("root.<xmlattr>.status_code", 200);
  }

  void clientpairingsecret(pair_session_t &sess, pt::ptree &tree, const std::string &client_pairing_secret) {
    p_named_cert_t authorized_client;
    {
      std::lock_guard session_lock(*sess.state_mutex);
      if (sess.last_phase != PAIR_PHASE::SERVERCHALLENGERESP) {
        fail_pair(sess, tree, "Out of order call to clientpairingsecret");
        return;
      }
      sess.last_phase = PAIR_PHASE::CLIENTPAIRINGSECRET;

      auto &client = sess.client;
      if (client_pairing_secret.size() <= 16) {
        fail_pair(sess, tree, "Client pairing secret too short");
        return;
      }

      std::string_view secret {client_pairing_secret.data(), 16};
      std::string_view sign {client_pairing_secret.data() + secret.size(), client_pairing_secret.size() - secret.size()};
      auto x509 = crypto::x509(client.cert);
      if (!x509) {
        fail_pair(sess, tree, "Invalid client certificate");
        return;
      }
      auto x509_sign = crypto::signature(x509);

      std::string data;
      data.reserve(sess.serverchallenge.size() + x509_sign.size() + secret.size());
      data.insert(std::end(data), std::begin(sess.serverchallenge), std::end(sess.serverchallenge));
      data.insert(std::end(data), std::begin(x509_sign), std::end(x509_sign));
      data.insert(std::end(data), std::begin(secret), std::end(secret));

      const auto hash = crypto::hash(data);
      const bool same_hash = hash.size() == sess.clienthash.size() &&
                             std::equal(hash.begin(), hash.end(), sess.clienthash.begin());
      const bool verified = crypto::verify256(crypto::x509(client.cert), secret, sign);
      if (same_hash && verified) {
        tree.put("root.paired", 1);
        authorized_client = std::make_shared<crypto::named_cert_t>();
        authorized_client->name = client.name;
        for (char &c : authorized_client->name) {
          if (c == '(') {
            c = '[';
          } else if (c == ')') {
            c = ']';
          }
        }
        authorized_client->cert = std::move(client.cert);
        authorized_client->uuid = uuid_util::uuid_t::generate().string();
      } else {
        tree.put("root.paired", 0);
        BOOST_LOG(warning) << "Pair attempt failed due to same_hash: " << same_hash
                           << ", verify: " << verified;
      }
    }

    // Persistence and tray updates may perform file/UI work. Do them after releasing the session
    // lock; the map holds shared ownership until remove_session() below.
    if (authorized_client) {
      add_authorized_client(authorized_client);
    }

    remove_session(sess);
    tree.put("root.<xmlattr>.status_code", 200);
  }

  template<class T>
  void pair(std::shared_ptr<typename SimpleWeb::ServerBase<T>::Response> response, std::shared_ptr<typename SimpleWeb::ServerBase<T>::Request> request) {
    print_req<T>(request);

    pt::ptree tree;

    auto fg = util::fail_guard([&]() {
      std::ostringstream data;

      pt::write_xml(data, tree);
      response->write(data.str());
      response->close_connection_after_response = true;
    });

    if (!config::sunshine.enable_pairing) {
      tree.put("root.<xmlattr>.status_code", 403);
      tree.put("root.<xmlattr>.status_message", "Pairing is disabled for this instance");

      return;
    }

    auto args = request->parse_query_string();
    if (args.find("uniqueid"s) == std::end(args)) {
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Missing uniqueid parameter");

      return;
    }

    auto uniqID {get_arg(args, "uniqueid")};

    args_t::const_iterator it;
    if (it = args.find("phrase"); it != std::end(args)) {
      if (it->second == "getservercert"sv) {
        auto session = std::make_shared<pair_session_t>();
        auto deviceName {get_arg(args, "devicename")};
        session->client.uniqueID = uniqID;
        session->client.name = std::move(deviceName);
        session->client.cert = util::from_hex_vec(get_arg(args, "clientcert"), true);
        session->async_insert_pin.salt = std::move(get_arg(args, "salt"));

        const auto otp_auth = args.find("otpauth");
        const bool manual_pin = otp_auth == args.end();
        if (manual_pin) {
          // Publish a complete pending-PIN session. pin() can never observe the map entry before
          // its asynchronous response endpoint has been attached.
          session->async_insert_pin.response = response;
          session->manual_pin_pending = true;
        }
        {
          std::lock_guard lock(pairing_state_mutex);
          prune_pairing_sessions_locked(std::chrono::steady_clock::now());

          const auto existing = map_id_sess.find(uniqID);
          const auto another_manual_pin_pending = std::any_of(map_id_sess.begin(), map_id_sess.end(), [&](const auto &entry) {
            return entry.first != uniqID && entry.second->manual_pin_pending;
          });
          const auto admission = detail::pairing_admission(
            map_id_sess.size(),
            MAX_PAIRING_SESSIONS,
            existing != map_id_sess.end(),
            manual_pin,
            another_manual_pin_pending
          );
          if (admission == detail::pairing_admission_e::capacity_reached) {
            tree.put("root.<xmlattr>.status_code", 503);
            tree.put("root.<xmlattr>.status_message", "Too many pairing requests are pending");
            return;
          }

          if (admission == detail::pairing_admission_e::manual_pin_conflict) {
            // The PIN submission API carries no unique ID. Admitting two manual waiters would
            // make it impossible to know which client the user intended to authorize.
            tree.put("root.<xmlattr>.status_code", 409);
            tree.put("root.<xmlattr>.status_message", "Another pairing request is awaiting a PIN");
            return;
          }

          session->created_at = std::chrono::steady_clock::now();
          session->insertion_sequence = ++pairing_sequence;
          map_id_sess.insert_or_assign(uniqID, session);
        }

        if (otp_auth != args.end()) {
          const auto result = authenticate_otp(otp_auth->second, session->async_insert_pin.salt);
          if (result.status == otp_auth_status::valid) {
            if (!result.device_name.empty()) {
              std::lock_guard session_lock(*session->state_mutex);
              session->client.name = result.device_name;
            }
            getservercert(*session, tree, result.pin);
          } else {
            if (result.status == otp_auth_status::unavailable) {
              tree.put("root.<xmlattr>.status_code", 503);
              tree.put("root.<xmlattr>.status_message", "OTP auth not available.");
            }
            // Keep the first response indistinguishable; attackers fail in the signed phases.
            getservercert(*session, tree, crypto::rand(16));
          }
          return;
        }

        system_tray::update_tray_require_pin();
        fg.disable();
        return;
      } else if (it->second == "pairchallenge"sv) {
        tree.put("root.paired", 1);
        tree.put("root.<xmlattr>.status_code", 200);
        return;
      }
    }

    std::shared_ptr<pair_session_t> session;
    {
      std::lock_guard lock(pairing_state_mutex);
      prune_pairing_sessions_locked(std::chrono::steady_clock::now());
      const auto sess_it = map_id_sess.find(uniqID);
      if (sess_it != map_id_sess.end()) {
        session = sess_it->second;
      }
    }
    if (!session) {
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Invalid uniqueid");

      return;
    }

    if (it = args.find("clientchallenge"); it != std::end(args)) {
      auto challenge = util::from_hex_vec(it->second, true);
      clientchallenge(*session, tree, challenge);
    } else if (it = args.find("serverchallengeresp"); it != std::end(args)) {
      auto encrypted_response = util::from_hex_vec(it->second, true);
      serverchallengeresp(*session, tree, encrypted_response);
    } else if (it = args.find("clientpairingsecret"); it != std::end(args)) {
      auto pairingsecret = util::from_hex_vec(it->second, true);
      clientpairingsecret(*session, tree, pairingsecret);
    } else {
      tree.put("root.<xmlattr>.status_code", 404);
      tree.put("root.<xmlattr>.status_message", "Invalid pairing request");
    }
  }

  template void pair<SunshineHTTPS>(resp_https_t response, req_https_t request);
  template void pair<SimpleWeb::HTTP>(resp_http_t response, req_http_t request);

  bool pin(std::string pin, std::string name) {
    pt::ptree tree;
    // ensure pin is 4 digits
    if (pin.size() != 4) {
      tree.put("root.paired", 0);
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put(
        "root.<xmlattr>.status_message",
        std::format("Pin must be 4 digits, {} provided", pin.size())
      );
      return false;
    }

    // ensure all pin characters are numeric
    if (!std::all_of(pin.begin(), pin.end(), ::isdigit)) {
      tree.put("root.paired", 0);
      tree.put("root.<xmlattr>.status_code", 400);
      tree.put("root.<xmlattr>.status_message", "Pin must be numeric");
      return false;
    }

    std::shared_ptr<pair_session_t> session;
    {
      std::lock_guard lock(pairing_state_mutex);
      prune_pairing_sessions_locked(std::chrono::steady_clock::now());
      const auto pending = std::min_element(map_id_sess.begin(), map_id_sess.end(), [](const auto &left, const auto &right) {
        const auto left_sequence = left.second->manual_pin_pending ? left.second->insertion_sequence : std::numeric_limits<std::uint64_t>::max();
        const auto right_sequence = right.second->manual_pin_pending ? right.second->insertion_sequence : std::numeric_limits<std::uint64_t>::max();
        return left_sequence < right_sequence;
      });
      if (pending != map_id_sess.end() && pending->second->manual_pin_pending) {
        session = pending->second;
        session->manual_pin_pending = false;
      }
    }
    if (!session) {
      return false;
    }

    if (!name.empty()) {
      std::lock_guard session_lock(*session->state_mutex);
      session->client.name = std::move(name);
    }
    getservercert(*session, tree, pin);

    // response to the request for pin
    std::ostringstream data;
    pt::write_xml(data, tree);

    std::shared_ptr<typename SimpleWeb::ServerBase<SimpleWeb::HTTP>::Response> http_response;
    std::shared_ptr<typename SimpleWeb::ServerBase<SunshineHTTPS>::Response> https_response;
    {
      std::lock_guard session_lock(*session->state_mutex);
      auto &async_response = session->async_insert_pin.response;
      if (async_response.has_left()) {
        http_response = async_response.left();
      } else if (async_response.has_right()) {
        https_response = async_response.right();
      }
      // Detach the endpoint before network I/O. A duplicate PIN submission cannot write a second
      // response, and no session mutex is held while the server flushes it.
      async_response = std::decay_t<decltype(async_response.left())>();
    }
    if (http_response) {
      http_response->write(data.str());
    } else if (https_response) {
      https_response->write(data.str());
    } else {
      return false;
    }
    return true;
  }

  void setup(const std::string &pkey, const std::string &cert) {
    conf_intern.pkey = pkey;
    conf_intern.servercert = cert;
  }

  std::string request_otp(const std::string &passphrase, const std::string &deviceName) {
    if (passphrase.size() < 4) {
      return "";
    }

    std::lock_guard lock(pairing_state_mutex);
    one_time_pin = crypto::rand_alphabet(4, "0123456789"sv);
    otp_passphrase = passphrase;
    otp_device_name = deviceName;
    otp_creation_time = std::chrono::steady_clock::now();
    return one_time_pin;
  }

  void clear_pairing_state() {
    std::lock_guard lock(pairing_state_mutex);
    map_id_sess.clear();
    clear_otp_locked();
  }
}  // namespace nvhttp
