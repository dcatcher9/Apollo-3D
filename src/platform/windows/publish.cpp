/**
 * @file src/platform/windows/publish.cpp
 * @brief Definitions for Windows mDNS service registration.
 */
// platform includes
// WinSock2.h must be included before Windows.h
// clang-format off
#include <WinSock2.h>
#include <Windows.h>
// clang-format on
#include <WinDNS.h>
#include <winerror.h>

// local includes
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/network.h"
#include "src/nvhttp.h"
#include "src/platform/common.h"
#include "src/thread_safe.h"

using namespace std::literals;

#define __SV(quote) L##quote##sv
#define SV(quote) __SV(quote)

#define SERVICE_DOMAIN "local"

namespace platf::publish {
  constexpr auto SERVICE_TYPE_DOMAIN = SV(SERVICE_TYPE "." SERVICE_DOMAIN);

  VOID WINAPI register_cb(DWORD status, PVOID pQueryContext, PDNS_SERVICE_INSTANCE pInstance) {
    auto alarm = (safe::alarm_t<PDNS_SERVICE_INSTANCE>::element_type *) pQueryContext;

    if (status) {
      print_status("register_cb()"sv, status);
    }

    alarm->ring(pInstance);
  }

  static int service(bool enable, PDNS_SERVICE_INSTANCE &existing_instance) {
    auto alarm = safe::make_alarm<PDNS_SERVICE_INSTANCE>();

    std::wstring domain {SERVICE_TYPE_DOMAIN.data(), SERVICE_TYPE_DOMAIN.size()};

    auto hostname = platf::get_host_name();
    auto name = from_utf8(net::mdns_instance_name(hostname) + '.') + domain;
    auto host = from_utf8(hostname + ".local");

    DNS_SERVICE_INSTANCE instance {};
    instance.pszInstanceName = name.data();
    instance.wPort = net::map_port(nvhttp::PORT_HTTP);
    instance.pszHostName = host.data();

    // Setting these values ensures Windows mDNS answers comply with RFC 1035.
    // If these are unset, Windows will send a TXT record that has zero strings,
    // which is illegal. Setting them to a single empty value causes Windows to
    // send a single empty string for the TXT record, which is the correct thing
    // to do when advertising a service without any TXT strings.
    //
    // Most clients aren't strictly checking TXT record compliance with RFC 1035,
    // but Apple's mDNS resolver does and rejects the entire answer if an invalid
    // TXT record is present.
    PWCHAR keys[] = {nullptr};
    PWCHAR values[] = {nullptr};
    instance.dwPropertyCount = 1;
    instance.keys = keys;
    instance.values = values;

    DNS_SERVICE_REGISTER_REQUEST req {};
    req.Version = DNS_QUERY_REQUEST_VERSION1;
    req.pQueryContext = alarm.get();
    req.pServiceInstance = enable ? &instance : existing_instance;
    req.pRegisterCompletionCallback = register_cb;

    DNS_STATUS status {};

    if (enable) {
      status = DnsServiceRegister(&req, nullptr);
      if (status != DNS_REQUEST_PENDING) {
        print_status("DnsServiceRegister()"sv, status);
        return -1;
      }
    } else {
      status = DnsServiceDeRegister(&req, nullptr);
      if (status != DNS_REQUEST_PENDING) {
        print_status("DnsServiceDeRegister()"sv, status);
        return -1;
      }
    }

    alarm->wait();

    auto registered_instance = alarm->status();
    if (enable) {
      // Store this instance for later deregistration
      existing_instance = registered_instance;
    } else if (registered_instance) {
      // Deregistration was successful
      DnsServiceFreeInstance(registered_instance);
      existing_instance = nullptr;
    }

    return registered_instance ? 0 : -1;
  }

  class mdns_registration_t: public ::platf::deinit_t {
  public:
    mdns_registration_t():
        existing_instance(nullptr) {
      if (service(true, existing_instance)) {
        BOOST_LOG(error) << "Unable to register Sunshine 3D mDNS service"sv;
        return;
      }

      BOOST_LOG(info) << "Registered Sunshine 3D mDNS service"sv;
    }

    ~mdns_registration_t() override {
      if (existing_instance) {
        if (service(false, existing_instance)) {
          BOOST_LOG(error) << "Unable to unregister Sunshine 3D mDNS service"sv;
          return;
        }

        BOOST_LOG(info) << "Unregistered Sunshine 3D mDNS service"sv;
      }
    }

  private:
    PDNS_SERVICE_INSTANCE existing_instance;
  };

  std::unique_ptr<::platf::deinit_t> start() {
    return std::make_unique<mdns_registration_t>();
  }
}  // namespace platf::publish
