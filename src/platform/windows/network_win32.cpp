/**
 * @file src/platform/windows/network_win32.cpp
 * @brief Network adapters, UDP sends, and QoS tagging on Windows.
 */
// standard includes
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

// lib includes
#include <boost/asio/ip/address.hpp>

// prevent clang format from "optimizing" the header include order
// clang-format off
#include <iphlpapi.h>
#include <WinSock2.h>
#include <Windows.h>
#include <WS2tcpip.h>
// clang-format on

// local includes
#include "misc.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/utility.h"

// UDP_SEND_MSG_SIZE was added in the Windows 10 20H1 SDK
#ifndef UDP_SEND_MSG_SIZE
  #define UDP_SEND_MSG_SIZE 2
#endif

#include <qos2.h>

using namespace std::literals;

namespace platf {
  using adapteraddrs_t = util::c_ptr<IP_ADAPTER_ADDRESSES>;

  HANDLE qos_handle = nullptr;

  std::string from_sockaddr(const sockaddr *const socket_address) {
    char data[INET6_ADDRSTRLEN] = {};

    auto family = socket_address->sa_family;
    if (family == AF_INET6) {
      inet_ntop(AF_INET6, &((sockaddr_in6 *) socket_address)->sin6_addr, data, INET6_ADDRSTRLEN);
    } else if (family == AF_INET) {
      inet_ntop(AF_INET, &((sockaddr_in *) socket_address)->sin_addr, data, INET_ADDRSTRLEN);
    }

    return std::string {data};
  }

  std::pair<std::uint16_t, std::string> from_sockaddr_ex(const sockaddr *const ip_addr) {
    char data[INET6_ADDRSTRLEN] = {};

    auto family = ip_addr->sa_family;
    std::uint16_t port = 0;
    if (family == AF_INET6) {
      inet_ntop(AF_INET6, &((sockaddr_in6 *) ip_addr)->sin6_addr, data, INET6_ADDRSTRLEN);
      port = ((sockaddr_in6 *) ip_addr)->sin6_port;
    } else if (family == AF_INET) {
      inet_ntop(AF_INET, &((sockaddr_in *) ip_addr)->sin_addr, data, INET_ADDRSTRLEN);
      port = ((sockaddr_in *) ip_addr)->sin_port;
    }

    return {port, std::string {data}};
  }

  adapteraddrs_t get_adapteraddrs() {
    adapteraddrs_t info {nullptr};
    ULONG size = 0;

    while (GetAdaptersAddresses(AF_UNSPEC, 0, nullptr, info.get(), &size) == ERROR_BUFFER_OVERFLOW) {
      info.reset((PIP_ADAPTER_ADDRESSES) malloc(size));
    }

    return info;
  }

  std::string get_mac_address(const std::string_view &address) {
    adapteraddrs_t info = get_adapteraddrs();
    for (auto adapter_pos = info.get(); adapter_pos != nullptr; adapter_pos = adapter_pos->Next) {
      for (auto addr_pos = adapter_pos->FirstUnicastAddress; addr_pos != nullptr; addr_pos = addr_pos->Next) {
        if (adapter_pos->PhysicalAddressLength != 0 && address == from_sockaddr(addr_pos->Address.lpSockaddr)) {
          std::stringstream mac_addr;
          mac_addr << std::hex;
          for (int i = 0; i < adapter_pos->PhysicalAddressLength; i++) {
            if (i > 0) {
              mac_addr << ':';
            }
            mac_addr << std::setw(2) << std::setfill('0') << (int) adapter_pos->PhysicalAddress[i];
          }
          return mac_addr.str();
        }
      }
    }
    BOOST_LOG(debug) << "Unable to find MAC address for "sv << address << ", is this a virtual network adapter?";
    return "00:00:00:00:00:00"s;
  }

  std::string get_local_ip_for_gateway() {
    PIP_ADAPTER_INFO pAdapterInfo;
    PIP_ADAPTER_INFO pAdapter = nullptr;
    DWORD dwRetVal = 0;
    ULONG ulOutBufLen = sizeof(IP_ADAPTER_INFO);

    pAdapterInfo = (IP_ADAPTER_INFO *)malloc(sizeof(IP_ADAPTER_INFO));
    if (pAdapterInfo == nullptr) {
      BOOST_LOG(warning) << "Error allocating memory needed to call GetAdaptersInfo";
      return "";
    }

    // Make an initial call to GetAdaptersInfo to get the necessary size into the ulOutBufLen variable
    if (GetAdaptersInfo(pAdapterInfo, &ulOutBufLen) == ERROR_BUFFER_OVERFLOW) {
      free(pAdapterInfo);
      pAdapterInfo = (IP_ADAPTER_INFO *)malloc(ulOutBufLen);
      if (pAdapterInfo == nullptr) {
        BOOST_LOG(warning) << "Error allocating memory needed to call GetAdaptersInfo";
        return "";
      }
    }

    if ((dwRetVal = GetAdaptersInfo(pAdapterInfo, &ulOutBufLen)) != NO_ERROR) {
      if (pAdapterInfo) {
        free(pAdapterInfo);
      }
      BOOST_LOG(warning) << "GetAdaptersInfo failed with error: " + std::to_string(dwRetVal);
      return "";
    }

    pAdapter = pAdapterInfo;
    std::string local_ip;

    // Iterate through the list of adapters
    while (pAdapter) {
      IP_ADDR_STRING* pGateway = &pAdapter->GatewayList;
      if (pGateway && pGateway->IpAddress.String[0] != '\0') {
        // This adapter has a default gateway, use its IP address
        local_ip = pAdapter->IpAddressList.IpAddress.String;
        break;
      }
      pAdapter = pAdapter->Next;
    }

    if (pAdapterInfo) {
      free(pAdapterInfo);
    }

    if (local_ip.empty()) {
      BOOST_LOG(warning) << "No associated IP address found for the default gateway";
    }

    return local_ip;
  }

  SOCKADDR_IN to_sockaddr(boost::asio::ip::address_v4 address, uint16_t port) {
    SOCKADDR_IN saddr_v4 = {};

    saddr_v4.sin_family = AF_INET;
    saddr_v4.sin_port = htons(port);

    auto addr_bytes = address.to_bytes();
    memcpy(&saddr_v4.sin_addr, addr_bytes.data(), sizeof(saddr_v4.sin_addr));

    return saddr_v4;
  }

  SOCKADDR_IN6 to_sockaddr(boost::asio::ip::address_v6 address, uint16_t port) {
    SOCKADDR_IN6 saddr_v6 = {};

    saddr_v6.sin6_family = AF_INET6;
    saddr_v6.sin6_port = htons(port);
    saddr_v6.sin6_scope_id = address.scope_id();

    auto addr_bytes = address.to_bytes();
    memcpy(&saddr_v6.sin6_addr, addr_bytes.data(), sizeof(saddr_v6.sin6_addr));

    return saddr_v6;
  }

  static bool send_udp_datagrams(
    std::uintptr_t socket,
    const boost::asio::ip::address &target_address,
    uint16_t target_port,
    const boost::asio::ip::address &source_address,
    WSABUF *buffers,
    DWORD buffer_count,
    DWORD segment_size
  ) {
    WSAMSG msg {};
    SOCKADDR_IN target_v4;
    SOCKADDR_IN6 target_v6;
    if (target_address.is_v6()) {
      target_v6 = to_sockaddr(target_address.to_v6(), target_port);
      msg.name = reinterpret_cast<SOCKADDR *>(&target_v6);
      msg.namelen = sizeof(target_v6);
    } else {
      target_v4 = to_sockaddr(target_address.to_v4(), target_port);
      msg.name = reinterpret_cast<SOCKADDR *>(&target_v4);
      msg.namelen = sizeof(target_v4);
    }

    msg.lpBuffers = buffers;
    msg.dwBufferCount = buffer_count;

    // One packet-info option and space for an optional UDP segmentation size.
    alignas(WSACMSGHDR) char control[WSA_CMSG_SPACE(sizeof(DWORD)) + std::max(WSA_CMSG_SPACE(sizeof(IN6_PKTINFO)), WSA_CMSG_SPACE(sizeof(IN_PKTINFO)))] = {};
    msg.Control.buf = control;
    msg.Control.len = sizeof(control);

    auto *option = WSA_CMSG_FIRSTHDR(&msg);
    ULONG control_size;
    if (source_address.is_v6()) {
      IN6_PKTINFO packet_info {};
      packet_info.ipi6_addr = to_sockaddr(source_address.to_v6(), 0).sin6_addr;
      control_size = WSA_CMSG_SPACE(sizeof(packet_info));
      option->cmsg_level = IPPROTO_IPV6;
      option->cmsg_type = IPV6_PKTINFO;
      option->cmsg_len = WSA_CMSG_LEN(sizeof(packet_info));
      std::memcpy(WSA_CMSG_DATA(option), &packet_info, sizeof(packet_info));
    } else {
      IN_PKTINFO packet_info {};
      packet_info.ipi_addr = to_sockaddr(source_address.to_v4(), 0).sin_addr;
      control_size = WSA_CMSG_SPACE(sizeof(packet_info));
      option->cmsg_level = IPPROTO_IP;
      option->cmsg_type = IP_PKTINFO;
      option->cmsg_len = WSA_CMSG_LEN(sizeof(packet_info));
      std::memcpy(WSA_CMSG_DATA(option), &packet_info, sizeof(packet_info));
    }

    if (segment_size != 0) {
      control_size += WSA_CMSG_SPACE(sizeof(segment_size));
      option = WSA_CMSG_NXTHDR(&msg, option);
      option->cmsg_level = IPPROTO_UDP;
      option->cmsg_type = UDP_SEND_MSG_SIZE;
      option->cmsg_len = WSA_CMSG_LEN(sizeof(segment_size));
      std::memcpy(WSA_CMSG_DATA(option), &segment_size, sizeof(segment_size));
    }

    msg.Control.len = control_size;
    DWORD bytes_sent;
    return WSASendMsg(static_cast<SOCKET>(socket), &msg, 0, &bytes_sent, nullptr, nullptr) != SOCKET_ERROR;
  }

  bool send_batch(const batched_send_info_t &send_info) {
    if (send_info.packet_count == 0 || send_info.packet_count > MAX_UDP_BATCH_PACKETS) {
      return false;
    }

    WSABUF buffers[2 * MAX_UDP_BATCH_PACKETS];
    const auto header_size = static_cast<ULONG>(send_info.header_size);
    const auto payload_size = static_cast<ULONG>(send_info.payloads.payload_size);
    for (size_t index = 0; index < send_info.packet_count; ++index) {
      buffers[2 * index] = {header_size, const_cast<char *>(send_info.header_at(index))};
      buffers[2 * index + 1] = {payload_size, const_cast<char *>(send_info.payload_at(index))};
    }

    // If segmentation offload is unavailable, the caller sends these packets individually.
    return send_udp_datagrams(
      send_info.native_socket,
      send_info.target_address,
      send_info.target_port,
      send_info.source_address,
      buffers,
      static_cast<DWORD>(2 * send_info.packet_count),
      send_info.packet_count > 1 ? header_size + payload_size : 0
    );
  }

  bool send(send_info_t &send_info) {
    WSABUF buffers[] {
      {static_cast<ULONG>(send_info.header_size), const_cast<char *>(send_info.header)},
      {static_cast<ULONG>(send_info.payload_size), const_cast<char *>(send_info.payload)},
    };
    const auto first = send_info.header ? 0u : 1u;
    if (!send_udp_datagrams(
          send_info.native_socket,
          send_info.target_address,
          send_info.target_port,
          send_info.source_address,
          buffers + first,
          2 - first,
          0
        )) {
      auto winerr = WSAGetLastError();
      BOOST_LOG(warning) << "WSASendMsg() failed: "sv << winerr;
      return false;
    }

    return true;
  }

  class qos_t: public deinit_t {
  public:
    qos_t(QOS_FLOWID flow_id):
        flow_id(flow_id) {
    }

    virtual ~qos_t() {
      if (!QOSRemoveSocketFromFlow(qos_handle, (SOCKET) nullptr, flow_id, 0)) {
        auto winerr = GetLastError();
        BOOST_LOG(warning) << "QOSRemoveSocketFromFlow() failed: "sv << winerr;
      }
    }

  private:
    QOS_FLOWID flow_id;
  };

  /**
   * @brief Enables QoS on the given socket for traffic to the specified destination.
   * @param native_socket The native socket handle.
   * @param address The destination address for traffic sent on this socket.
   * @param port The destination port for traffic sent on this socket.
   * @param data_type The type of traffic sent on this socket.
   * @param dscp_tagging Specifies whether to enable DSCP tagging on outgoing traffic.
   */
  std::unique_ptr<deinit_t> enable_socket_qos(uintptr_t native_socket, boost::asio::ip::address &address, uint16_t port, qos_data_type_e data_type, bool dscp_tagging) {
    SOCKADDR_IN saddr_v4;
    SOCKADDR_IN6 saddr_v6;
    PSOCKADDR dest_addr;
    bool using_connect_hack = false;

    // Windows doesn't support any concept of traffic priority without DSCP tagging
    if (!dscp_tagging) {
      return nullptr;
    }

    static std::once_flag qos_init_once_flag;
    std::call_once(qos_init_once_flag, []() {
      QOS_VERSION qos_version {1, 0};
      if (!QOSCreateHandle(&qos_version, &qos_handle)) {
        auto winerr = GetLastError();
        BOOST_LOG(warning) << "QOSCreateHandle() failed: "sv << winerr;
        return;
      }
    });

    // A failed service initialization leaves QoS disabled.
    if (!qos_handle) {
      return nullptr;
    }

    auto disconnect_fg = util::fail_guard([&]() {
      if (using_connect_hack) {
        SOCKADDR_IN6 empty = {};
        empty.sin6_family = AF_INET6;
        if (connect((SOCKET) native_socket, (PSOCKADDR) &empty, sizeof(empty)) < 0) {
          auto wsaerr = WSAGetLastError();
          BOOST_LOG(error) << "qWAVE dual-stack workaround failed: "sv << wsaerr;
        }
      }
    });

    if (address.is_v6()) {
      auto address_v6 = address.to_v6();

      saddr_v6 = to_sockaddr(address_v6, port);
      dest_addr = (PSOCKADDR) &saddr_v6;

      // qWAVE doesn't properly support IPv4-mapped IPv6 addresses, nor does it
      // correctly support IPv4 addresses on a dual-stack socket (despite MSDN's
      // claims to the contrary). To get proper QoS tagging when hosting in dual
      // stack mode, we will temporarily connect() the socket to allow qWAVE to
      // successfully initialize a flow, then disconnect it again so WSASendMsg()
      // works later on.
      if (address_v6.is_v4_mapped()) {
        if (connect((SOCKET) native_socket, (PSOCKADDR) &saddr_v6, sizeof(saddr_v6)) < 0) {
          auto wsaerr = WSAGetLastError();
          BOOST_LOG(error) << "qWAVE dual-stack workaround failed: "sv << wsaerr;
        } else {
          BOOST_LOG(debug) << "Using qWAVE connect() workaround for QoS tagging"sv;
          using_connect_hack = true;
          dest_addr = nullptr;
        }
      }
    } else {
      saddr_v4 = to_sockaddr(address.to_v4(), port);
      dest_addr = (PSOCKADDR) &saddr_v4;
    }

    QOS_TRAFFIC_TYPE traffic_type;
    switch (data_type) {
      case qos_data_type_e::audio:
        traffic_type = QOSTrafficTypeVoice;
        break;
      case qos_data_type_e::video:
        traffic_type = QOSTrafficTypeAudioVideo;
        break;
      default:
        BOOST_LOG(error) << "Unknown traffic type: "sv << (int) data_type;
        return nullptr;
    }

    QOS_FLOWID flow_id = 0;
    if (!QOSAddSocketToFlow(qos_handle, (SOCKET) native_socket, dest_addr, traffic_type, QOS_NON_ADAPTIVE_FLOW, &flow_id)) {
      auto winerr = GetLastError();
      BOOST_LOG(warning) << "QOSAddSocketToFlow() failed: "sv << winerr;
      return nullptr;
    }

    return std::make_unique<qos_t>(flow_id);
  }

  std::string get_host_name() {
    WCHAR hostname[256];
    if (GetHostNameW(hostname, ARRAYSIZE(hostname)) == SOCKET_ERROR) {
      BOOST_LOG(error) << "GetHostNameW() failed: "sv << WSAGetLastError();
      return "Sunshine"s;
    }
    return to_utf8(hostname);
  }
}  // namespace platf
