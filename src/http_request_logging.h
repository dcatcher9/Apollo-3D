/**
 * @file src/http_request_logging.h
 * @brief Shared, admitted debug logging for HTTP requests.
 */
#pragma once

#include "httpcommon.h"
#include "logging.h"

#include <boost/log/sources/record_ostream.hpp>
#include <optional>
#include <string_view>
#include <utility>

namespace http {
  template<class Request>
  void log_request_debug(const Request &request, std::optional<std::string_view> tunnel = std::nullopt) {
    auto first_record = debug.open_record();
    if (!first_record) {
      return;
    }

    // Use the admitted record for the first real line, rather than emitting an empty probe.
    {
      boost::log::record_ostream output(first_record);
      if (tunnel) {
        output << "TUNNEL :: " << *tunnel;
      } else {
        output << "METHOD :: " << request.method;
      }
      output.flush();
    }
    debug.push_record(std::move(first_record));

    if (tunnel) {
      BOOST_LOG(debug) << "METHOD :: " << request.method;
    }
    BOOST_LOG(debug) << "DESTINATION :: " << request.path;
    for (const auto &[name, value] : request.header) {
      BOOST_LOG(debug) << name << " -- " << redact_request_header(name, value);
    }
    BOOST_LOG(debug) << " [--] ";
    for (const auto &[name, value] : request.parse_query_string()) {
      BOOST_LOG(debug) << name << " -- " << redact_query_parameter(name, value);
    }
    BOOST_LOG(debug) << " [--] ";
  }
}  // namespace http
