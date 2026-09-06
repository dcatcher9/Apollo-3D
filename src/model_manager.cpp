#include "model_manager.h"
#include "logging.h"
#include <curl/curl.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <system_error>
#include <unordered_map>

using namespace std::literals;

namespace models {

    static size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
        return fwrite(ptr, size, nmemb, stream);
    }

    static std::string engine_identity_digest(
        std::string_view domain, std::string_view model, std::string_view recipe,
        std::string_view compatibility_tag) {
        // Encoding the complete identity verbatim in a Windows path can exceed MAX_PATH once
        // `.part` is appended during atomic publication. The contract retains the readable
        // recipe, while the active-engine manifest publishes this digest-bearing filename and
        // the authenticated ONNX hash. Length-framed fields make the cache identity unambiguous;
        // the domain prevents this digest from being confused with an unrelated SHA-256 use.
        using context_ptr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
        context_ptr context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest {};
        unsigned int digest_size = 0;
        if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
            return {};
        }

        const auto update_field = [&](std::string_view field) {
            std::array<unsigned char, sizeof(std::uint64_t)> encoded_size {};
            std::uint64_t remaining = static_cast<std::uint64_t>(field.size());
            for (auto byte = encoded_size.rbegin(); byte != encoded_size.rend(); ++byte) {
                *byte = static_cast<unsigned char>(remaining & 0xffu);
                remaining >>= 8u;
            }
            return EVP_DigestUpdate(
                     context.get(), encoded_size.data(), encoded_size.size()) == 1 &&
                   (field.empty() ||
                    EVP_DigestUpdate(context.get(), field.data(), field.size()) == 1);
        };
        if (!update_field(domain) ||
            !update_field(model) ||
            !update_field(recipe) ||
            !update_field(compatibility_tag) ||
            EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
            digest_size != 32u) {
            return {};
        }

        std::ostringstream token;
        token << std::hex << std::setfill('0');
        for (unsigned int i = 0; i < digest_size; ++i) {
            token << std::setw(2) << static_cast<unsigned int>(digest[i]);
        }
        return token.str();
    }

    std::string engine_filename(const config::depth_model_info& model, std::string_view compatibility_tag) {
        const auto prefix = model.name + "." + std::string(prod_zipdepth_convex2x::engine_recipe);
        // The empty-tag name is a preparation status key, never a selected disk cache artifact.
        if (compatibility_tag.empty()) {
            return prefix + ".engine";
        }
        const auto digest = engine_identity_digest(
            "sunshine3d/depth-tensorrt-engine-cache/v1", model.name,
            prod_zipdepth_convex2x::engine_recipe, compatibility_tag);
        return digest.empty() ? std::string {} : prefix + ".cache-" + digest + ".engine";
    }

    bool is_current_depth_engine_filename(std::string_view filename) {
        static const std::string prefix = std::string(prod_zipdepth_convex2x::logical_model) +
            "." + std::string(prod_zipdepth_convex2x::engine_recipe) + ".cache-";
        constexpr std::string_view suffix = ".engine";
        if (filename.size() != prefix.size() + 64u + suffix.size() ||
            !filename.starts_with(prefix) || !filename.ends_with(suffix)) {
            return false;
        }
        const auto digest = filename.substr(prefix.size(), 64u);
        return std::all_of(digest.begin(), digest.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
    }

    std::string ocr_engine_filename(std::string_view compatibility_tag) {
        const auto digest = engine_identity_digest(
            "sunshine3d/ocr-tensorrt-engine-cache/v1", ocr_model_name,
            ocr_engine_recipe, compatibility_tag);
        return digest.empty() ? std::string {} :
            std::string(ocr_model_name) + ".cache-" + digest + ".engine";
    }

    bool engine_cache_path_fits_native_limit(const std::filesystem::path& path) {
        if (path.empty()) return false;
#ifdef _WIN32
        std::error_code ec;
        const auto absolute = std::filesystem::absolute(path, ec);
        // The native builder's file APIs do not opt into extended-length paths.
        return !ec && absolute.native().size() + std::string_view(".part").size() < 260u;
#else
        return true;
#endif
    }

    std::filesystem::path ensure_onnx_available(const std::filesystem::path& assets_dir, const std::string& model_name, const std::string& model_url) {
        // The ONNX source is retained even after compilation. Its content hash is part of the
        // engine filename, so replacing a local model under the same logical name cannot silently
        // reuse tactics built for different weights or graph structure.
        auto onnx_path = assets_dir / (model_name + ".onnx");
        if (std::filesystem::exists(onnx_path)) {
            return onnx_path;
        }
        if (model_url.empty()) {
            BOOST_LOG(warning) << "Model '" << model_name << "' not found in assets and no download URL configured.";
            return "";
        }

        // Download to a .part file and rename into place only on success, so an interrupted
        // download never leaves a truncated .onnx that would be treated as valid forever
        // (and a concurrent download of the same model can't corrupt the final file).
        auto part_path = onnx_path;
        part_path += ".part";

        auto download_file = [&](const std::string& url, const std::filesystem::path& path) -> bool {
            CURL* curl = curl_easy_init();
            if (!curl) return false;

            FILE* fp = fopen(path.string().c_str(), "wb");
            if (!fp) {
                curl_easy_cleanup(curl);
                return false;
            }

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);  // treat HTTP >= 400 as failure, don't save an error page

            CURLcode res = curl_easy_perform(curl);
            fclose(fp);
            curl_easy_cleanup(curl);

            if (res != CURLE_OK) {
                BOOST_LOG(error) << "Failed to download " << path.filename() << ": " << curl_easy_strerror(res);
                std::filesystem::remove(path);
                return false;
            }
            return true;
        };

        BOOST_LOG(info) << "Depth model '" << model_name << "' not found. Downloading from " << model_url << " (this may take a minute)...";
        if (!download_file(model_url, part_path)) {
            return "";
        }

        std::error_code ec;
        std::filesystem::rename(part_path, onnx_path, ec);
        if (ec) {
            BOOST_LOG(error) << "Failed to move downloaded model into place: " << ec.message();
            std::filesystem::remove(part_path, ec);
            return "";
        }

        BOOST_LOG(info) << "Successfully downloaded depth model '" << model_name << "'.";
        return onnx_path;
    }

    std::string file_sha256_hex(const std::filesystem::path& path) {
        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(path, ec);
        if (ec) {
            return {};
        }
        const auto size = std::filesystem::file_size(canonical, ec);
        if (ec) {
            return {};
        }
        const auto write_time = std::filesystem::last_write_time(canonical, ec);
        if (ec) {
            return {};
        }
        const std::string cache_key = canonical.generic_string() + "\n" +
                                      std::to_string(size) + "\n" +
                                      std::to_string(write_time.time_since_epoch().count());
        static std::mutex digest_mutex;
        static std::unordered_map<std::string, std::string> digest_cache;
        std::lock_guard<std::mutex> digest_lock(digest_mutex);
        if (const auto found = digest_cache.find(cache_key); found != digest_cache.end()) {
            return found->second;
        }

        using context_ptr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
        context_ptr context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
        if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
            return {};
        }

        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            return {};
        }
        std::array<char, 64 * 1024> block {};
        while (stream) {
            stream.read(block.data(), block.size());
            const auto count = stream.gcount();
            if (count > 0 && EVP_DigestUpdate(context.get(), block.data(), (std::size_t) count) != 1) {
                return {};
            }
        }
        if (!stream.eof()) {
            return {};
        }

        std::array<unsigned char, EVP_MAX_MD_SIZE> digest {};
        unsigned int digest_size = 0;
        if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1) {
            return {};
        }
        std::ostringstream hex;
        hex << std::hex << std::setfill('0');
        for (unsigned int i = 0; i < digest_size; ++i) {
            hex << std::setw(2) << (unsigned int) digest[i];
        }
        auto result = hex.str();
        digest_cache.emplace(cache_key, result);
        return result;
    }
}
