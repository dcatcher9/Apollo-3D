#include "model_manager.h"
#include "logging.h"
#include <curl/curl.h>
#include <cstdio>
#include <system_error>

using namespace std::literals;

namespace models {

    static size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
        return fwrite(ptr, size, nmemb, stream);
    }

    std::string engine_filename(const config::depth_model_info& model) {
        return model.name + ".engine";
    }

    std::filesystem::path ensure_model_available(const std::filesystem::path& assets_dir, const std::string& model_name, const std::string& model_url, const std::string& engine_name) {
        // Files are named after the model (not the URL) so each model gets its own cached
        // engine: switching sbs_3d_depth_model never reuses a stale engine, and different
        // models coexist. To use another model, point sbs_3d_depth_model_url at its ONNX.
        // The engine is looked up under its recipe-specific name (see engine_filename()).
        auto onnx_path = assets_dir / (model_name + ".onnx");
        auto engine_path = assets_dir / (engine_name.empty() ? (model_name + ".engine") : engine_name);

        if (std::filesystem::exists(engine_path)) {
            return engine_path;
        }
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
}
