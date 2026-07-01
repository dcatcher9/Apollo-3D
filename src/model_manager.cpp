#include "model_manager.h"
#include "logging.h"
#include <curl/curl.h>
#include <fstream>
#include <cstdio>

using namespace std::literals;

namespace models {

    static size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
        return fwrite(ptr, size, nmemb, stream);
    }

    std::filesystem::path ensure_model_available(const std::filesystem::path& assets_dir, const std::string& model_name, const std::string& model_url) {
        // Files are named after the model (not the URL) so each model gets its own cached
        // engine: switching sbs_3d_depth_model never reuses a stale engine, and different
        // models coexist. To use another model, point sbs_3d_depth_model_url at its ONNX.
        auto onnx_path = assets_dir / (model_name + ".onnx");
        auto engine_path = assets_dir / (model_name + ".engine");

        if (std::filesystem::exists(engine_path)) {
            return engine_path;
        }
        if (std::filesystem::exists(onnx_path)) {
            return onnx_path;
        }

        BOOST_LOG(info) << "Depth model '" << model_name << "' not found. Downloading from " << model_url << " ...";

        const std::string& url = model_url;


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
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
            
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

        BOOST_LOG(info) << "Downloading " << model_name << ".onnx. This may take a minute...";
        if (!download_file(url, onnx_path)) {
            return "";
        }

        BOOST_LOG(info) << "Successfully downloaded Depth Anything V2 model '" << model_name << "'.";
        return onnx_path;
    }
}
