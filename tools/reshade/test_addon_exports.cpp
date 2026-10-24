// SPDX-License-Identifier: GPL-3.0-only
// The add-ons' export tables, read from their mapped images without running
// any of their code. The shipping SunshineSBS.addon64 exports exactly
// ReShade's add-on entry points. Test entry points, the API contract matrix's
// structured refusal query (SunshineDepthTestLastRefusal) among them, exist
// only in SunshineSBSTest.addon64 (SUNSHINE_SBS_RUNTIME_TEST_ADDON) and name
// themselves as tests.
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <Windows.h>

namespace {
  bool exported_names(const char *path, std::set<std::string> &out) {
    // An image resource mapping lays the sections out at their RVAs and runs
    // no DllMain; its handle carries a flag in the low bits.
    const auto module = LoadLibraryExA(path, nullptr, LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    if (!module) {
      std::fprintf(stderr, "Cannot map %s: %lu\n", path, GetLastError());
      return false;
    }
    const auto *base = reinterpret_cast<const std::uint8_t *>(reinterpret_cast<std::uintptr_t>(module) & ~std::uintptr_t(3));
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS64 *>(base + dos->e_lfanew);
    bool valid = dos->e_magic == IMAGE_DOS_SIGNATURE && nt->Signature == IMAGE_NT_SIGNATURE && nt->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    if (valid) {
      const auto &directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
      if (directory.VirtualAddress && directory.Size) {
        const auto *table = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY *>(base + directory.VirtualAddress);
        const auto *names = reinterpret_cast<const DWORD *>(base + table->AddressOfNames);
        for (DWORD i = 0; i != table->NumberOfNames; ++i) {
          out.insert(reinterpret_cast<const char *>(base + names[i]));
        }
      }
    } else {
      std::fprintf(stderr, "%s is not a 64-bit PE image\n", path);
    }
    FreeLibrary(module);
    return valid;
  }

  std::string joined(const std::set<std::string> &names) {
    std::string out;
    for (const auto &name : names) {
      out += out.empty() ? "" : " ";
      out += name;
    }
    return out;
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::fprintf(stderr, "Usage: reshade_addon_exports_tests <SunshineSBS.addon64> [<SunshineSBSTest.addon64>]\n");
    return 2;
  }
  std::set<std::string> shipping;
  if (!exported_names(argv[1], shipping)) {
    return 2;
  }
  // reshade.hpp: NAME and DESCRIPTION, AddonInit and AddonUninit.
  const std::set<std::string> addon_contract {"AddonInit", "AddonUninit", "DESCRIPTION", "NAME"};
  if (shipping != addon_contract) {
    std::fprintf(stderr, "FAIL the shipping add-on exports %s, not exactly ReShade's add-on entry points\n", joined(shipping).c_str());
    return 1;
  }
  if (argc == 2) {
    std::puts("PASS the shipping add-on exports only ReShade's add-on entry points (no test add-on built)");
    return 0;
  }
  std::set<std::string> test;
  if (!exported_names(argv[2], test)) {
    return 2;
  }
  std::set<std::string> extra;
  for (const auto &name : addon_contract) {
    if (!test.count(name)) {
      std::fprintf(stderr, "FAIL the test add-on lacks the add-on entry point %s\n", name.c_str());
      return 1;
    }
  }
  for (const auto &name : test) {
    if (addon_contract.count(name)) {
      continue;
    }
    if (name.find("Test") == std::string::npos) {
      std::fprintf(stderr, "FAIL the test add-on export %s does not name itself as a test entry point\n", name.c_str());
      return 1;
    }
    extra.insert(name);
  }
  if (!extra.count("SunshineDepthTestLastRefusal")) {
    std::fprintf(stderr, "FAIL the test add-on lacks the structured refusal query SunshineDepthTestLastRefusal\n");
    return 1;
  }
  std::printf(
    "PASS the shipping add-on exports only ReShade's add-on entry points; the test add-on adds %zu test entry points, "
    "SunshineDepthTestLastRefusal among them\n",
    extra.size()
  );
  return 0;
}
