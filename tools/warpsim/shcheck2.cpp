#include <windows.h>
#include <d3dcompiler.h>
#include <cstdio>
int main(int argc, char** argv){
  if(argc < 4){ printf("usage: shcheck2 <file> <target> <entry>\n"); return 2; }
  wchar_t wpath[1024]; MultiByteToWideChar(CP_UTF8, 0, argv[1], -1, wpath, 1024);
  ID3DBlob *code=nullptr,*err=nullptr;
  HRESULT hr=D3DCompileFromFile(wpath, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
    argv[3], argv[2], D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
  if(err) printf("%s\n",(char*)err->GetBufferPointer());
  if(FAILED(hr)){ printf("FAILED %s hr=0x%08lx\n", argv[1], (unsigned long)hr); return 1; }
  printf("OK %s\n", argv[1]); return 0;
}
