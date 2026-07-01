import os, glob, re

for file in glob.glob('E:/TensorRT-11.1.0.106/include/*.h'):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if '#if defined(__GNUC__)' in line and i+4 < len(lines) and 'msvc_dummy_destructor' in lines[i+1]:
            orig_line = lines[i+3]
            out_lines.append(orig_line)
            i += 5
            continue
        if '#if !defined(__GNUC__)' in line and i+2 < len(lines) and '::~' in lines[i+1]:
            orig_line = lines[i+1]
            out_lines.append(orig_line)
            i += 3
            continue
        out_lines.append(line)
        i += 1
        
    content = "".join(out_lines)
    
    def repl(m):
        decl = m.group(0).strip()
        spaces = m.group(1)
        if '= 0' in decl:
            return f"#if defined(__GNUC__)\n{spaces}virtual void msvc_dummy_destructor(char flags) = 0;\n#else\n{spaces}{decl}\n#endif\n"
        else:
            return f"#if defined(__GNUC__)\n{spaces}virtual void msvc_dummy_destructor(char flags) {{}}\n#else\n{spaces}{decl}\n#endif\n"

    new_content = re.sub(r'^([ \t]*)virtual ~[a-zA-Z0-9_]+\(\)(?: noexcept)? = (?:0|default);\s*$', repl, content, flags=re.MULTILINE)
    
    def repl_inline(m):
        decl = m.group(0).strip()
        return f"#if !defined(__GNUC__)\n{decl}\n#endif\n"
        
    new_content = re.sub(r'^inline [a-zA-Z0-9_]+::~[a-zA-Z0-9_]+\(\)(?: noexcept)? = default;\s*$', repl_inline, new_content, flags=re.MULTILINE)
    
    if new_content != content:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Patched {file}")
