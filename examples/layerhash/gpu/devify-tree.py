"""devify-tree.py SRC_ROOT OUT_ROOT FILE...: copies of C sources and headers
with "__device__" before every function declaration or definition at
column 0 (an optional static/inline, a type of one or more words, a name
and "(", or static/inline and a type alone on their line), and before
every file-scope constant ("static const" / "const" with an initializer
or array) and mutable static variable, keeping their relative paths so that #include "..." still
resolves. typedefs, struct/enum lines and indented lines are untouched.
--init TEMPLATE (repeatable, before SRC_ROOT): a brace group of
designators only ({.hi = X, .lo = Y, ...}) whose names are exactly the
$names of one TEMPLATE becomes that TEMPLATE with the values put in.
CORE-MATH's dint.h and qint.h initialize the members of an anonymous
struct inside a union by designator, which C accepts and nvcc's C++ does not
in any order ("duplicate designator": each one starts the union again), so
the template initializes the union's first member by position instead
"""
import os, re, string, sys
args = sys.argv[1:]; inits = []
while args and args[0] == '--init':
    t = string.Template(args[1]); inits.append((t, sorted(m.group('named') for m in t.pattern.finditer(t.template) if m.group('named')))); args = args[2:]
src, out = args[0], args[1]
group = re.compile(r'\{(\s*\.\w+\s*=\s*[^,{};]+,)*\s*\.\w+\s*=\s*[^,{};]+\}')
def reorder(m):
    d = dict((i.split('=', 1)[0].strip()[1:], i.split('=', 1)[1].strip()) for i in m.group(0)[1:-1].split(','))
    for t, names in inits:
        if sorted(d) == names:
            return t.substitute(d)
    return m.group(0)
fn = re.compile(r'^((static |inline |__attribute__\(\([a-z_]+\)\) )*(?!typedef\b|return\b|struct\b|enum\b|union\b|else\b|if\b|for\b|while\b|switch\b|case\b|do\b)(?:[A-Za-z_][A-Za-z_0-9]*[ *]+)+[A-Za-z_][A-Za-z_0-9]* ?\()')
const = re.compile(r'^((static )?const [A-Za-z_].*[=\[].*)$')
var = re.compile(r'^(static (?!const\b|inline\b)[A-Za-z_][A-Za-z_0-9]*[ *]+[A-Za-z_][A-Za-z_0-9]* *(=|\[|;).*)$')   # a mutable file-scope static
# a definition whose specifiers and type stand alone on their line, the name and "(" on the next
# ("static inline void" / "add_dint (dint64_t *r, ..."; CORE-MATH's pow, dint.h, qint.h; 2026-10-10)
head = re.compile(r'^((static |inline )+[A-Za-z_][A-Za-z_0-9]*( [A-Za-z_][A-Za-z_0-9]*)*[ *]*)$')
for f in args[2:]:
    lines = group.sub(reorder, open(os.path.join(src, f)).read()).split('\n')
    o = []
    for l in lines:
        if fn.match(l) or const.match(l) or var.match(l) or head.match(l):
            l = '__device__ ' + l
        o.append(l)
    p = os.path.join(out, f); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w').write('\n'.join(o))
