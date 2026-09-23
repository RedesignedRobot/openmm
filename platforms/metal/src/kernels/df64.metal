// Double-float arithmetic for OpenMM's mixed type on Apple GPUs, which have no double precision.
// MetalContext places it right after the Metal standard library, followed by the mixed precision
// definitions that make mixed and double df64.
//
// A df64 is the unevaluated sum hi + lo of two floats with hi = RN(hi + lo) (a double-word number,
// Definition 1.4 of JMP 2017 below): 48 significand bits (49 counting the sign of lo) and the float
// exponent range. Every operation and the decoder return such pairs, and the operations and
// comparisons assume them. Apple GPUs flush float subnormals, so full precision holds down to
// |x| ~ 2^-102 (lo must stay normal); below that it degrades to float.
//
// Arithmetic follows Joldes, Muller, Popescu, "Tight and rigorous error bounds for basic building
// blocks of double-word arithmetic", ACM TOMS 44(2), 2017 (JMP), with the bounds as formally proven
// in Muller, Rideau, "Formalization of double-word arithmetic, and comments on ...", ACM TOMS 48(1),
// 2022 (MR). Relative error bounds, u = 2^-24, valid while no intermediate over- or underflows:
//     df64 + df64  AccurateDWPlusDW (JMP Alg. 6)   3u^2 + 13u^3 (JMP Thm 3.1)
//     df64 * df64  DWTimesDW3 (JMP Alg. 12)        4u^2 (MR Thm 2.8; JMP Thm 5.4 gave 5u^2)
//     df64 / df64  DWDivDW2 (JMP Alg. 17, with DWTimesFP1 at line 2 as stated) 15u^2 + 56u^3 (JMP Thm 7.1)
//     df64 + float DWPlusFP (JMP Alg. 4)           2u^2 (JMP Thm 2.2)
//     df64 * float DWTimesFP3 (JMP Alg. 9)         2u^2 (JMP Thm 4.3)
//     df64 / float DWDivFP3 (JMP Alg. 15)          3u^2 (JMP Thm 6.2)
// sqrt is one correction of the float square root (SQRTDWtoDW in Lefevre, Louvet, Muller, Picot,
// Rideau, ACM TOMS 2023, bound 25/8 u^2). exp and log are Taylor/Newton constructions on top.
//
// Compile with MTLCompileOptions.mathMode = .safe. Relaxed and fast math allow reassociation,
// which folds the error terms of the transformations below to zero.
//
// Storage. Device and constant memory hold df64 values as IEEE-754 binary64: every load from device
// or constant memory decodes and every store encodes, so host code that reads and writes doubles,
// and kernel arguments passed as doubles, work unchanged. Thread and threadgroup memory hold (hi, lo).

// ---------------------------------------------------------------------------------------------
// Error-free transformations. Each returns (result, exact error) as float2.

inline float2 df64_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}

// Requires |a| >= |b| or a == 0.
inline float2 df64_fast_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}

inline float2 df64_two_prod(float a, float b) {
    float p = a * b;
    return float2(p, fma(a, b, -p));
}

// ---------------------------------------------------------------------------------------------
// IEEE-754 binary64 bit patterns (uint2: .x low word, .y high word) to and from df64.

// RN(+-m * 2^e) as a float over the full float range: subnormals, underflow to zero, overflow to inf.
inline float df64_round_to_float(bool negative, ulong m, int e) {
    uint sign = negative ? 0x80000000u : 0u;
    if (m == 0)
        return as_type<float>(sign);
    int msb = 63 - (int) clz(m);
    int exponent = msb + e;
    if (exponent > 127)
        return as_type<float>(sign | 0x7F800000u);
    int ulpExponent = max(exponent - 23, -149);
    int drop = ulpExponent - e;
    ulong r;
    if (drop <= 0)
        r = m << (-drop);
    else if (drop > msb + 1)
        r = 0;
    else {
        ulong q = drop == 64 ? 0 : m >> drop;
        ulong rest = drop == 64 ? m : m & ((1ul << drop) - 1);
        ulong halfway = 1ul << (drop - 1);
        r = q + ((rest > halfway || (rest == halfway && (q & 1))) ? 1 : 0);
    }
    // r <= 2^24 includes the implicit bit, so a rounding carry moves into the exponent (and to inf).
    return as_type<float>(sign | (((uint) (ulpExponent + 149) << 23) + (uint) r));
}

// Magnitude of a finite float as integer significand * 2^exponent.
inline void df64_decompose(float x, thread ulong& m, thread int& e) {
    uint b = as_type<uint>(x) & 0x7FFFFFFFu;
    uint biased = b >> 23;
    m = biased == 0 ? (b & 0x7FFFFFu) : ((b & 0x7FFFFFu) | 0x800000u);
    e = biased == 0 ? -149 : (int) biased - 150;
}

// lo = RN(d - hi) can round up to exactly half an ulp of hi. When hi is odd, hi + lo then rounds
// away from hi, so the pair would not be a double-word number: lo moves one float towards zero.
inline float df64_below_tie(float hi, float lo) {
    uint h = as_type<uint>(hi);
    uint biased = (h >> 23) & 0xFFu;
    // Bits of ulp(hi)/2 = 2^(biased - 151) as a float, normal or subnormal; 0 when not representable.
    uint halfUlp = biased > 24u ? (biased - 24u) << 23 : (biased >= 2u ? 1u << (biased - 2u) : 0u);
    bool tie = (h & 1u) != 0 && halfUlp != 0 && (as_type<uint>(lo) & 0x7FFFFFFFu) == halfUlp;
    return tie ? as_type<float>(as_type<uint>(lo) - 1u) : lo;
}

struct df64;
inline df64 df64_from_ieee(uint2 bits);
inline uint2 df64_to_ieee(df64 v);

// ---------------------------------------------------------------------------------------------
// The scalar type.

struct alignas(8) df64 {
    float hi, lo;

    df64() = default;
    df64() threadgroup = default;
    df64(float h, float l) : hi(h), lo(l) {}
    df64(float x) : hi(x), lo(0.0f) {}
    df64(int x) : df64((long) x) {}
    df64(uint x) : df64((long) x) {}
    df64(long x) {
        hi = (float) x;
        long rest = hi >= 0x1p63f ? (x - LONG_MAX) - 1 : x - (long) hi;
        lo = (float) rest;
    }
    df64(ulong x) {
        hi = (float) x;
        long rest = hi >= 0x1p64f ? (long) x : (long) (x - (ulong) hi);
        lo = (float) rest;
    }

    df64(const thread df64&) = default;
    thread df64& operator=(const thread df64&) thread = default;
    thread df64& operator=(const device df64& o) thread { return *this = df64(o); }
    thread df64& operator=(const constant df64& o) thread { return *this = df64(o); }
    thread df64& operator=(const threadgroup df64& o) thread { return *this = df64(o); }
    thread df64& operator=(const volatile threadgroup df64& o) thread { return *this = df64(o); }
    df64(const threadgroup df64& o) : hi(o.hi), lo(o.lo) {}
    df64(const volatile threadgroup df64& o) : hi(o.hi), lo(o.lo) {}
    // `x = cond ? volatileLocal[i] : 0` materializes a volatile thread temporary.
    df64(const volatile thread df64&& o) : hi(o.hi), lo(o.lo) {}
    thread df64& operator=(const volatile thread df64&& o) thread { hi = o.hi; lo = o.lo; return *this; }
    threadgroup df64& operator=(df64 v) threadgroup { hi = v.hi; lo = v.lo; return *this; }
    volatile threadgroup df64& operator=(df64 v) volatile threadgroup { hi = v.hi; lo = v.lo; return *this; }

    df64(const device df64& o) : df64(df64_load_ieee(reinterpret_cast<const device uint*>(&o))) {}
    df64(const constant df64& o) : df64(df64_load_ieee(reinterpret_cast<const constant uint*>(&o))) {}
    device df64& operator=(df64 v) device {
        uint2 b = df64_to_ieee(v);
        device uint* w = reinterpret_cast<device uint*>(this);
        w[0] = b.x;
        w[1] = b.y;
        return *this;
    }
    static df64 df64_load_ieee(const device uint* w) { return df64_from_ieee(uint2(w[0], w[1])); }
    static df64 df64_load_ieee(const constant uint* w) { return df64_from_ieee(uint2(w[0], w[1])); }

    // Implicit narrowing to float only, as `real x = mixedValue` needs. A plain operator float() would
    // also convert to int and make `cond ? mixedValue : 0` ambiguous.
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const thread { return hi; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const device { return df64(*this).hi; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const constant { return df64(*this).hi; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const threadgroup { return hi; }
    // Truncating casts, as (long) on a double (minimize.cc converts forces to fixed point this way).
    explicit operator long() const thread;
    explicit operator int() const thread { return (int) (long) *this; }

    // Compound assignment, defined for every address space a kernel can name a mixed lvalue in.
#define DF64_COMPOUND(AS) \
    AS df64& operator+=(df64 v) AS; \
    AS df64& operator-=(df64 v) AS; \
    AS df64& operator*=(df64 v) AS; \
    AS df64& operator/=(df64 v) AS;
    DF64_COMPOUND(thread)
    DF64_COMPOUND(device)
    DF64_COMPOUND(threadgroup)
    DF64_COMPOUND(volatile threadgroup)
#undef DF64_COMPOUND
};

inline df64 df64_from_ieee_slow(uint2 b) {
    bool negative = (b.y >> 31) != 0;
    uint biased = (b.y >> 20) & 0x7FFu;
    ulong fraction = ((ulong) (b.y & 0xFFFFFu) << 32) | b.x;
    uint sign = negative ? 0x80000000u : 0u;
    if (biased == 0x7FFu)
        return df64(as_type<float>(sign | (fraction == 0 ? 0x7F800000u : 0x7FC00000u | (uint) (fraction >> 29))), 0.0f);
    if (biased == 0) {
        // Zero, or a subnormal double: below half the smallest float subnormal, so hi and lo are signed zeros.
        float zero = as_type<float>(sign);
        return df64(zero, fraction == 0 ? 0.0f : zero);
    }
    ulong m = fraction | (1ul << 52);
    int e = (int) biased - 1075;
    float hi = df64_round_to_float(negative, m, e);
    if (isinf(hi))
        return df64(hi, 0.0f);
    ulong mh;
    int eh;
    df64_decompose(hi, mh, eh);
    // hi is m*2^e rounded to at most 24 bits, so eh >= e and (mh << (eh-e)) stays within 2^54.
    long rest = mh == 0 ? (long) m : (long) m - (long) (mh << (eh - e));
    if (rest == 0)
        return df64(hi, 0.0f);
    return df64(hi, df64_below_tie(hi, df64_round_to_float((rest < 0) != negative, (ulong) abs(rest), e)));
}

// hi = RN(d), and lo = RN(d - hi) unless that is half an ulp of an odd hi, where it is the next float
// towards zero, so hi = RN(hi + lo) always and (float) of the result is RN(d). NaN and +-inf give lo = 0.
inline df64 df64_from_ieee(uint2 b) {
    uint biased = (b.y >> 20) & 0x7FFu;
    // Fast path for unbiased exponents -74..127: hi is a normal float and lo is normal or zero.
    if (biased - 949u > 201u)
        return df64_from_ieee_slow(b);
    uint sign = b.y & 0x80000000u;
    uint top = ((b.y & 0xFFFFFu) << 3) | (b.x >> 29);
    uint low = b.x & 0x1FFFFFFFu;
    bool up = low > 0x10000000u || (low == 0x10000000u && (top & 1u));
    uint hiBits = ((biased - 896u) << 23) + top + (up ? 1u : 0u);
    if (hiBits >= 0x7F800000u)
        return df64(as_type<float>(sign | 0x7F800000u), 0.0f);
    int rest = up ? (int) low - 0x20000000 : (int) low;
    float scale = as_type<float>((biased - 948u) << 23);
    float hi = as_type<float>(sign | hiBits);
    return df64(hi, df64_below_tie(hi, (float) (sign ? -rest : rest) * scale));
}

inline uint2 df64_to_ieee_slow(float hi, float lo) {
    uint sign = as_type<uint>(hi) & 0x80000000u;
    ulong mh, ml;
    int eh, el;
    df64_decompose(hi, mh, eh);
    df64_decompose(lo, ml, el);
    // Place hi at bits 36..59 so rounding happens at least 6 bits above the sticky bit of lo.
    long a = (long) (mh << 36);
    int shift = el - (eh - 36);
    long b;
    if (ml == 0)
        b = 0;
    else if (shift >= 0)
        b = (long) (ml << shift);
    else if (shift > -64) {
        b = (long) (ml >> (-shift));
        if (((ulong) b << (-shift)) != ml)
            b |= 1;
    }
    else
        b = 1;
    bool subtract = ml != 0 && ((as_type<uint>(lo) & 0x80000000u) != sign);
    ulong m = (ulong) (subtract ? a - b : a + b);
    int msb = 63 - (int) clz(m);
    int drop = msb - 52;
    ulong r;
    if (drop <= 0)
        r = m << (-drop);
    else {
        ulong q = m >> drop;
        ulong rest = m & ((1ul << drop) - 1);
        ulong halfway = 1ul << (drop - 1);
        r = q + ((rest > halfway || (rest == halfway && (q & 1))) ? 1 : 0);
    }
    int exponent = eh - 36 + drop;
    ulong bits = ((ulong) sign << 32) | (((ulong) (exponent + 1074) << 52) + r);
    return uint2((uint) bits, (uint) (bits >> 32));
}

// RN(hi + lo) as a double, with the sign of hi kept when the value is zero. Bit-exact for every
// double-word number and for unnormalized pairs with |lo| <= |hi|. (0, lo) encodes lo, and a non-finite
// lo gives the float sum hi + lo (inf or NaN).
// Float arithmetic on Apple GPUs flushes subnormal operands and results to zero, so a subnormal lo,
// or a tiny hi whose renormalization error could be subnormal, takes the integer path. So does a pair
// whose float sum overflows although hi + lo is a finite double.
inline uint2 df64_to_ieee(df64 v) {
    float hi = v.hi;
    float lo = v.lo;
    if ((as_type<uint>(hi) & 0x7FFFFFFFu) == 0 && (as_type<uint>(lo) & 0x7FFFFFFFu) != 0) {
        hi = lo;
        lo = 0.0f;
    }
    else if (!isfinite(lo)) {
        hi += lo;
        lo = 0.0f;
    }
    uint loBits = as_type<uint>(lo);
    bool loSubnormal = (loBits & 0x7F800000u) == 0 && (loBits & 0x7FFFFFu) != 0;
    bool hiTiny = ((as_type<uint>(hi) >> 23) & 0xFFu) < 53u;
    float sum = hi + lo;
    bool sumOverflows = !isfinite(sum) && isfinite(hi);
    if (!loSubnormal && !hiTiny && !sumOverflows && lo != 0.0f && sum != hi) {
        float2 s = df64_two_sum(hi, lo);
        hi = s.x;
        lo = s.y;
    }
    uint hiBits = as_type<uint>(hi);
    uint sign = hiBits & 0x80000000u;
    uint biased = (hiBits >> 23) & 0xFFu;
    if (biased == 0xFFu)
        return uint2(0u, sign | ((hiBits & 0x7FFFFFu) != 0 ? 0x7FF80000u : 0x7FF00000u));
    if ((hiBits & 0x7FFFFFFFu) == 0)
        return uint2(0u, sign);
    if (biased < 53u || loSubnormal || sumOverflows)
        return df64_to_ieee_slow(hi, lo);
    // hi = mh * 2^(biased-150) exactly; lo in units of 2^(biased-150-shift) is at most 2^28 and
    // rounds to an integer with ties to even, which is also the tie rule for the whole sum.
    uint mh = (hiBits & 0x7FFFFFu) | 0x800000u;
    bool oppositeSigns = lo != 0.0f && (as_type<uint>(lo) & 0x80000000u) != sign;
    uint shift = (mh == 0x800000u && oppositeSigns) ? 30u : 29u;
    float scale = as_type<float>((306u - biased + shift - 29u) << 23);
    float units = rint((sign ? -lo : lo) * scale);
    ulong m = ((ulong) mh << shift) + (ulong) (long) units;
    ulong bits = ((ulong) (biased + 924u - shift) << 52) + m;
    return uint2((uint) bits, (uint) (bits >> 32) | sign);
}

// ---------------------------------------------------------------------------------------------
// Arithmetic.

inline df64 operator-(df64 a) { return df64(-a.hi, -a.lo); }
inline df64 operator+(df64 a) { return a; }

inline df64 operator+(df64 x, df64 y) {
    float2 s = df64_two_sum(x.hi, y.hi);
    if (!isfinite(s.x))
        return df64(s.x, 0.0f);
    float2 t = df64_two_sum(x.lo, y.lo);
    float2 v = df64_fast_two_sum(s.x, s.y + t.x);
    float2 z = df64_fast_two_sum(v.x, t.y + v.y);
    return df64(z.x, z.y);
}

inline df64 operator+(df64 x, float y) {
    float2 s = df64_two_sum(x.hi, y);
    if (!isfinite(s.x))
        return df64(s.x, 0.0f);
    float2 z = df64_fast_two_sum(s.x, x.lo + s.y);
    return df64(z.x, z.y);
}

inline df64 operator-(df64 x, df64 y) { return x + (-y); }
inline df64 operator-(df64 x, float y) { return x + (-y); }

inline df64 operator*(df64 x, df64 y) {
    float2 c = df64_two_prod(x.hi, y.hi);
    if (!isfinite(c.x))
        return df64(c.x, 0.0f);
    float t = fma(x.hi, y.lo, x.lo * y.lo);
    float2 z = df64_fast_two_sum(c.x, c.y + fma(x.lo, y.hi, t));
    return df64(z.x, z.y);
}

inline df64 operator*(df64 x, float y) {
    float2 c = df64_two_prod(x.hi, y);
    if (!isfinite(c.x))
        return df64(c.x, 0.0f);
    float2 z = df64_fast_two_sum(c.x, fma(x.lo, y, c.y));
    return df64(z.x, z.y);
}

inline df64 operator/(df64 x, float y) {
    float th = x.hi / y;
    if (!isfinite(th))
        return df64(th, 0.0f);
    float2 p = df64_two_prod(th, y);
    float d = ((x.hi - p.x) - p.y) + x.lo;
    float2 z = df64_fast_two_sum(th, d / y);
    return df64(z.x, z.y);
}

// DWDivDW2 exactly as JMP state it, with DWTimesFP1 for (yh, yl) * th: the proof of the 15u^2 bound
// covers that variant, not the FMA product DWTimesFP3 that operator*(df64, float) uses.
inline df64 operator/(df64 x, df64 y) {
    float th = x.hi / y.hi;
    if (!isfinite(th))
        return df64(th, 0.0f);
    float2 c = df64_two_prod(y.hi, th);
    float2 t = df64_fast_two_sum(c.x, y.lo * th);
    float2 r = df64_fast_two_sum(t.x, t.y + c.y);
    float delta = (x.hi - r.x) + (x.lo - r.y);
    float2 z = df64_fast_two_sum(th, delta / y.hi);
    return df64(z.x, z.y);
}

inline df64 operator+(float x, df64 y) { return y + x; }
inline df64 operator-(float x, df64 y) { return (-y) + x; }
inline df64 operator*(float x, df64 y) { return y * x; }
inline df64 operator/(float x, df64 y) { return df64(x) / y; }

// Integer operands (fixed-point forces, atom counts, literals) convert through the df64 constructors.
#define DF64_INTEGRAL(T) typename T, typename = enable_if_t<is_integral<T>::value>
template <DF64_INTEGRAL(T)> inline df64 operator+(df64 x, T y) { return x + df64(y); }
template <DF64_INTEGRAL(T)> inline df64 operator-(df64 x, T y) { return x - df64(y); }
template <DF64_INTEGRAL(T)> inline df64 operator*(df64 x, T y) { return x * df64(y); }
template <DF64_INTEGRAL(T)> inline df64 operator/(df64 x, T y) { return x / df64(y); }
template <DF64_INTEGRAL(T)> inline df64 operator+(T x, df64 y) { return df64(x) + y; }
template <DF64_INTEGRAL(T)> inline df64 operator-(T x, df64 y) { return df64(x) - y; }
template <DF64_INTEGRAL(T)> inline df64 operator*(T x, df64 y) { return df64(x) * y; }
template <DF64_INTEGRAL(T)> inline df64 operator/(T x, df64 y) { return df64(x) / y; }

#define DF64_COMPOUND(AS) \
inline AS df64& df64::operator+=(df64 v) AS { return *this = df64(*this) + v; } \
inline AS df64& df64::operator-=(df64 v) AS { return *this = df64(*this) - v; } \
inline AS df64& df64::operator*=(df64 v) AS { return *this = df64(*this) * v; } \
inline AS df64& df64::operator/=(df64 v) AS { return *this = df64(*this) / v; }
DF64_COMPOUND(thread)
DF64_COMPOUND(device)
DF64_COMPOUND(threadgroup)
DF64_COMPOUND(volatile threadgroup)
#undef DF64_COMPOUND

// ---------------------------------------------------------------------------------------------
// Comparisons. A normalized pair orders lexicographically; any NaN compares unordered.

inline bool operator==(df64 a, df64 b) { return a.hi == b.hi && a.lo == b.lo; }
inline bool operator!=(df64 a, df64 b) { return !(a == b); }
inline bool operator<(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline bool operator>(df64 a, df64 b) { return b < a; }
inline bool operator<=(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo); }
inline bool operator>=(df64 a, df64 b) { return b <= a; }

#define DF64_ARITHMETIC(T) typename T, typename = enable_if_t<is_arithmetic<T>::value>
#define DF64_MIXED_COMPARE(OP) \
template <DF64_ARITHMETIC(T)> inline bool operator OP(df64 a, T b) { return a OP df64(b); } \
template <DF64_ARITHMETIC(T)> inline bool operator OP(T a, df64 b) { return df64(a) OP b; }
DF64_MIXED_COMPARE(==)
DF64_MIXED_COMPARE(!=)
DF64_MIXED_COMPARE(<)
DF64_MIXED_COMPARE(>)
DF64_MIXED_COMPARE(<=)
DF64_MIXED_COMPARE(>=)
#undef DF64_MIXED_COMPARE

// ---------------------------------------------------------------------------------------------
// Math functions with the same names as the float versions, so SQRT(x), fabs(x), min(a, b) and
// friends resolve to these for mixed arguments.

inline bool isnan(df64 x) { return isnan(x.hi); }
inline bool isinf(df64 x) { return isinf(x.hi); }
inline bool isfinite(df64 x) { return isfinite(x.hi); }
inline df64 fabs(df64 x) { return x.hi < 0.0f ? -x : x; }
inline df64 abs(df64 x) { return fabs(x); }
inline df64 min(df64 a, df64 b) { return b < a ? b : a; }
inline df64 max(df64 a, df64 b) { return a < b ? b : a; }
inline df64 fmin(df64 a, df64 b) { return min(a, b); }
inline df64 fmax(df64 a, df64 b) { return max(a, b); }
template <DF64_ARITHMETIC(T)> inline df64 min(df64 a, T b) { return min(a, df64(b)); }
template <DF64_ARITHMETIC(T)> inline df64 min(T a, df64 b) { return min(df64(a), b); }
template <DF64_ARITHMETIC(T)> inline df64 max(df64 a, T b) { return max(a, df64(b)); }
template <DF64_ARITHMETIC(T)> inline df64 max(T a, df64 b) { return max(df64(a), b); }

inline df64 floor(df64 x) {
    float f = floor(x.hi);
    if (f != x.hi)
        return df64(f, 0.0f);
    float2 z = df64_fast_two_sum(f, floor(x.lo));
    return df64(z.x, z.y);
}

inline df64 ceil(df64 x) { return -floor(-x); }

// A whole df64 has integral hi and lo, so both convert exactly.
inline df64::operator long() const thread {
    df64 t = hi < 0.0f ? ceil(*this) : floor(*this);
    return (long) t.hi + (long) t.lo;
}

inline df64 sqrt(df64 x) {
    if (x.hi <= 0.0f)
        return x.hi == 0.0f ? df64(x.hi, 0.0f) : df64(NAN, 0.0f);
    // The bound needs a correctly rounded float sqrt; metal::sqrt under safe math is not (about a
    // quarter of results are off by an ulp on M3), precise::sqrt is.
    float s = precise::sqrt(x.hi);
    if (isinf(s))
        return df64(s, 0.0f);
    float r = fma(-s, s, x.hi) + x.lo;
    float2 z = df64_fast_two_sum(s, r / (2.0f * s));
    return df64(z.x, z.y);
}

inline df64 rsqrt(df64 x) { return df64(1.0f) / sqrt(x); }

inline df64 ldexp(df64 x, int k) { return df64(ldexp(x.hi, k), ldexp(x.lo, k)); }

// expm1 of a reduced argument |r| <= ln2/2: r scaled by 2^-5, a degree-7 Taylor series, then five
// doublings expm1(2t) = expm1(t) * (expm1(t) + 2). Keeping the -1 off preserves relative accuracy near 0.
inline df64 df64_expm1_reduced(df64 r) {
    r = ldexp(r, -5);
    df64 e = r * (1.0f / 5040.0f) + 1.0f / 720.0f;
    e = e * r + 1.0f / 120.0f;
    e = e * r + 1.0f / 24.0f;
    e = e * r + df64(0x1.555556p-3f, -0x1.555556p-28f);
    e = e * r + 0.5f;
    e = e * r + 1.0f;
    e = e * r;
    for (int i = 0; i < 5; i++)
        e = e * (e + 2.0f);
    return e;
}

// x - k*ln2 with ln2 split in three parts; k*c1 is exact for |k| < 2^8.
inline df64 df64_minus_k_ln2(df64 x, float k) {
    const float c1 = 0x1.62e4p-1f;
    const float c2 = 0x1.7f7d1cp-20f;
    const float c3 = 0x1.ef357ap-45f;
    float2 kc2 = df64_two_prod(k, c2);
    return (x - k * c1) - df64(kc2.x, kc2.y) - k * c3;
}

// exp: x = k*ln2 + r, then 2^k * (1 + expm1(r)).
inline df64 exp(df64 x) {
    if (x.hi > 88.72283f)
        return df64(INFINITY, 0.0f);
    if (x.hi < -103.97208f)
        return df64(0.0f, 0.0f);
    if (isnan(x.hi))
        return x;
    float k = rint(x.hi * 0x1.715476p+0f);
    return ldexp(df64_expm1_reduced(df64_minus_k_ln2(x, k)) + 1.0f, (int) k);
}

// log: from a float estimate y, log x = y + log1p(c) with c = x*exp(-y) - 1, and log1p(c) ~ c - c^2/2
// (a plain Newton step drops c^2/2, which for |y| ~ 16 and a float y is ~2^-45 relative). With
// y = k*ln2 + r and x' = x * 2^-k (exact), c = x'*expm1(-r) + (x' - 1): no cancellation against 1, so
// the result stays relatively accurate near x = 1. y includes the first-order term of lo, which near 1
// is as large as log x itself.
inline df64 log(df64 x) {
    if (!(x.hi > 0.0f) || isinf(x.hi))
        return df64(x.hi == 0.0f ? -INFINITY : (isinf(x.hi) && x.hi > 0.0f ? INFINITY : NAN), 0.0f);
    float y = log(x.hi) + x.lo / x.hi;
    float k = rint(y * 0x1.715476p+0f);
    df64 scaled = ldexp(x, -(int) k);
    df64 c = scaled * df64_expm1_reduced(-df64_minus_k_ln2(df64(y), k)) + (scaled - 1.0f);
    return (c - 0.5f * c.hi * c.hi) + y;
}

// A SIMD shuffle moves both words.
inline df64 simd_shuffle(df64 v, ushort lane) { return df64(simd_shuffle(v.hi, lane), simd_shuffle(v.lo, lane)); }
inline df64 simd_shuffle_down(df64 v, ushort delta) { return df64(simd_shuffle_down(v.hi, delta), simd_shuffle_down(v.lo, delta)); }
inline df64 simd_shuffle_xor(df64 v, ushort mask) { return df64(simd_shuffle_xor(v.hi, mask), simd_shuffle_xor(v.lo, mask)); }

// No 64-bit float or integer atomics exist on Apple GPUs before M3's 64-bit min/max, and a df64 spans
// two words. Kernels that atomically add into mixed memory (minimize.cc) must fail loudly.
void atomicAdd(device df64* target, df64 value) = delete;

// ---------------------------------------------------------------------------------------------
// Vector types. Components are df64, so a df64_4 has the size and component offsets of a double4.

// Copies between address spaces go component by component through the df64 conversions.
#define DF64_VECTOR_COMPOUND(V, AS, ASSIGN, SCALE) \
    AS V& operator+=(V o) AS { ASSIGN(+=) return *this; } \
    AS V& operator-=(V o) AS { ASSIGN(-=) return *this; } \
    AS V& operator*=(V o) AS { ASSIGN(*=) return *this; } \
    AS V& operator/=(V o) AS { ASSIGN(/=) return *this; } \
    AS V& operator*=(df64 s) AS { SCALE(*=) return *this; } \
    AS V& operator/=(df64 s) AS { SCALE(/=) return *this; }
#define DF64_VECTOR_ACCESS(V, INIT, ASSIGN, SCALE) \
    V() = default; \
    V() threadgroup = default; \
    V(const thread V&) = default; \
    thread V& operator=(const thread V&) thread = default; \
    V(const device V& o) : INIT {} \
    V(const constant V& o) : INIT {} \
    V(const threadgroup V& o) : INIT {} \
    thread V& operator=(const device V& o) thread { return *this = V(o); } \
    thread V& operator=(const constant V& o) thread { return *this = V(o); } \
    thread V& operator=(const threadgroup V& o) thread { return *this = V(o); } \
    device V& operator=(V o) device { ASSIGN(=) return *this; } \
    threadgroup V& operator=(V o) threadgroup { ASSIGN(=) return *this; } \
    DF64_VECTOR_COMPOUND(V, thread, ASSIGN, SCALE) \
    DF64_VECTOR_COMPOUND(V, device, ASSIGN, SCALE) \
    DF64_VECTOR_COMPOUND(V, threadgroup, ASSIGN, SCALE)

#define DF64_COMMA ,
#define DF64_ASSIGN2(OP) x OP o.x; y OP o.y;
#define DF64_SCALE2(OP) x OP s; y OP s;
#define DF64_ASSIGN3(OP) x OP o.x; y OP o.y; z OP o.z;
#define DF64_SCALE3(OP) x OP s; y OP s; z OP s;
#define DF64_ASSIGN4(OP) x OP o.x; y OP o.y; z OP o.z; w OP o.w;
#define DF64_SCALE4(OP) x OP s; y OP s; z OP s; w OP s;

struct df64_2 {
    df64 x, y;
    DF64_VECTOR_ACCESS(df64_2, x(o.x) DF64_COMMA y(o.y), DF64_ASSIGN2, DF64_SCALE2)
    df64_2(df64 x, df64 y) : x(x), y(y) {}
    explicit df64_2(df64 s) : x(s), y(s) {}
    explicit df64_2(float2 v) : x(v.x), y(v.y) {}
    explicit operator float2() const { return float2(x.hi, y.hi); }
};

struct df64_3 {
    df64 x, y, z;
    DF64_VECTOR_ACCESS(df64_3, x(o.x) DF64_COMMA y(o.y) DF64_COMMA z(o.z), DF64_ASSIGN3, DF64_SCALE3)
    df64_3(df64 x, df64 y, df64 z) : x(x), y(y), z(z) {}
    explicit df64_3(df64 s) : x(s), y(s), z(s) {}
    explicit df64_3(float3 v) : x(v.x), y(v.y), z(v.z) {}
    explicit operator float3() const { return float3(x.hi, y.hi, z.hi); }
};

struct df64_4 {
    df64 x, y, z, w;
    DF64_VECTOR_ACCESS(df64_4, x(o.x) DF64_COMMA y(o.y) DF64_COMMA z(o.z) DF64_COMMA w(o.w), DF64_ASSIGN4, DF64_SCALE4)
    df64_4(df64 x, df64 y, df64 z, df64 w) : x(x), y(y), z(z), w(w) {}
    explicit df64_4(df64 s) : x(s), y(s), z(s), w(s) {}
    explicit df64_4(float4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
    explicit operator float4() const { return float4(x.hi, y.hi, z.hi, w.hi); }
};
#undef DF64_VECTOR_ACCESS
#undef DF64_VECTOR_COMPOUND
#undef DF64_ASSIGN2
#undef DF64_SCALE2
#undef DF64_ASSIGN3
#undef DF64_SCALE3
#undef DF64_ASSIGN4
#undef DF64_SCALE4
#undef DF64_COMMA

inline df64_2 operator+(df64_2 a, df64_2 b) { return df64_2(a.x + b.x, a.y + b.y); }
inline df64_2 operator-(df64_2 a, df64_2 b) { return df64_2(a.x - b.x, a.y - b.y); }
inline df64_2 operator-(df64_2 a) { return df64_2(-a.x, -a.y); }
inline df64_2 operator*(df64_2 a, df64_2 b) { return df64_2(a.x * b.x, a.y * b.y); }
inline df64_2 operator*(df64_2 a, df64 s) { return df64_2(a.x * s, a.y * s); }
inline df64_2 operator*(df64 s, df64_2 a) { return a * s; }
inline df64_2 operator/(df64_2 a, df64_2 b) { return df64_2(a.x / b.x, a.y / b.y); }
inline df64_2 operator/(df64_2 a, df64 s) { return df64_2(a.x / s, a.y / s); }
inline df64_2 operator/(df64 s, df64_2 a) { return df64_2(s / a.x, s / a.y); }

inline df64_3 operator+(df64_3 a, df64_3 b) { return df64_3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline df64_3 operator-(df64_3 a, df64_3 b) { return df64_3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline df64_3 operator-(df64_3 a) { return df64_3(-a.x, -a.y, -a.z); }
inline df64_3 operator*(df64_3 a, df64_3 b) { return df64_3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline df64_3 operator*(df64_3 a, df64 s) { return df64_3(a.x * s, a.y * s, a.z * s); }
inline df64_3 operator*(df64 s, df64_3 a) { return a * s; }
inline df64_3 operator/(df64_3 a, df64_3 b) { return df64_3(a.x / b.x, a.y / b.y, a.z / b.z); }
inline df64_3 operator/(df64_3 a, df64 s) { return df64_3(a.x / s, a.y / s, a.z / s); }
inline df64_3 operator/(df64 s, df64_3 a) { return df64_3(s / a.x, s / a.y, s / a.z); }
inline df64 dot(df64_3 a, df64_3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline df64_3 cross(df64_3 a, df64_3 b) { return df64_3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

inline df64_4 operator+(df64_4 a, df64_4 b) { return df64_4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline df64_4 operator-(df64_4 a, df64_4 b) { return df64_4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline df64_4 operator-(df64_4 a) { return df64_4(-a.x, -a.y, -a.z, -a.w); }
inline df64_4 operator*(df64_4 a, df64_4 b) { return df64_4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline df64_4 operator*(df64_4 a, df64 s) { return df64_4(a.x * s, a.y * s, a.z * s, a.w * s); }
inline df64_4 operator*(df64 s, df64_4 a) { return a * s; }
inline df64_4 operator/(df64_4 a, df64_4 b) { return df64_4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
inline df64_4 operator/(df64_4 a, df64 s) { return df64_4(a.x / s, a.y / s, a.z / s, a.w / s); }
inline df64_4 operator/(df64 s, df64_4 a) { return df64_4(s / a.x, s / a.y, s / a.z, s / a.w); }
inline df64 dot(df64_4 a, df64_4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline df64_3 trimTo3(df64_4 v) { return df64_3(v.x, v.y, v.z); }
inline df64_4 cross(df64_4 a, df64_4 b) { return df64_4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f); }

// Custom expressions on double3 (CustomIntegrator per-DOF steps) call pow on vectors. Like pow on a
// df64, which has no overload of its own, each component narrows to float.
inline df64_3 pow(df64_3 a, df64_3 b) { return df64_3(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z)); }

// The host maps SQRT, RSQRT, EXP, LOG and RECIP to metal::precise:: or metal::fast:: functions.
// A qualified call only sees that namespace, so without these a df64 argument would narrow to float.
#define DF64_QUALIFIED_MATH \
    inline df64 sqrt(df64 x) { return ::sqrt(x); } \
    inline df64 rsqrt(df64 x) { return ::rsqrt(x); } \
    inline df64 exp(df64 x) { return ::exp(x); } \
    inline df64 log(df64 x) { return ::log(x); } \
    inline df64 divide(float x, df64 y) { return x / y; } \
    inline df64_2 divide(float x, df64_2 y) { return df64(x) / y; } \
    inline df64_3 divide(float x, df64_3 y) { return df64(x) / y; } \
    inline df64_4 divide(float x, df64_4 y) { return df64(x) / y; }
namespace metal {
namespace precise { DF64_QUALIFIED_MATH }
namespace fast { DF64_QUALIFIED_MATH }
}
#undef DF64_QUALIFIED_MATH

#undef DF64_INTEGRAL
#undef DF64_ARITHMETIC
