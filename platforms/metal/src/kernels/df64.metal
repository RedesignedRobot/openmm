inline float2 df64_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}

inline float2 df64_fast_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}

inline float2 df64_two_prod(float a, float b) {
    float p = a * b;
    return float2(p, fma(a, b, -p));
}

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
    return as_type<float>(sign | (((uint) (ulpExponent + 149) << 23) + (uint) r));
}

inline void df64_decompose(float x, thread ulong& m, thread int& e) {
    uint b = as_type<uint>(x) & 0x7FFFFFFFu;
    uint biased = b >> 23;
    m = biased == 0 ? (b & 0x7FFFFFu) : ((b & 0x7FFFFFu) | 0x800000u);
    e = biased == 0 ? -149 : (int) biased - 150;
}

inline float df64_below_tie(float hi, float lo) {
    uint h = as_type<uint>(hi);
    uint biased = (h >> 23) & 0xFFu;
    uint halfUlp = biased > 24u ? (biased - 24u) << 23 : (biased >= 2u ? 1u << (biased - 2u) : 0u);
    bool tie = (h & 1u) != 0 && halfUlp != 0 && (as_type<uint>(lo) & 0x7FFFFFFFu) == halfUlp;
    return tie ? as_type<float>(as_type<uint>(lo) - 1u) : lo;
}

struct df64;
struct df64s;
inline df64 df64_from_ieee(uint2 bits);
inline uint2 df64_to_ieee(df64 v);

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
    // Without these two, every thread temporary binds to the volatile rvalue overloads below, which spill it to the
    // stack and read it back.
    df64(thread df64&&) = default;
    thread df64& operator=(thread df64&&) thread = default;
    df64(const volatile thread df64&& o) : hi(o.hi), lo(o.lo) {}
    thread df64& operator=(const volatile thread df64&& o) thread { hi = o.hi; lo = o.lo; return *this; }
    threadgroup df64& operator=(df64 v) threadgroup { hi = v.hi; lo = v.lo; return *this; }
    volatile threadgroup df64& operator=(df64 v) volatile threadgroup { hi = v.hi; lo = v.lo; return *this; }

    df64(const device df64& o) : df64(df64_load_ieee(reinterpret_cast<const device uint*>(&o))) {}
    df64(const device df64s& o);
    thread df64& operator=(const device df64s& o) thread { return *this = df64(o); }
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

    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const thread { return hi; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const device { return df64(*this).hi; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const constant { return df64(*this).hi; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> operator T() const threadgroup { return hi; }
    explicit operator long() const thread;
    explicit operator int() const thread { return (int) (long) *this; }

#define DF64_COMPOUND(AS) \
    AS df64& operator+=(df64 v) AS; \
    AS df64& operator-=(df64 v) AS; \
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> AS df64& operator+=(T v) AS; \
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> AS df64& operator-=(T v) AS; \
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
    long rest = mh == 0 ? (long) m : (long) m - (long) (mh << (eh - e));
    if (rest == 0)
        return df64(hi, 0.0f);
    return df64(hi, df64_below_tie(hi, df64_round_to_float((rest < 0) != negative, (ulong) abs(rest), e)));
}

inline df64 df64_from_ieee(uint2 b) {
    uint biased = (b.y >> 20) & 0x7FFu;
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
    uint mh = (hiBits & 0x7FFFFFu) | 0x800000u;
    bool oppositeSigns = lo != 0.0f && (as_type<uint>(lo) & 0x80000000u) != sign;
    uint shift = (mh == 0x800000u && oppositeSigns) ? 30u : 29u;
    float scale = as_type<float>((306u - biased + shift - 29u) << 23);
    float units = rint((sign ? -lo : lo) * scale);
    ulong m = ((ulong) mh << shift) + (ulong) (long) units;
    ulong bits = ((ulong) (biased + 924u - shift) << 52) + m;
    return uint2((uint) bits, (uint) (bits >> 32) | sign);
}

inline df64 operator-(df64 a) { return df64(-a.hi, -a.lo); }
inline df64 operator+(df64 a) { return a; }

// An overflowed sum or product returns inf with a zero low part; unguarded, the two-sum makes it NaN.  A program whose
// mixed values stay finite defines DF64_FINITE to drop these guards: finite results are unchanged, and the guards cost
// about a fifth of the fused Langevin kernel, whose SHAKE sweeps are a serial chain of these operators.
#ifdef DF64_FINITE
#define DF64_GUARD_OVERFLOW(v)
#else
#define DF64_GUARD_OVERFLOW(v) if (!isfinite(v)) return df64(v, 0.0f);
#endif

// One rounding joins the low parts instead of a second two-sum, so the error is about 2^-47 of |x|+|y|.
inline df64 operator+(df64 x, df64 y) {
    float2 s = df64_two_sum(x.hi, y.hi);
    DF64_GUARD_OVERFLOW(s.x)
    float2 z = df64_fast_two_sum(s.x, s.y + (x.lo + y.lo));
    return df64(z.x, z.y);
}

inline df64 operator+(df64 x, float y) {
    float2 s = df64_two_sum(x.hi, y);
    DF64_GUARD_OVERFLOW(s.x)
    float2 z = df64_fast_two_sum(s.x, x.lo + s.y);
    return df64(z.x, z.y);
}

inline df64 operator-(df64 x, df64 y) { return x + (-y); }
inline df64 operator-(df64 x, float y) { return x + (-y); }

inline df64 operator*(df64 x, df64 y) {
    float2 c = df64_two_prod(x.hi, y.hi);
    DF64_GUARD_OVERFLOW(c.x)
    float2 z = df64_fast_two_sum(c.x, c.y + fma(x.lo, y.hi, x.hi * y.lo));
    return df64(z.x, z.y);
}

inline df64 operator*(df64 x, float y) {
    float2 c = df64_two_prod(x.hi, y);
    DF64_GUARD_OVERFLOW(c.x)
    float2 z = df64_fast_two_sum(c.x, fma(x.lo, y, c.y));
    return df64(z.x, z.y);
}

// Both divisions multiply by 1/y, which is accurate only for 2^-126 <= |y| <= 2^126.
inline df64 operator/(df64 x, float y) {
    float r = 1.0f / y;
    float th = x.hi * r;
    if (!isfinite(th))
        return df64(th, 0.0f);
    float2 p = df64_two_prod(th, y);
    float d = ((x.hi - p.x) - p.y) + x.lo;
    float2 z = df64_fast_two_sum(th, d * r);
    return df64(z.x, z.y);
}

inline df64 operator/(df64 x, df64 y) {
    float r = 1.0f / y.hi;
    float th = x.hi * r;
    if (!isfinite(th))
        return df64(th, 0.0f);
    float2 p = df64_two_prod(th, y.hi);
    float d = (((x.hi - p.x) - p.y) + x.lo) - th * y.lo;
    float2 z = df64_fast_two_sum(th, d * r);
    return df64(z.x, z.y);
}

inline df64 operator+(float x, df64 y) { return y + x; }
inline df64 operator-(float x, df64 y) { return (-y) + x; }
inline df64 operator*(float x, df64 y) { return y * x; }
inline df64 operator/(float x, df64 y) { return df64(x) / y; }

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
template <typename T, typename> inline AS df64& df64::operator+=(T v) AS { return *this = df64(*this) + v; } \
template <typename T, typename> inline AS df64& df64::operator-=(T v) AS { return *this = df64(*this) - v; } \
inline AS df64& df64::operator*=(df64 v) AS { return *this = df64(*this) * v; } \
inline AS df64& df64::operator/=(df64 v) AS { return *this = df64(*this) / v; }
DF64_COMPOUND(thread)
DF64_COMPOUND(device)
DF64_COMPOUND(threadgroup)
DF64_COMPOUND(volatile threadgroup)
#undef DF64_COMPOUND

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

inline df64::operator long() const thread {
    df64 t = hi < 0.0f ? ceil(*this) : floor(*this);
    return (long) t.hi + (long) t.lo;
}

inline df64 sqrt(df64 x) {
    if (x.hi <= 0.0f)
        return x.hi == 0.0f ? df64(x.hi, 0.0f) : df64(NAN, 0.0f);
    if (isinf(x.hi))
        return df64(x.hi, 0.0f);
    float rs = rsqrt(x.hi);
    float s = x.hi * rs;
    float r = fma(-s, s, x.hi) + x.lo;
    float2 z = df64_fast_two_sum(s, r * (0.5f * rs));
    return df64(z.x, z.y);
}

inline df64 rsqrt(df64 x) { return df64(1.0f) / sqrt(x); }

inline df64 ldexp(df64 x, int k) { return df64(ldexp(x.hi, k), ldexp(x.lo, k)); }

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

inline df64 df64_minus_k_ln2(df64 x, float k) {
    const float c1 = 0x1.62e4p-1f;
    const float c2 = 0x1.7f7d1cp-20f;
    const float c3 = 0x1.ef357ap-45f;
    float2 kc2 = df64_two_prod(k, c2);
    return (x - k * c1) - df64(kc2.x, kc2.y) - k * c3;
}

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

inline df64 log(df64 x) {
    if (!(x.hi > 0.0f) || isinf(x.hi))
        return df64(x.hi == 0.0f ? -INFINITY : (isinf(x.hi) && x.hi > 0.0f ? INFINITY : NAN), 0.0f);
    float y = log(x.hi) + x.lo / x.hi;
    float k = rint(y * 0x1.715476p+0f);
    df64 scaled = ldexp(x, -(int) k);
    df64 c = scaled * df64_expm1_reduced(-df64_minus_k_ln2(df64(y), k)) + (scaled - 1.0f);
    return (c - 0.5f * c.hi * c.hi) + y;
}

// Reduces x to r in [-pi/4, pi/4] with x = r + k*pi/2, returns k mod 4, and sets s and c to sin(r)
// and cos(r) from Taylor series, whose terms past r^17 are below df64 precision. The reduction keeps
// df64 precision for |x| below about 2e4, and k*c1 stays exact up to |x| of about 1e5.
inline int df64_sincos_reduced(df64 x, thread df64& s, thread df64& c) {
    const float c1 = 0x1.92p+0f;
    const float c2 = 0x1.fb5444p-12f;
    const float c3 = 0x1.68c234p-39f;
    float k = rint(x.hi * 0x1.45f306p-1f);
    float2 kc2 = df64_two_prod(k, c2);
    df64 r = (x - k * c1) - df64(kc2.x, kc2.y) - k * c3;
    df64 r2 = r * r;
    df64 sinTerm = r, cosTerm = df64(1.0f);
    s = r;
    c = df64(1.0f);
    for (int n = 2; n <= 16; n += 2) {
        cosTerm = -cosTerm * r2 / (float) ((n - 1) * n);
        sinTerm = -sinTerm * r2 / (float) (n * (n + 1));
        c += cosTerm;
        s += sinTerm;
    }
    return ((int) k) & 3;
}

inline df64 sin(df64 x) {
    if (!isfinite(x.hi))
        return df64(NAN, 0.0f);
    df64 s, c;
    int quadrant = df64_sincos_reduced(x, s, c);
    return quadrant == 0 ? s : (quadrant == 1 ? c : (quadrant == 2 ? -s : -c));
}

inline df64 cos(df64 x) {
    if (!isfinite(x.hi))
        return df64(NAN, 0.0f);
    df64 s, c;
    int quadrant = df64_sincos_reduced(x, s, c);
    return quadrant == 0 ? c : (quadrant == 1 ? -s : (quadrant == 2 ? -c : s));
}

df64 tan(df64) = delete;
df64 asin(df64) = delete;
df64 acos(df64) = delete;
df64 atan(df64) = delete;
df64 atan2(df64, df64) = delete;
df64 sinh(df64) = delete;
df64 cosh(df64) = delete;
df64 tanh(df64) = delete;
df64 erf(df64) = delete;
df64 erfc(df64) = delete;
df64 pow(df64, df64) = delete;

inline df64 simd_shuffle(df64 v, ushort lane) { return df64(simd_shuffle(v.hi, lane), simd_shuffle(v.lo, lane)); }
inline df64 simd_shuffle_down(df64 v, ushort delta) { return df64(simd_shuffle_down(v.hi, delta), simd_shuffle_down(v.lo, delta)); }
inline df64 simd_shuffle_xor(df64 v, ushort mask) { return df64(simd_shuffle_xor(v.hi, mask), simd_shuffle_xor(v.lo, mask)); }

void atomicAdd(device df64* target, df64 value) = delete;

// A df64 stored in device memory as the float pair it is in registers, with no IEEE decode or encode.  Metal mixed
// stores the integration state this way (mixed4s: velm and posDelta), and MetalArray converts it on host transfers.
// It reads as a df64 through df64's constructor.  Its float conversion is explicit: an implicit one would tie with that
// constructor in mixed expressions like 1.0f/v.
struct alignas(8) df64s {
    float shi, slo;

    device df64s& operator=(df64 v) device { shi = v.hi; slo = v.lo; return *this; }
    explicit operator float() const device { return shi; }
    device df64s& operator+=(df64 v) device { return *this = df64(*this) + v; }
    device df64s& operator-=(df64 v) device { return *this = df64(*this) - v; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> device df64s& operator+=(T v) device { return *this = df64(*this) + v; }
    template <typename T, typename = enable_if_t<is_same<T, float>::value>> device df64s& operator-=(T v) device { return *this = df64(*this) - v; }
    device df64s& operator*=(df64 v) device { return *this = df64(*this) * v; }
    device df64s& operator/=(df64 v) device { return *this = df64(*this) / v; }
};

inline df64::df64(const device df64s& o) : hi(o.shi), lo(o.slo) {}

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

struct df64s_4;

struct df64_4 {
    df64 x, y, z, w;
    DF64_VECTOR_ACCESS(df64_4, x(o.x) DF64_COMMA y(o.y) DF64_COMMA z(o.z) DF64_COMMA w(o.w), DF64_ASSIGN4, DF64_SCALE4)
    df64_4(const device df64s_4& o);
    thread df64_4& operator=(const device df64s_4& o) thread { return *this = df64_4(o); }
    df64_4(df64 x, df64 y, df64 z, df64 w) : x(x), y(y), z(z), w(w) {}
    explicit df64_4(df64 s) : x(s), y(s), z(s), w(s) {}
    explicit df64_4(float4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
    explicit operator float4() const { return float4(x.hi, y.hi, z.hi, w.hi); }
};
// The pair-stored form of df64_4 (see df64s).
struct df64s_4 {
    df64s x, y, z, w;

    device df64s_4& operator=(df64_4 o) device { DF64_ASSIGN4(=) return *this; }
    device df64s_4& operator+=(df64_4 o) device { DF64_ASSIGN4(+=) return *this; }
    device df64s_4& operator-=(df64_4 o) device { DF64_ASSIGN4(-=) return *this; }
    device df64s_4& operator*=(df64_4 o) device { DF64_ASSIGN4(*=) return *this; }
    device df64s_4& operator/=(df64_4 o) device { DF64_ASSIGN4(/=) return *this; }
    device df64s_4& operator*=(df64 s) device { DF64_SCALE4(*=) return *this; }
    device df64s_4& operator/=(df64 s) device { DF64_SCALE4(/=) return *this; }
};

inline df64_4::df64_4(const device df64s_4& o) : x(o.x), y(o.y), z(o.z), w(o.w) {}

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

df64_3 pow(df64_3, df64_3) = delete;

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

#define SUPPORTS_DOUBLE_PRECISION 1
#define double df64
#define double2 df64_2
#define double3 df64_3
#define double4 df64_4
#define make_double2 df64_2
#define make_double3 df64_3
#define make_double4 df64_4
