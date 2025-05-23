// Updated from MLX commit has f70764a

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)    \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }                                                                       \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                 \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);         \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(            \
      addr_space _MLX_BFloat16& lhs, itype rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }                                                                       \
  constexpr METAL_FUNC addr_space itype& __operator__(                    \
      addr_space itype& lhs, _MLX_BFloat16 rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype) \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                             \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype); \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype); \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype); \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(     \
      addr_space _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) {          \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);  \
    return lhs;                                                    \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__) \
  bfloat_inplace_op_helper(__op__, __operator__, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif

// ============ "mlx/backend/metal/kernels/scaled_dot_product_attention_params.h"

struct MLXFastAttentionParams {
  const int M;
  const int N;
  const int K;

  const int ldq; // ldq == ldo
  const int ldk;
  const int ldv;
  const int lds;
  const int ldo;

  const int tiles_n;
  const int tiles_m;

  const int batch_stride_q;
  const int batch_stride_k;
  const int batch_stride_v;
  const int batch_stride_o;

  const int swizzle_log;
  const int gemm_n_iterations_aligned;
  const int gemm_k_iterations_aligned;
  const int gemm_sv_m_block_iterations;

  const int batch_ndim;
  const float alpha;
  const float softcapping;
};

struct MLXScaledDotProductAttentionParams {
  // Associated dimensions & transposition information
  const uint QUERY_SEQUENCE_LENGTH = 1;
  const uint N_Q_HEADS = 32;
  const uint N_KV_HEADS = 32;
  const uint KV_TILES = 1;
  const float INV_ALPHA = 0.08838834764831843f;
};

// ============ "mlx/backend/metal/kernels/scaled_dot_product_attention_params.sdpa_vector"

template <typename T, int D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    const constant float& softcapping,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  const int stride = BN * D;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + simd_gid * D + simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + simd_gid * D + simd_lid * elem_per_thread;
  out += head_idx * D + simd_gid * elem_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    // Read the key
    for (int i = 0; i < elem_per_thread; i++) {
      k[i] = keys[i];
    }

    // Compute the i-th score
    U score = 0;
    for (int i = 0; i < elem_per_thread; i++) {
      score += q[i] * k[i];
    }
    score = simd_sum(score);
    if (softcapping != 1.) {
      score = precise::tanh(score);
      score = score * softcapping;
    }

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * values[i];
    }

    // Move the pointers to the next kv
    keys += stride;
    values += stride;
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    const constant float& softcapping,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;
  constexpr int blocks = 32;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + (block_idx * BN + simd_gid) * D +
      simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + (block_idx * BN + simd_gid) * D +
      simd_lid * elem_per_thread;
  out += head_idx * blocks * D + block_idx * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks + block_idx;
  maxs += head_idx * blocks + block_idx;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // For each key
  for (int i = block_idx * BN + simd_gid; i < N; i += blocks * BN) {
    // Read the key
    for (int i = 0; i < elem_per_thread; i++) {
      k[i] = keys[i];
    }

    // Compute the i-th score
    U score = 0;
    for (int i = 0; i < elem_per_thread; i++) {
      score += q[i] * k[i];
    }
    score = simd_sum(score);
    if (softcapping != 1.) {
      score = precise::tanh(score);
      score = score * softcapping;
    }

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * values[i];
    }

    // Move the pointers to the next kv
    keys += blocks * stride;
    values += blocks * stride;
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] =
        o[i] * fast::exp(max_scores[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // And write the output
    if (simd_gid == 0) {
      U output = outputs[simd_lid * BN];
      for (int j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = 32;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.y;
  partials += head_idx * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks;
  maxs += head_idx * blocks;
  out += head_idx * D + simd_gid * elem_per_thread;

  // First everybody reads the max and sum_exp
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);

  // Now read the block into registers and then use shared memory to transpose
  // it
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ============ "mlx/backend/metal/kernels/steel/defines.h"

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// ============ "mlx/backend/metal/kernels/steel/gemm/transforms.h"

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT) {
    return static_cast<OutT>(x);
  }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(x * alpha + (beta * c));
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

// ============ "mlx/backend/metal/kernels/utils.h"

typedef half float16_t;

METAL_FUNC ulong2 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
  }
  return ulong2(loc_a, loc_b);
}

METAL_FUNC ulong3 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  ulong loc_c{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
    loc_c += pos_in_dim * c_strides[i];
  }
  return ulong3(loc_a, loc_b, loc_c);
}

// ============ "mlx/backend/metal/kernels/scaled_dot_product_attention_params.metal"

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderFA {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  /* Constructor */
  METAL_FUNC BlockLoaderFA(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out unneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }
  METAL_FUNC void next(short n) {
    src += n * tile_stride;
  }
};

template <bool M_aligned, bool N_aligned, bool K_aligned>
struct LoopAlignment {};

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMAFA {
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = 8 * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = 8 * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

  // Strides of A, B along reduction axis
  STEEL_CONST short simd_stride_a = {
      transpose_a ? TM_stride : TM_stride * lda_tgp};
  STEEL_CONST short simd_stride_b = {
      transpose_b ? TN_stride * ldb_tgp : TN_stride};

  // Jump between elements
  STEEL_CONST short jump_a = {transpose_a ? lda_tgp : 1};
  STEEL_CONST short jump_b = {transpose_b ? ldb_tgp : 1};

  STEEL_CONST short tile_stride_a = {transpose_a ? 8 * lda_tgp : 8};
  STEEL_CONST short tile_stride_b = {transpose_b ? 8 : 8 * ldb_tgp};

  // Simdgroup matrices
  simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
  simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
  simdgroup_matrix<AccumType, 8, 8> results[TM * TN] = {
      simdgroup_matrix<AccumType, 8, 8>(0)};

  // Offsets within threadgroup
  const short tm;
  const short tn;

  short sm;
  short sn;

  ushort sid;
  ushort slid;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMAFA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : tm(8 * (simd_group_id / WN)), tn(8 * (simd_group_id % WN)) {
    // Determine thread position in simdgroup matrix
    short qid = simd_lane_id / 4;
    slid = simd_lane_id;
    sid = simd_group_id;

    sm = (qid & 4) + (simd_lane_id / 2) % 4;
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

    // Determine thread and simdgroup offset
    As_offset =
        transpose_a ? ((sn)*lda_tgp + (tm + sm)) : ((sn) + (tm + sm) * lda_tgp);
    Bs_offset =
        transpose_b ? ((tn + sn) * ldb_tgp + (sm)) : ((sm)*ldb_tgp + (tn + sn));
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of 8
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += 8) {
      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup A as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        Asimd[i].thread_elements()[0] =
            static_cast<AccumType>(As[i * simd_stride_a + 0]);
        Asimd[i].thread_elements()[1] =
            static_cast<AccumType>(As[i * simd_stride_a + jump_a]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup B as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        Bsimd[j].thread_elements()[0] =
            static_cast<AccumType>(Bs[j * simd_stride_b + 0]);
        Bsimd[j].thread_elements()[1] =
            static_cast<AccumType>(Bs[j * simd_stride_b + jump_b]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Multiply and accumulate into result simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          short j_serp = (i % 2) ? (TN - 1 - j) : j;

          simdgroup_multiply_accumulate(
              results[i * TN + j_serp],
              Asimd[i],
              Bsimd[j_serp],
              results[i * TN + j_serp]);
        }
      }

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  METAL_FUNC void rescale_output(const threadgroup float* Corrections) {
    // Loop over all simdgroup tiles

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      short row = sm + tm + i * TM_stride;
      float scale_value = Corrections[row];

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = results[i * TN + j].thread_elements();
        // int offset = (i * TM_stride) * ldc + (j * TN_stride);
        accum[0] *= scale_value;
        accum[1] *= scale_value;
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* C, const int ldc) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset = (i * TM_stride) * ldc + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {Epilogue::apply(accum[0]), Epilogue::apply(accum[1])};

        // Write out C
        C[offset] = outs[0];
        C[offset + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void store_result_to_tgp_memory(
      threadgroup U* C,
      const int ldc,
      short2 dst_tile_dims) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn);
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset = (i * TM_stride) * ldc + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            C[offset] = Epilogue::apply(accum[0]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            C[offset + 1] = Epilogue::apply(accum[1]);
          }
        }
      }
    }
  }

  METAL_FUNC void
  store_result_safe(device U* C, const int ldc, short2 dst_tile_dims) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn);
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset = (i * TM_stride) * ldc + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            C[offset] = Epilogue::apply(accum[0]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            C[offset + 1] = Epilogue::apply(accum[1]);
          }
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {
            epilogue_op.apply(accum[0], C[offset_c]),
            epilogue_op.apply(accum[1], C[offset_c + fdc])};

        // Write out D
        D[offset_d] = outs[0];
        D[offset_d + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            D[offset_d] = epilogue_op.apply(accum[0], C[offset_c]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            D[offset_d + 1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
          }
        }
      }
    }
  }

  METAL_FUNC void clear_results() {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        results[i * TN + j] = simdgroup_matrix<AccumType, 8, 8>(0);
      }
    }
  }
};

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_q,
    bool transpose_k,
    bool transpose_v,
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<U, AccumType>>
struct FastAttentionKernel {
  STEEL_CONST short tgp_padding = 16 / sizeof(T);
  STEEL_CONST short float_padding = 16 / sizeof(float);
  STEEL_CONST short tgp_mem_size_q =
      transpose_q ? BK * (BM + tgp_padding) : BM * (BK + tgp_padding);
  STEEL_CONST short tgp_mem_size_k =
      transpose_k ? BK * (BN + tgp_padding) : BN * (BK + tgp_padding);
  STEEL_CONST short tgp_mem_size_v =
      transpose_v ? BK * (BN + tgp_padding) : BN * (BK + tgp_padding);
  STEEL_CONST short tgp_mem_size_s = BM * (BN + tgp_padding);

  // maxes, rowsums, rescale
  STEEL_CONST short tgp_mem_size_corrections =
      4 * (BM * sizeof(float) + float_padding);

  STEEL_CONST bool share_kv_smem = transpose_k != transpose_v;

  STEEL_CONST short tgp_mem_size = share_kv_smem
      ? tgp_mem_size_q + tgp_mem_size_k + tgp_mem_size_s +
          tgp_mem_size_corrections
      : tgp_mem_size_q + tgp_mem_size_k + tgp_mem_size_s +
          tgp_mem_size_corrections + tgp_mem_size_v;

  STEEL_CONST short tgp_size = WM * WN * 32;

  static_assert(transpose_q == false, "Expected Q not transposed.");
  static_assert(transpose_k == true, "Expected K transposed.");
  static_assert(transpose_v == false, "Expected V not transposed.");
  static_assert(tgp_mem_size <= 32768, "Excessive tgp memory requested.");

  using loader_q_t = BlockLoaderFA<
      T,
      transpose_q ? BK : BM,
      transpose_q ? BM : BK,
      transpose_q ? BM + tgp_padding : BK + tgp_padding,
      !transpose_q,
      tgp_size>;

  using loader_k_t = BlockLoaderFA<
      T,
      transpose_k ? BN : BK,
      transpose_k ? BK : BN,
      transpose_k ? BK + tgp_padding : BN + tgp_padding,
      transpose_k,
      tgp_size>;

  using loader_v_t = BlockLoaderFA<
      T,
      transpose_v ? BK : BN,
      transpose_v ? BN : BK,
      transpose_v ? BN + tgp_padding : BK + tgp_padding,
      transpose_v,
      tgp_size>;

  using mma_qk_t = BlockMMAFA<
      T,
      U,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_q,
      transpose_k,
      transpose_q ? BM + tgp_padding : BK + tgp_padding,
      transpose_k ? BK + tgp_padding : BN + tgp_padding,
      AccumType,
      Epilogue>;

  using mma_sv_t = BlockMMAFA<
      T,
      U,
      BM,
      BK,
      BN,
      WM,
      WN,
      false,
      transpose_v,
      BN + tgp_padding,
      BK + tgp_padding,
      AccumType,
      Epilogue>;

  /* Main kernel function */
  template <bool M_aligned, bool N_aligned, bool K_aligned_>
  static METAL_FUNC void gemm_loop(
      threadgroup T* As [[threadgroup(0)]],
      threadgroup T* Bs [[threadgroup(1)]],
      const int gemm_k_iterations,
      thread loader_k_t& loader_b,
      thread mma_qk_t& mma_op,
      thread const short& tgp_bm,
      thread const short& tgp_bn,
      LoopAlignment<M_aligned, N_aligned, K_aligned_> l = {}) {
    // Appease the compiler
    (void)l;
    (void)tgp_bm;

    short2 tile_dims_B = transpose_k ? short2(BK, tgp_bn) : short2(tgp_bn, BK);

    // not valid for gemm_k_iterations > 1 (so, BK == d_k)
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (N_aligned) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dims_B);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);
    }
  }

  static METAL_FUNC void initialize_corrections(
      threadgroup float* C,
      uint simd_lane_id,
      uint simd_group_id) {
    if (simd_group_id == 0) {
      threadgroup float* maxes = C;
      threadgroup float* sums = C + (BM + float_padding);
      threadgroup float* o_rescale = sums + (BM + float_padding);
      threadgroup float* output_rescale = o_rescale + (BM + float_padding);

      if (simd_lane_id < BM) {
        maxes[simd_lane_id] = -INFINITY; // m_i
        sums[simd_lane_id] = 0.f; // l_i
        o_rescale[simd_lane_id] = 1.f; // li * exp(mi - mi_new)
        output_rescale[simd_lane_id] = 1.f; // 1.0 / l_i
      }
    }
  }

  static METAL_FUNC void rescale_ss(
      threadgroup T* Ss,
      threadgroup float* Corrections,
      uint simd_group_id,
      uint simd_lane_id,
      short2 local_blocks,
      float alpha,
      float softcapping) {
    if (simd_group_id == 0) {
      short row_offset = BM + float_padding;
      threadgroup float* maxes = Corrections;
      threadgroup float* sums = Corrections + row_offset;
      threadgroup float* o_rescale = sums + row_offset;
      threadgroup float* output_scales = o_rescale + row_offset;

      if (simd_lane_id < uint(local_blocks.y)) {
        float m_i_old = maxes[simd_lane_id];
        float l_i_old = sums[simd_lane_id];

        float m_i_new = m_i_old;
        float l_i_new = l_i_old;

        short offset = simd_lane_id * (BN + tgp_padding);

        float m_ij = -INFINITY;

        for (short j = 0; j < local_blocks.x; j++) {
          float val = alpha * float(Ss[offset + j]);
          if (softcapping != 1.) {
            val = precise::tanh(val);
            val = val * softcapping;
          }
          m_ij = max(m_ij, val);
        }

        m_i_new = max(m_ij, m_i_new);

        float rowsum = 0.f; // lij

        for (short j = 0; j < local_blocks.x; j++) {
          float val = alpha * float(Ss[offset + j]);
          if (softcapping != 1.) {
            val = precise::tanh(val);
            val = val * softcapping;
          }
          float P_i_j = exp(val - m_ij);
          rowsum += P_i_j;
          P_i_j = P_i_j * exp(m_ij - m_i_new);
          Ss[offset + j] = T(P_i_j);
        }

        l_i_new =
            exp(m_i_old - m_i_new) * l_i_old + exp(m_ij - m_i_new) * rowsum;
        maxes[simd_lane_id] = m_i_new;
        sums[simd_lane_id] = l_i_new;
        float rescale = l_i_old * exp(m_i_old - m_i_new);
        o_rescale[simd_lane_id] = rescale;
        output_scales[simd_lane_id] = 1.0 / l_i_new;
      }
    }
  }

  /* Main kernel function */
  static METAL_FUNC void run(
      const device T* Q [[buffer(0)]],
      const device T* K [[buffer(1)]],
      const device T* V [[buffer(2)]],
      device U* O [[buffer(3)]],
      const constant MLXFastAttentionParams* params [[buffer(4)]],
      threadgroup T* Qs [[threadgroup(0)]],
      threadgroup T* Ks [[threadgroup(1)]],
      threadgroup T* Ss [[threadgroup(2)]],
      threadgroup T* Vs [[threadgroup(3)]],
      threadgroup float* Corrections [[threadgroup(4)]],
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    // Pacifying compiler
    (void)lid;

    const int tid_y = ((tid.y) << params->swizzle_log) +
        ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int tid_x = (tid.x) >> params->swizzle_log;

    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Find block in Q, O; and head in K, V.
    const int c_row = tid_y * BM;

    Q += transpose_q ? c_row : c_row * params->ldq;
    thread loader_q_t loader_q(Q, params->ldq, Qs, simd_group_id, simd_lane_id);

    short tgp_bm = min(BM, params->M - c_row);
    short2 tile_dims_Q = transpose_q ? short2(tgp_bm, BK) : short2(BK, tgp_bm);

    loader_q.load_safe(tile_dims_Q);

    initialize_corrections(Corrections, simd_lane_id, simd_group_id);

    O += c_row * params->ldo;

    // Prepare threadgroup mma operation
    thread mma_qk_t mma_qk_op(simd_group_id, simd_lane_id);
    thread mma_sv_t mma_softmax_sv_op(simd_group_id, simd_lane_id);
    thread loader_k_t loader_k(K, params->ldk, Ks, simd_group_id, simd_lane_id);
    thread loader_v_t loader_v(V, params->ldv, Vs, simd_group_id, simd_lane_id);

    for (short n_block = 0; n_block < params->gemm_n_iterations_aligned;
         n_block++) {
      short c_col = BN;

      // Prepare threadgroup loading operations
      short gemm_k_iterations = params->gemm_k_iterations_aligned;
      short tgp_bn_qk = min(BN, params->N - c_col * n_block);
      threadgroup_barrier(mem_flags::mem_none);

      ///////////////////////////////////////////////////////////////////////////////
      { // Loop over K - unaligned case

        if (tgp_bm == BM && tgp_bn_qk == BN) {
          gemm_loop<true, true, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);
        } else if (tgp_bn_qk == BN) {
          gemm_loop<false, true, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);

        } else if (tgp_bm == BM) {
          gemm_loop<true, false, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);

        } else {
          gemm_loop<false, false, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);
        }
      }

      mma_qk_op.store_result_to_tgp_memory(
          Ss, BN + tgp_padding, short2(BN, BM));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      rescale_ss(
          Ss,
          Corrections,
          simd_group_id,
          simd_lane_id,
          short2(tgp_bn_qk, tgp_bm),
          params->alpha,
          params->softcapping);

      loader_v.load_safe(short2(BK, tgp_bn_qk));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      threadgroup float* o_scales = Corrections + 2 * (BM + float_padding);
      mma_softmax_sv_op.rescale_output(o_scales);

      mma_softmax_sv_op.mma(Ss, Vs);

      threadgroup float* final_output_scales =
          Corrections + 3 * (BM + float_padding);

      mma_softmax_sv_op.rescale_output(final_output_scales);

      loader_v.next();
      loader_k.next(BN);

      mma_qk_op.clear_results();
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_softmax_sv_op.store_result_safe(O, params->ldo, short2(BK, tgp_bm));
  }
};

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_q,
    bool transpose_k,
    bool transpose_v,
    bool MN_aligned,
    bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant MLXFastAttentionParams* params [[buffer(4)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant size_t* batch_strides [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using attention_kernel = FastAttentionKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_q,
      transpose_k,
      transpose_v,
      MN_aligned,
      K_aligned>;

  // Adjust for batch
  if (params->batch_ndim > 1) {
    const constant size_t* Q_bstrides = batch_strides;
    const constant size_t* KV_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, Q_bstrides, KV_bstrides, params->batch_ndim);

    Q += batch_offsets.x;
    K += batch_offsets.y;
    V += batch_offsets.y;

  } else {
    Q += params->batch_stride_q * tid.z;
    K += params->batch_stride_k * tid.z;
    V += params->batch_stride_v * tid.z;
  }

  // same shape as input
  O += params->batch_stride_o * tid.z;
  threadgroup T Qs[attention_kernel::tgp_mem_size_q];
  threadgroup T Ss[attention_kernel::tgp_mem_size_s];
  threadgroup float Corrections[attention_kernel::tgp_mem_size_corrections];

  if (attention_kernel::share_kv_smem) {
    threadgroup T Ks[attention_kernel::tgp_mem_size_k];
    threadgroup T* Vs = Ks; //[attention_kernel::tgp_mem_size_v];
    attention_kernel::run(
        Q,
        K,
        V,
        O,
        params,
        Qs,
        Ks,
        Ss,
        Vs,
        Corrections,
        simd_lane_id,
        simd_group_id,
        tid,
        lid);
  } else {
    threadgroup T Ks[attention_kernel::tgp_mem_size_k];
    threadgroup T Vs[attention_kernel::tgp_mem_size_v];
    attention_kernel::run(
        Q,
        K,
        V,
        O,
        params,
        Qs,
        Ks,
        Ss,
        Vs,
        Corrections,
        simd_lane_id,
        simd_group_id,
        tid,
        lid);
  }
}

// clang-format off

// SDPA full instantiations
#define instantiate_fast_inference_self_attention_kernel(                   \
    itype, otype, bm, bn, bk, wm, wn)                                       \
  template [[host_name("steel_gemm_attention_bm_" #bm "_bn_" #bn "_bk_" #bk \
                       "_itype_" #itype)]] [[kernel]] void                  \
  attention<itype, bm, bn, bk, wm, wn, false, true, false, false, true>(    \
      const device itype* Q [[buffer(0)]],                                  \
      const device itype* K [[buffer(1)]],                                  \
      const device itype* V [[buffer(2)]],                                  \
      device otype* O [[buffer(3)]],                                        \
      const constant MLXFastAttentionParams* params [[buffer(4)]],          \
      const constant int* batch_shape [[buffer(5)]],                        \
      const constant size_t* batch_strides [[buffer(6)]],                   \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                \
      uint3 tid [[threadgroup_position_in_grid]],                           \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_fast_inference_self_attention_kernel_heads(type) \
  instantiate_fast_inference_self_attention_kernel(type, type, 16, 16, 32, 2, 2);         \
  instantiate_fast_inference_self_attention_kernel(type, type, 16, 16, 64, 2, 2);         \
  instantiate_fast_inference_self_attention_kernel(type, type, 16, 16, 96, 2, 2);         \
  instantiate_fast_inference_self_attention_kernel(type, type, 16, 16, 128, 2, 2);        \
  instantiate_fast_inference_self_attention_kernel(type, type, 16, 16, 256, 2, 2);        \

instantiate_fast_inference_self_attention_kernel_heads(float)
instantiate_fast_inference_self_attention_kernel_heads(half)
instantiate_fast_inference_self_attention_kernel_heads(bfloat16_t)

// SDPA vector instantiations
#define instantiate_sdpa_vector(type, head_dim)                              \
  template [[host_name("sdpa_vector_" #type "_" #head_dim)]]                 \
  [[kernel]] void sdpa_vector<type, head_dim>(                               \
      const device type* queries [[buffer(0)]],                              \
      const device type* keys [[buffer(1)]],                                 \
      const device type* values [[buffer(2)]],                               \
      device type* out [[buffer(3)]],                                        \
      const constant int& gqa_factor,                                        \
      const constant int& N,                                                 \
      const constant size_t& k_stride,                                       \
      const constant size_t& v_stride,                                       \
      const constant float& scale,                                           \
      const constant float& softcapping,                                     \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);                          \
  template [[host_name("sdpa_vector_2pass_1_" #type "_" #head_dim)]]         \
  [[kernel]] void sdpa_vector_2pass_1<type, head_dim>(                       \
      const device type* queries [[buffer(0)]],                              \
      const device type* keys [[buffer(1)]],                                 \
      const device type* values [[buffer(2)]],                               \
      device float* out [[buffer(3)]],                                       \
      device float* sums [[buffer(4)]],                                      \
      device float* maxs [[buffer(5)]],                                      \
      const constant int& gqa_factor,                                        \
      const constant int& N,                                                 \
      const constant size_t& k_stride,                                       \
      const constant size_t& v_stride,                                       \
      const constant float& scale,                                           \
      const constant float& softcapping,                                     \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);                          \
  template [[host_name("sdpa_vector_2pass_2_" #type "_" #head_dim)]]         \
  [[kernel]] void sdpa_vector_2pass_2<type, head_dim>(                       \
      const device float* partials [[buffer(0)]],                            \
      const device float* sums [[buffer(1)]],                                \
      const device float* maxs [[buffer(2)]],                                \
      device type* out [[buffer(3)]],                                           \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);                          \

#define instantiate_sdpa_vector_heads(type) \
  instantiate_sdpa_vector(type, 32)         \
  instantiate_sdpa_vector(type, 64)         \
  instantiate_sdpa_vector(type, 96)         \
  instantiate_sdpa_vector(type, 128)         \
  instantiate_sdpa_vector(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on
