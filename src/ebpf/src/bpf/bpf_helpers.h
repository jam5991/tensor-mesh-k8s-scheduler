/* bpf_helpers.h - Vendored BPF helper macros for cross-compilation.
 *
 * These definitions mirror libbpf's bpf_helpers.h so that the eBPF C code
 * compiles cleanly on macOS dev hosts while remaining valid for the
 * Linux BPF target (clang -target bpf).
 */

#ifndef __BPF_HELPERS_H__
#define __BPF_HELPERS_H__

/* Section attribute macro used by the BPF loader to identify programs/maps.
 * On macOS (IDE-only), we strip the section() attr to avoid Mach-O format
 * errors since this code is never compiled natively — it is always
 * cross-compiled with `clang -target bpf` on a Linux build host or CI. */
#include "vmlinux.h"
#ifdef __APPLE__
#define SEC(name) __attribute__((used))
#else
#define SEC(name) __attribute__((section(name), used))
#endif

/* BTF-style map definition macros */
#define __uint(name, val) int(*name)[val]
#define __type(name, val) typeof(val) *name

/* BPF helper function prototypes (subset used by this project).
 * Each helper is identified by its BPF syscall number, cast to the
 * matching function-pointer type via typeof() to satisfy strict C
 * type-checking (Clang forbids void* → function-pointer conversions). */
static void *(*bpf_map_lookup_elem)(void *map, const void *key) =
    (typeof(bpf_map_lookup_elem))1;
static long (*bpf_map_update_elem)(void *map, const void *key,
                                   const void *value, __u64 flags) =
    (typeof(bpf_map_update_elem))2;
static long (*bpf_map_delete_elem)(void *map, const void *key) =
    (typeof(bpf_map_delete_elem))3;
static long (*bpf_probe_read)(void *dst, __u32 size,
                              const void *src) = (typeof(bpf_probe_read))4;
static __u64 (*bpf_ktime_get_ns)(void) = (typeof(bpf_ktime_get_ns))5;
static long (*bpf_trace_printk)(const char *fmt, __u32 fmt_size,
                                ...) = (typeof(bpf_trace_printk))6;

#endif /* __BPF_HELPERS_H__ */
