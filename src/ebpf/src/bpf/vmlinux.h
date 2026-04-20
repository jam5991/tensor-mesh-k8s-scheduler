/* vmlinux.h - Minimal vendored kernel type definitions for BPF programs.
 *
 * In production, this file is auto-generated from a running Linux kernel via:
 *   bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
 *
 * This stub provides the subset of types required for cross-compilation
 * on non-Linux hosts (e.g., macOS dev machines targeting Linux k8s nodes).
 */

#ifndef __VMLINUX_H__
#define __VMLINUX_H__

typedef unsigned char       __u8;
typedef unsigned short      __u16;
typedef unsigned int        __u32;
typedef unsigned long long  __u64;
typedef signed char         __s8;
typedef signed short        __s16;
typedef signed int          __s32;
typedef signed long long    __s64;

/* BPF map types */
enum bpf_map_type {
    BPF_MAP_TYPE_UNSPEC        = 0,
    BPF_MAP_TYPE_HASH          = 1,
    BPF_MAP_TYPE_ARRAY         = 2,
    BPF_MAP_TYPE_PERF_EVENT_ARRAY = 4,
    BPF_MAP_TYPE_RINGBUF       = 27,
};

#endif /* __VMLINUX_H__ */
