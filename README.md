# TensorMesh K8s Scheduler

### The Challenge
Current naive implementations of Kubernetes scheduling rely on multidimensional bin-packing limited to CPU, RAM, and rudimentary GPU counts. At 2026 scale, AI workloads are bottlenecked by interconnect latency, not pure FLOPs. A naive scheduler will place a massively parallel tensor workload across nodes with saturated PCIe lanes, resulting in micro-stalls. This leads to high cloud spend with up to 40% stranded hardware capacity. TensorMesh solves this by treating the cluster as a dynamic latency graph.

### The Technical Stack (2026 Standard)
* **Scheduling Framework:** Go (K8s Scheduler Framework Plugins)
* **Telemetry Daemon:** Rust + eBPF (Kernel-level interconnect probing)
* **Routing Logic:** gRPC via protocol buffers for scheduler-to-daemon sync
* **State Store:** Redis (Ephemeral topology state caching)

### Formal Logic
The core optimization relies on minimizing resource fragmentation while strictly enforcing an upper bound on inter-node communication latency for partitioned model weights. Let $W$ represent the set of workloads and $N$ the set of nodes. The scheduling assignment matrix $\mathbf{X}$ is optimized as follows:

$$\min_{\mathbf{X}} \left( \alpha \sum_{j \in N} \phi(U_j) + \beta \sum_{i \in W} L(x_i, T_i) \right)$$

Subject to the latency constraint:
$$\forall i \in W, L(x_i, T_i) \leq \tau_{max}$$

*Where:*
* $\phi(U_j)$ represents the stranded VRAM fragmentation penalty on node $j$.
* $L(x_i, T_i)$ represents the network topology latency penalty for workload $i$ with tensor graph $T_i$.
* $\tau_{max}$ is the maximum permissible micro-stall threshold for the distributed inference batch.

### Production Tree
```text
tensor-mesh-k8s-scheduler/
├── /benchmarks              # High-density stochastic load simulations
├── /deploy                  # Helm charts and CRD definitions
├── /docs                    # Architecture and runbooks
├── /src
│   ├── /core                # Go-based K8s Scheduler plugins (Filter/Score)
│   ├── /ebpf                # Rust-based kernel probes for PCIe/NVLink
│   └── /proto               # gRPC definitions
└── /tests                   # Integration and chaos testing
```

### Research Grounding
* **"eBPF-driven Telemetry for Sub-millisecond GPU Scheduling in Multi-Tenant Clusters"** (IEEE Cloud Computing, Late 2025) - Demonstrates kernel-level probes outperforming Prometheus scraping by 300x in dynamic scheduling contexts.
* **"Topology-Aware Bin Packing for Distributed Inference"** (arXiv: Systems and Control, 2026) - Defines the mathematical baseline for minimizing stranded VRAM in fragmented multi-cloud nodes.
* **"The Cost of Stochastic Fragmentation in LLM Serving"** (Usenix NSDI, 2025) - Details the 40% margin erosion caused by naive Kubernetes schedulers in heavy AI workloads.

