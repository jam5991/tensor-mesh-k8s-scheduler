.PHONY: all proto core ebpf helm-lint

all: proto core ebpf

proto:
	@echo "Compiling Protocol Buffers..."
	mkdir -p src/core/pkg/proto
	protoc --go_out=src/core/pkg/proto --go_opt=paths=source_relative \
	    --go-grpc_out=src/core/pkg/proto --go-grpc_opt=paths=source_relative \
	    --proto_path=src/proto src/proto/telemetry.proto

core:
	@echo "Building Go Scheduler Plugin..."
	cd src/core && go build -o ../../bin/kube-scheduler-tensormesh main.go

ebpf:
	@echo "Building Rust eBPF Daemon..."
	cd src/ebpf && cargo build --release

helm-lint:
	@echo "Linting Helm Charts..."
	helm lint deploy/helm/tensor-mesh

deploy:
	@echo "Deploying to Kubernetes..."
	helm upgrade --install tensor-mesh deploy/helm/tensor-mesh -n kube-system
