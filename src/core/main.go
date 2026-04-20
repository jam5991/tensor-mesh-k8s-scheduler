package main

import (
	"math/rand"
	"os"
	"time"

	"github.com/jorgemartinez/tensor-mesh-k8s-scheduler/src/core/pkg/tensormesh"
	"k8s.io/component-base/logs"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	
	// Register the TensorMesh plugin with the K8s scheduler framework
	command := app.NewSchedulerCommand(
		app.WithPlugin(tensormesh.Name, tensormesh.New),
	)

	logs.InitLogs()
	defer logs.FlushLogs()

	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
