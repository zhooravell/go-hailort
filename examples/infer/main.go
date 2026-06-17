// Example: load a HEF model and run one inference pass.
//
// Build on Raspberry Pi 5 (requires HailoRT installed):
//
//	CGO_ENABLED=1 go build -o infer .
//
// Run:
//
//	./infer --model model.hef --input frame_512x512_rgb.bin
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/zhooravell/go-hailort"
)

func main() {
	model := flag.String("model", "", "path to HEF file (required)")
	input := flag.String("input", "", "path to raw input file: H×W×3 UINT8 RGB bytes")
	flag.Parse()

	if *model == "" {
		log.Fatal("--model is required")
	}

	dev, err := hailort.Open(*model)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer dev.Close()

	fmt.Printf("inputs:  %v  sizes: %v\n", dev.InputNames(), dev.InFrameSizes())
	fmt.Printf("outputs: %v\n", dev.OutputNames())

	var inputBytes []byte
	if *input != "" {
		inputBytes, err = os.ReadFile(*input)
		if err != nil {
			log.Fatalf("read input: %v", err)
		}
	} else {
		// Synthesise a zero-filled frame when no input file is provided.
		inputBytes = make([]byte, dev.InFrameSize())
		fmt.Printf("no --input given, using %d zero bytes\n", len(inputBytes))
	}

	tensors, err := dev.Infer(context.Background(), inputBytes)
	if err != nil {
		log.Fatalf("infer: %v", err)
	}

	for _, t := range tensors {
		min, max := t.Data[0], t.Data[0]
		for _, v := range t.Data {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		fmt.Printf("  %-40s  len=%-8d  min=%.4f  max=%.4f\n",
			t.Name, len(t.Data), min, max)
	}
}
