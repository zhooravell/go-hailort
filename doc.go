// Package hailort provides Go bindings for the Hailo AI processor via the
// HailoRT C API (libhailort).
//
// # Requirements
//
//   - HailoRT 4.15 or later (tested on 4.23)
//   - Linux (arm64 Raspberry Pi 5 or x86_64 with PCIe Hailo card)
//   - CGO_ENABLED=1
//   - libhailort.so in the linker path (/usr/lib/)
//   - Hailo headers in /usr/include/hailo/
//
// # Quick start
//
//	dev, err := hailort.Open("model.hef")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer dev.Close()
//
//	tensors, err := dev.Infer(context.Background(), rgbBytes)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	for _, t := range tensors {
//	    fmt.Println(t.Name, len(t.Data), "float32 values")
//	}
//
// # Thread safety
//
// A single Device must not be used concurrently from multiple goroutines.
// For parallel inference, create one Device per goroutine — each VDevice
// holds its own connection to the hardware and the Hailo model scheduler
// handles resource allocation automatically.
//
// # Format types
//
// By default, input vstreams expect raw UINT8 bytes (pre-quantized, matching
// the model's quantization parameters). Output vstreams return dequantized
// FLOAT32 values. Use [WithInputFormat] and [WithOutputFormat] to override.
//
// # Compatibility
//
// This package targets the HailoRT C API. The following status codes have been
// renamed across versions:
//
//   - HAILO_STREAM_ABORTED_BY_USER → HAILO_STREAM_ABORT (since 4.15)
//
// [IsAbort] handles both variants.
package hailort
