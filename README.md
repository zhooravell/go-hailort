# go-hailort

Go bindings for the [Hailo AI processor](https://hailo.ai) via the HailoRT C API.

Load a `.hef` model, run inference, get dequantized float32 tensors — in a few lines of Go.

```go
dev, err := hailort.Open("yolov8n.hef")
if err != nil { log.Fatal(err) }
defer dev.Close()

tensors, err := dev.Infer(context.Background(), rgbBytes)
```

## Requirements

| Requirement | Version                                                     |
|-------------|-------------------------------------------------------------|
| HailoRT     | 4.15 or later (tested on 4.23)                              |
| Go          | 1.22+                                                       |
| CGO_ENABLED | 1                                                           |
| OS          | Linux (arm64 Raspberry Pi 5 or x86_64 with PCIe Hailo card) |

HailoRT must be installed before building:

```bash
# Raspberry Pi 5 — Hailo AI HAT+ (standard Pi OS Bookworm install)
sudo apt install hailort

# Verify
hailortcli fw-control identify
```

Headers are expected at `/usr/include/hailo/hailort.h`  
Library at `/usr/lib/libhailort.so`

## Installation

```bash
go get github.com/zhooravell/go-hailort
```

Build with CGO enabled (required for all CGo packages):

```bash
CGO_ENABLED=1 go build ./...
```

## Usage

### Basic inference (single input model)

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zhooravell/go-hailort"
)

func main() {
	dev, err := hailort.Open("model.hef")
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()

	fmt.Println("inputs: ", dev.InputNames(), "sizes:", dev.InFrameSizes())
	fmt.Println("outputs:", dev.OutputNames())

	// input must be InFrameSize() bytes — pre-quantized UINT8 (e.g. RGB 512×512×3)
	input := make([]byte, dev.InFrameSize())

	tensors, err := dev.Infer(context.Background(), input)
	if err != nil {
		log.Fatal(err)
	}

	for _, t := range tensors {
		fmt.Printf("%s: %d float32 values\n", t.Name, len(t.Data))
	}
}
```

### Context cancellation

Cancelling the context calls `hailo_shutdown_network_group` internally, which
unblocks any in-flight write/read and returns `ctx.Err()`:

```go
ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
defer cancel()

tensors, err := dev.Infer(ctx, input)
if errors.Is(err, context.DeadlineExceeded) {
log.Println("inference timed out")
}
```

### Custom format types

```go
// Input already dequantized on the caller side — pass float32 bytes directly.
dev, err := hailort.Open("model.hef",
hailort.WithInputFormat(hailort.FormatTypeFloat32),
)

// Raw output without dequantization (e.g. for custom post-processing).
dev, err := hailort.Open("model.hef",
hailort.WithOutputFormat(hailort.FormatTypeAuto),
)
```

### Multiple input vstreams

```go
// Provide one []byte slice per input vstream, in InputNames() order.
tensors, err := dev.InferMulti(ctx, [][]byte{inputA, inputB})
```

### Error handling

```go
tensors, err := dev.Infer(ctx, input)
switch {
case err == nil:
// success
case hailort.IsAbort(err):
// device was shut down or context cancelled
case hailort.IsTimeout(err):
// write back-pressure: output not being read fast enough
case hailort.IsDeviceInUse(err):
// another process holds the device
default:
log.Fatal(err)
}
```

## API reference

### Opening a device

```go
func Open(hefPath string, opts ...Option) (*Device, error)
```

| Option                           | Default             | Description             |
|----------------------------------|---------------------|-------------------------|
| `WithInputFormat(f FormatType)`  | `FormatTypeAuto`    | Host-side input format  |
| `WithOutputFormat(f FormatType)` | `FormatTypeFloat32` | Host-side output format |

### Device methods

| Method                                      | Description                                       |
|---------------------------------------------|---------------------------------------------------|
| `InFrameSize() int`                         | Required input buffer size in bytes (first input) |
| `InFrameSizes() []int`                      | Required sizes for all inputs                     |
| `InputNames() []string`                     | vstream name for each input                       |
| `OutputNames() []string`                    | vstream name for each output                      |
| `Infer(ctx, input) ([]Tensor, error)`       | Run inference (single input)                      |
| `InferMulti(ctx, inputs) ([]Tensor, error)` | Run inference (multiple inputs)                   |
| `Close()`                                   | Release hardware resources (idempotent)           |

### Tensor

```go
type Tensor struct {
Name string    // vstream name from HEF
Data []float32 // dequantized values (or raw if FormatTypeAuto)
}
```

## Thread safety

A single `Device` must not be used from multiple goroutines simultaneously.
For parallel inference, create one `Device` per goroutine:

```go
// OK: each goroutine has its own Device
go func () {
dev, _ := hailort.Open("model.hef")
defer dev.Close()
dev.Infer(ctx, input)
}()
```

The Hailo model scheduler on the VDevice handles resource sharing across
multiple `Device` instances within the same process automatically.

## HailoRT version compatibility

| Function                           | Notes                                                                       |
|------------------------------------|-----------------------------------------------------------------------------|
| `hailo_make_input_vstream_params`  | Deprecated since 4.15 (`quantized` arg ignored), still supported in 4.23+   |
| `hailo_make_output_vstream_params` | Same as above                                                               |
| `hailo_shutdown_network_group`     | Added in 4.16, used for context cancellation                                |
| `HAILO_STREAM_ABORT`               | Renamed from `HAILO_STREAM_ABORTED_BY_USER` in 4.15; `IsAbort` handles both |

## License

MIT — see [LICENSE](LICENSE)
