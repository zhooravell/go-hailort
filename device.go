package hailort

/*
#cgo CFLAGS:  -I/usr/include/hailo
#cgo LDFLAGS: -lhailort

#include <hailo/hailort.h>
#include <stdlib.h>
#include <string.h>

// hailo_go_true is a helper because Go cannot take the address of a C literal.
static inline _Bool hailo_go_true(void) { return (_Bool)1; }
*/
import "C"
import (
	"context"
	"fmt"
	"sync"
	"unsafe"
)

// maxVStreams is the maximum number of input or output vstreams per network group.
// Most models have 1 input and 2–6 outputs; 16 covers all known cases.
const maxVStreams = 16

// FormatType is the host-side data type for a vstream buffer.
// It controls whether HailoRT performs quantization/dequantization on the fly.
type FormatType C.hailo_format_type_t

var (
	// FormatTypeAuto matches the device's native format (typically UINT8 for
	// input, UINT8/UINT16 for output). No host-side conversion is performed.
	// Use this when the caller handles quantization manually.
	FormatTypeAuto = FormatType(C.HAILO_FORMAT_TYPE_AUTO)

	// FormatTypeUint8 explicitly requests 1-byte unsigned integer buffers.
	// Equivalent to FormatTypeAuto for pre-quantized UINT8 models.
	FormatTypeUint8 = FormatType(C.HAILO_FORMAT_TYPE_UINT8)

	// FormatTypeFloat32 requests host-side dequantization: HailoRT converts
	// the device's fixed-point output to 32-bit floats automatically.
	// This is the recommended format for output vstreams in most use cases.
	FormatTypeFloat32 = FormatType(C.HAILO_FORMAT_TYPE_FLOAT32)
)

// Tensor holds the result of one output vstream after a successful Infer call.
type Tensor struct {
	// Name is the vstream name as defined in the HEF (e.g. "yolov8n/conv93").
	Name string
	// Data contains dequantized float32 values when the output format is
	// FormatTypeFloat32 (the default), or raw bytes reinterpreted as float32
	// when using FormatTypeAuto or FormatTypeUint8.
	Data []float32
}

// options holds configuration for Open, populated via Option functions.
type options struct {
	inputFormat  C.hailo_format_type_t
	outputFormat C.hailo_format_type_t
}

// Option is a functional option for Open.
type Option func(*options)

// WithInputFormat sets the host-side format for all input vstreams.
// Default: FormatTypeAuto (no conversion, pass pre-quantized UINT8 bytes).
func WithInputFormat(f FormatType) Option {
	return func(o *options) { o.inputFormat = C.hailo_format_type_t(f) }
}

// WithOutputFormat sets the host-side format for all output vstreams.
// Default: FormatTypeFloat32 (HailoRT dequantizes output automatically).
func WithOutputFormat(f FormatType) Option {
	return func(o *options) { o.outputFormat = C.hailo_format_type_t(f) }
}

// Device encapsulates a Hailo VDevice, a loaded HEF model, and the configured
// vstreams needed for inference.
//
// A Device is not safe for concurrent use from multiple goroutines. For parallel
// inference, create one Device per goroutine — the Hailo model scheduler
// handles resource sharing across VDevice instances automatically.
type Device struct {
	vdevice C.hailo_vdevice
	hef     C.hailo_hef
	ng      C.hailo_configured_network_group

	inputs  [maxVStreams]C.hailo_input_vstream
	outputs [maxVStreams]C.hailo_output_vstream
	nIn     int
	nOut    int

	// inFrameSizes holds the expected byte length for each input vstream.
	// Index matches the inputs array.
	inFrameSizes []int
	// inNames holds the vstream name for each input vstream.
	inNames []string

	// outSizes holds the expected byte length for each output vstream.
	outSizes []int
	// outNames holds the vstream name for each output vstream.
	outNames []string

	// mu protects ng during hailo_shutdown_network_group calls that may
	// arrive concurrently from the context watchdog goroutine in Infer.
	mu     sync.Mutex
	closed bool
}

// Open loads a HEF file, creates a VDevice with the Hailo model scheduler,
// and initialises all input and output vstreams.
//
// The caller must call Close when done to release hardware resources.
// Failing to do so prevents other processes from accessing the device.
//
// opts can be used to override the default input/output format types.
// By default inputs use FormatTypeAuto and outputs use FormatTypeFloat32.
func Open(hefPath string, opts ...Option) (*Device, error) {
	cfg := &options{
		inputFormat:  C.hailo_format_type_t(FormatTypeAuto),
		outputFormat: C.hailo_format_type_t(FormatTypeFloat32),
	}
	for _, o := range opts {
		o(cfg)
	}

	d := &Device{}

	// 1. VDevice — connects to the Hailo chip.
	//    Passing nil uses default params (model scheduler enabled, auto device discovery).
	if err := check(C.hailo_create_vdevice(nil, &d.vdevice), "create_vdevice"); err != nil {
		return nil, err
	}

	// 2. Load the HEF (Hailo Executable File) that contains the compiled model.
	cpath := C.CString(hefPath)
	defer C.free(unsafe.Pointer(cpath))
	if err := check(C.hailo_create_hef_file(&d.hef, cpath), "create_hef_file"); err != nil {
		C.hailo_release_vdevice(d.vdevice)
		return nil, err
	}

	// 3. Configure the network group — prepares the model for execution on VDevice.
	//    hailo_init_configure_params_by_vdevice fills in scheduler-compatible defaults.
	var cfgParams C.hailo_configure_params_t
	if err := check(
		C.hailo_init_configure_params_by_vdevice(d.hef, d.vdevice, &cfgParams),
		"init_configure_params",
	); err != nil {
		C.hailo_release_hef(d.hef)
		C.hailo_release_vdevice(d.vdevice)
		return nil, err
	}

	var ngCount C.size_t = 1
	if err := check(
		C.hailo_configure_vdevice(d.vdevice, d.hef, &cfgParams, &d.ng, &ngCount),
		"configure_vdevice",
	); err != nil {
		C.hailo_release_hef(d.hef)
		C.hailo_release_vdevice(d.vdevice)
		return nil, err
	}

	// 4. Input vstreams — one per model input tensor.
	//    hailo_make_input_vstream_params is deprecated since 4.15 (the `quantized`
	//    argument is now ignored), but the function remains fully supported in 4.23+.
	var inParams [maxVStreams]C.hailo_input_vstream_params_by_name_t
	nIn := C.size_t(maxVStreams)
	if err := check(
		C.hailo_make_input_vstream_params(
			d.ng, C.hailo_go_true(), cfg.inputFormat, &inParams[0], &nIn,
		),
		"make_input_vstream_params",
	); err != nil {
		return nil, d.closeWithErr(err)
	}
	if err := check(
		C.hailo_create_input_vstreams(d.ng, &inParams[0], nIn, &d.inputs[0]),
		"create_input_vstreams",
	); err != nil {
		return nil, d.closeWithErr(err)
	}
	d.nIn = int(nIn)

	d.inFrameSizes = make([]int, d.nIn)
	d.inNames = make([]string, d.nIn)
	for i := 0; i < d.nIn; i++ {
		d.inNames[i] = C.GoString((*C.char)(unsafe.Pointer(&inParams[i].name[0])))
		var sz C.size_t
		C.hailo_get_input_vstream_frame_size(d.inputs[i], &sz)
		d.inFrameSizes[i] = int(sz)
	}

	// 5. Output vstreams — one per model output tensor.
	var outParams [maxVStreams]C.hailo_output_vstream_params_by_name_t
	nOut := C.size_t(maxVStreams)
	if err := check(
		C.hailo_make_output_vstream_params(
			d.ng, C.hailo_go_true(), cfg.outputFormat, &outParams[0], &nOut,
		),
		"make_output_vstream_params",
	); err != nil {
		return nil, d.closeWithErr(err)
	}
	if err := check(
		C.hailo_create_output_vstreams(d.ng, &outParams[0], nOut, &d.outputs[0]),
		"create_output_vstreams",
	); err != nil {
		return nil, d.closeWithErr(err)
	}
	d.nOut = int(nOut)

	d.outNames = make([]string, d.nOut)
	d.outSizes = make([]int, d.nOut)
	for i := 0; i < d.nOut; i++ {
		d.outNames[i] = C.GoString((*C.char)(unsafe.Pointer(&outParams[i].name[0])))
		var sz C.size_t
		C.hailo_get_output_vstream_frame_size(d.outputs[i], &sz)
		d.outSizes[i] = int(sz)
	}

	return d, nil
}

// InFrameSize returns the required input buffer size in bytes for the first
// (or only) input vstream. For models with a single input this is H×W×C.
//
// For models with multiple inputs use InFrameSizes.
func (d *Device) InFrameSize() int {
	if len(d.inFrameSizes) == 0 {
		return 0
	}
	return d.inFrameSizes[0]
}

// InFrameSizes returns the required buffer sizes in bytes for each input vstream,
// in the same order as InputNames.
func (d *Device) InFrameSizes() []int { return d.inFrameSizes }

// InputNames returns the vstream name for each input, in the order expected by
// the inputs parameter of InferMulti.
func (d *Device) InputNames() []string { return d.inNames }

// OutputNames returns the vstream name for each output tensor, in the same
// order as the Tensor slice returned by Infer.
func (d *Device) OutputNames() []string { return d.outNames }

// Infer runs one inference pass on a model with a single input vstream.
//
// input must be exactly InFrameSize() bytes long. The bytes must be in the
// format expected by the input vstream (UINT8 RGB for most vision models when
// using the default FormatTypeAuto).
//
// Infer writes input to the Hailo pipeline and reads all output vstreams
// concurrently, following the canonical HailoRT pattern. Both operations are
// blocking: Infer returns only after all output data has been received.
//
// Cancelling ctx causes hailo_shutdown_network_group to be called, which
// unblocks the pipeline and makes Infer return ctx.Err(). Any in-progress
// Infer on the same Device must complete (or be aborted via context) before
// the next call is made.
//
// For models with multiple input vstreams, use InferMulti.
func (d *Device) Infer(ctx context.Context, input []byte) ([]Tensor, error) {
	if d.nIn == 0 {
		return nil, fmt.Errorf("hailort: device has no input vstreams")
	}
	if len(input) != d.inFrameSizes[0] {
		return nil, fmt.Errorf("hailort: input len %d != expected %d", len(input), d.inFrameSizes[0])
	}
	return d.InferMulti(ctx, [][]byte{input})
}

// InferMulti runs one inference pass on a model with one or more input vstreams.
//
// inputs[i] corresponds to InputNames()[i] and must be exactly InFrameSizes()[i]
// bytes long. For single-input models, prefer Infer.
func (d *Device) InferMulti(ctx context.Context, inputs [][]byte) ([]Tensor, error) {
	if len(inputs) != d.nIn {
		return nil, fmt.Errorf("hailort: got %d inputs, model has %d", len(inputs), d.nIn)
	}
	for i, inp := range inputs {
		if len(inp) != d.inFrameSizes[i] {
			return nil, fmt.Errorf(
				"hailort: input[%d] len %d != expected %d", i, len(inp), d.inFrameSizes[i],
			)
		}
	}

	nOut := d.nOut
	results := make([]Tensor, nOut)
	errs := make([]error, d.nIn+nOut)

	// Watchdog goroutine: abort the Hailo pipeline when ctx is cancelled.
	// hailo_shutdown_network_group unblocks all in-flight write/read calls
	// with HAILO_STREAM_ABORT, allowing the goroutines below to return.
	stopped := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			d.mu.Lock()
			if !d.closed {
				C.hailo_shutdown_network_group(d.ng)
			}
			d.mu.Unlock()
		case <-stopped:
		}
	}()

	var wg sync.WaitGroup

	// Write goroutine per input vstream (typically just one).
	for i := 0; i < d.nIn; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			inp := inputs[idx]
			errs[idx] = check(
				C.hailo_vstream_write_raw_buffer(
					d.inputs[idx],
					unsafe.Pointer(&inp[0]),
					C.size_t(len(inp)),
				),
				"write_raw_buffer",
			)
		}(i)
	}

	// Read goroutine per output vstream.
	for i := 0; i < nOut; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sz := d.outSizes[idx]
			data := make([]float32, sz/4)
			if err := check(
				C.hailo_vstream_read_raw_buffer(
					d.outputs[idx],
					unsafe.Pointer(&data[0]),
					C.size_t(sz),
				),
				fmt.Sprintf("read_raw_buffer[%d]", idx),
			); err != nil {
				errs[d.nIn+idx] = err
				return
			}
			results[idx] = Tensor{Name: d.outNames[idx], Data: data}
		}(i)
	}

	wg.Wait()
	close(stopped) // signal watchdog to exit if context is still active

	// Return context error in preference to HAILO_STREAM_ABORT so callers can
	// distinguish a deliberate cancellation from a hardware failure.
	for _, err := range errs {
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, err
		}
	}
	return results, nil
}

// Close releases all HailoRT resources in the required order:
//
//  1. input vstreams
//  2. output vstreams
//  3. HEF
//  4. VDevice
//
// Close is safe to call multiple times. Any blocked Infer call should be
// cancelled via its context before calling Close; otherwise Close blocks until
// the inference goroutines finish.
func (d *Device) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return
	}
	d.closed = true

	if d.nIn > 0 {
		C.hailo_release_input_vstreams(&d.inputs[0], C.size_t(d.nIn))
	}
	if d.nOut > 0 {
		C.hailo_release_output_vstreams(&d.outputs[0], C.size_t(d.nOut))
	}
	C.hailo_release_hef(d.hef)
	C.hailo_release_vdevice(d.vdevice)
}

// check converts a non-SUCCESS hailo_status into a *HailoError.
func check(s C.hailo_status, op string) error {
	if s == C.HAILO_SUCCESS {
		return nil
	}
	return &HailoError{Op: op, Status: int(s)}
}

// closeWithErr releases partially-initialised resources and returns err.
// Used internally when Open fails after some resources were already created.
func (d *Device) closeWithErr(err error) error {
	// Release only what was successfully created.
	// vstreams are released first if any were created.
	if d.nIn > 0 {
		C.hailo_release_input_vstreams(&d.inputs[0], C.size_t(d.nIn))
	}
	if d.nOut > 0 {
		C.hailo_release_output_vstreams(&d.outputs[0], C.size_t(d.nOut))
	}
	C.hailo_release_hef(d.hef)
	C.hailo_release_vdevice(d.vdevice)

	return err
}
