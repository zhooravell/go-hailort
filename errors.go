package hailort

import (
	"errors"
	"fmt"
)

// HailoError is returned when a HailoRT C API call fails.
type HailoError struct {
	// Op is the C API function name that failed (e.g. "create_vdevice").
	Op string
	// Status is the raw hailo_status integer code.
	Status int
}

func (e *HailoError) Error() string {
	name, ok := statusNames[e.Status]
	if !ok {
		name = fmt.Sprintf("status_%d", e.Status)
	}
	return fmt.Sprintf("hailort: %s: %s", e.Op, name)
}

// IsAbort reports whether err is caused by a vstream shutdown — either by
// calling Device.Close, context cancellation, or hailo_shutdown_network_group.
// It handles both the old name (HAILO_STREAM_ABORTED_BY_USER, pre-4.15) and
// the current name (HAILO_STREAM_ABORT, 4.15+).
func IsAbort(err error) bool {
	var he *HailoError
	if !errors.As(err, &he) {
		return false
	}
	// Both values are checked for cross-version compatibility.
	return he.Status == hailoStreamAbort || he.Status == hailoStreamAbortedByUser
}

// IsTimeout reports whether err is a HailoRT timeout (HAILO_TIMEOUT).
// This usually means the read goroutine is not draining output fast enough.
func IsTimeout(err error) bool {
	var he *HailoError
	return errors.As(err, &he) && he.Status == hailoTimeout
}

// IsDeviceInUse reports whether err indicates the Hailo device is already
// opened by another process (HAILO_DEVICE_IN_USE).
func IsDeviceInUse(err error) bool {
	var he *HailoError
	return errors.As(err, &he) && he.Status == hailoDeviceInUse
}

// Well-known hailo_status values kept as untyped ints so they can be compared
// against the C enum without importing cgo from this file.
// Values are stable across HailoRT 4.x releases.
const (
	hailoSuccess             = 0
	hailoTimeout             = 4  // HAILO_TIMEOUT
	hailoStreamAbort         = 14 // HAILO_STREAM_ABORT (4.15+)
	hailoStreamAbortedByUser = 32 // HAILO_STREAM_ABORTED_BY_USER (pre-4.15 name, kept for compat)
	hailoDeviceInUse         = 44 // HAILO_DEVICE_IN_USE
	hailoInvalidHEF          = 25 // HAILO_INVALID_HEF
)

// statusNames maps known hailo_status codes to human-readable names.
var statusNames = map[int]string{
	0:  "HAILO_SUCCESS",
	4:  "HAILO_TIMEOUT",
	14: "HAILO_STREAM_ABORT",
	25: "HAILO_INVALID_HEF",
	32: "HAILO_STREAM_ABORTED_BY_USER",
	44: "HAILO_DEVICE_IN_USE",
}
