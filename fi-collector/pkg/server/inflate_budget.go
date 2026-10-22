package server

import (
	"math"
	"runtime/debug"
	"sync/atomic"
	"time"

	"golang.org/x/sync/semaphore"
)

// defaultGzipInflightBytes is the inflate budget when the config sets none (or
// an eighth of GOMEMLIMIT when that is lower).
const defaultGzipInflightBytes = 128 << 20

// minGzipInflightBytes is the smallest budget: a full-size body of ordinary
// spans (16 MiB decompressed, a few thousand spans with large attributes)
// leases about 60 MiB, and must always be able to fit.
const minGzipInflightBytes = 4 * maxOTLPHTTPBodyBytes

// inflateBudget caps the memory that all in-flight gzip OTLP/HTTP requests
// hold at once. The per-request cap alone does not bound memory once gzip is
// accepted: a few KiB on the wire can become a 16 MiB body, and a 16 MiB body
// of small elements can decode into gigabytes. So each gzip request leases
// budget as its inflate buffer grows, then leases the estimated cost of
// decoding and converting the body (decode_cost.go) before decoding it, and
// gives it all back when the handler returns. A request that cannot get its
// lease is answered 503 with Retry-After, which OTLP exporters retry, or 413
// when its lease alone would be larger than the whole budget.
//
// Uncompressed bodies are not counted. They cost the sender one byte of
// upload for every byte decoded, exactly as before gzip was accepted.
type inflateBudget struct {
	sem  *semaphore.Weighted
	size int64

	inUse atomic.Int64 // bytes leased right now
	peak  atomic.Int64 // most bytes ever leased at once

	// One log line per second per rejection reason, so a flood of cheap
	// rejected requests is not also a flood of log lines.
	overCapLog, busyLog, decodeCostLog logSample
}

func newInflateBudget(size int64) *inflateBudget {
	return &inflateBudget{sem: semaphore.NewWeighted(size), size: size}
}

// gzipInflightBytes turns the gzip_inflight_mib setting into bytes. Unset
// means 128 MiB, or an eighth of GOMEMLIMIT when that is lower. The result is
// never below minGzipInflightBytes, or a full-size body could never be
// decoded. Peak heap under a flood of gzip requests measured up to about 5x
// the budget (the Go GC lets garbage build up between cycles), so an eighth
// keeps it under GOMEMLIMIT.
func gzipInflightBytes(mib int) int64 {
	n := int64(defaultGzipInflightBytes)
	switch {
	case mib > 0:
		n = int64(min(mib, 1<<16)) << 20
	default:
		if lim := debug.SetMemoryLimit(-1); lim < math.MaxInt64 && lim/8 < n {
			n = lim / 8
		}
	}
	return max(n, minGzipInflightBytes)
}

// inflateLease is one request's share of an inflateBudget. The zero lease of
// a budget holds nothing; release returns whatever it grew to.
type inflateLease struct {
	b    *inflateBudget
	held int64
}

// grow takes n more bytes of budget without waiting, and reports whether it
// could.
func (l *inflateLease) grow(n int64) bool {
	if n <= 0 {
		return true
	}
	if !l.b.sem.TryAcquire(n) {
		return false
	}
	l.held += n
	now := l.b.inUse.Add(n)
	for p := l.b.peak.Load(); now > p && !l.b.peak.CompareAndSwap(p, now); p = l.b.peak.Load() {
	}
	return true
}

func (l *inflateLease) release() {
	if l.held == 0 {
		return
	}
	l.b.inUse.Add(-l.held)
	l.b.sem.Release(l.held)
	l.held = 0
}

// logSample lets one log line through per second and counts the others.
type logSample struct {
	next       atomic.Int64 // unix nanoseconds before which lines are held back
	suppressed atomic.Int64
}

// allow reports whether to log now, and how many lines were held back since
// the last one that was logged.
func (l *logSample) allow(now time.Time) (bool, int64) {
	t := now.UnixNano()
	next := l.next.Load()
	if t < next || !l.next.CompareAndSwap(next, t+int64(time.Second)) {
		l.suppressed.Add(1)
		return false, 0
	}
	return true, l.suppressed.Swap(0)
}
