package server

import (
	"context"
	"errors"
	"math"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/semaphore"
)

// defaultGzipInflightBytes is the inflate budget when the config sets none (or
// an eighth of GOMEMLIMIT when that is lower).
const defaultGzipInflightBytes = 128 << 20

// minGzipInflightBytes is the smallest budget: one body at the 16 MiB cap must
// always fit, or it could never be decompressed.
const minGzipInflightBytes = maxOTLPHTTPBodyBytes

// gzipInflightWait is how long a gzip request waits for its share of the
// budget before it gets 503 + Retry-After.
const gzipInflightWait = 2 * time.Second

// inflateBudget caps the decompressed bytes that all in-flight gzip OTLP/HTTP
// requests hold at once. The 16 MiB cap bounds one request. A gzip request is
// read off the wire and measured (see inflateGzip) before it leases anything;
// it then leases exactly its decompressed size, inflates into a buffer of that
// size, and holds the lease until the handler returns. A body that is corrupt
// or over the cap never takes a lease.
//
// The lease counts decompressed body bytes only. It does not count the
// compressed bytes read off the wire, or what decoding the body allocates: the
// decoded request and its rows can take several times the body size, for gzip
// and plain bodies alike (TH-8745). Uncompressed bodies are not counted.
//
// Waiters are served in order (golang.org/x/sync/semaphore is FIFO), so a large
// body is not starved by a stream of small ones. A request that does not get
// its lease within wait is answered 503 with Retry-After, which OTLP exporters
// retry.
type inflateBudget struct {
	sem  *semaphore.Weighted
	size int64
	wait time.Duration

	inUse atomic.Int64 // bytes leased right now; logged when a request is turned away
	peak  atomic.Int64 // most bytes ever leased at once; read by tests

	// One log line per second per rejection reason, so a flood of cheap
	// rejected requests is not also a flood of log lines.
	overCapLog, busyLog logSample
}

func newInflateBudget(size int64) *inflateBudget {
	return &inflateBudget{sem: semaphore.NewWeighted(size), size: size, wait: gzipInflightWait}
}

// gzipInflightBytes turns the gzip_inflight_mib setting into bytes. Unset
// means 128 MiB, or an eighth of GOMEMLIMIT when that is lower. The result is
// never below minGzipInflightBytes, so one full-size body always fits.
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

// errInflateBudgetBusy reports a gzip request that did not get its reservation
// in time.
var errInflateBudgetBusy = errors.New("inflate budget busy")

// reserve leases n bytes, waiting up to b.wait (or until ctx ends) behind
// earlier waiters. n is a body's decompressed size, at most the 16 MiB cap.
// The caller releases the lease when the handler returns, or on error.
func (b *inflateBudget) reserve(ctx context.Context, n int64) (*inflateLease, error) {
	if n < 0 || n > b.size {
		return nil, errInflateBudgetBusy
	}
	ctx, cancel := context.WithTimeout(ctx, b.wait)
	defer cancel()
	if err := b.sem.Acquire(ctx, n); err != nil {
		return nil, errInflateBudgetBusy
	}
	now := b.inUse.Add(n)
	for p := b.peak.Load(); now > p && !b.peak.CompareAndSwap(p, now); p = b.peak.Load() {
	}
	return &inflateLease{b: b, held: n}, nil
}

// inflateLease is a request's in-flight share of a budget.
type inflateLease struct {
	b    *inflateBudget
	held int64
	once sync.Once
}

// release returns the lease and may be called more than once.
func (l *inflateLease) release() {
	l.once.Do(func() {
		l.b.inUse.Add(-l.held)
		l.b.sem.Release(l.held)
	})
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
