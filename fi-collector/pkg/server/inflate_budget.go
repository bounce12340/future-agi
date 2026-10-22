package server

import (
	"math"
	"runtime/debug"
	"sync/atomic"

	"golang.org/x/sync/semaphore"
)

// defaultGzipInflightBytes is the inflate budget when the config sets none (or
// an eighth of GOMEMLIMIT when that is lower).
const defaultGzipInflightBytes = 128 << 20

// inflateBudget caps the decompressed bytes that all in-flight gzip OTLP/HTTP
// requests hold at once. The per-request cap alone does not bound memory once
// gzip is accepted: a few KiB on the wire can become a 16 MiB body, so N
// concurrent requests hold N x 16 MiB for the price of N small uploads. Each
// gzip request leases budget as its inflate buffer grows and gives it back
// when the handler returns; a request that cannot grow its lease is answered
// 503 with Retry-After, which OTLP exporters retry.
//
// Uncompressed bodies are not counted. They cost the sender one byte of
// upload for every byte held, exactly as before gzip was accepted.
type inflateBudget struct {
	sem  *semaphore.Weighted
	size int64

	inUse atomic.Int64 // bytes leased right now
	peak  atomic.Int64 // most bytes ever leased at once
}

func newInflateBudget(size int64) *inflateBudget {
	return &inflateBudget{sem: semaphore.NewWeighted(size), size: size}
}

// gzipInflightBytes turns the gzip_inflight_mib setting into bytes. Unset
// means 128 MiB, or an eighth of GOMEMLIMIT when that is lower. The result is
// never below one full request, or a body at the cap could never be decoded.
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
	return max(n, maxOTLPHTTPBodyBytes)
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
