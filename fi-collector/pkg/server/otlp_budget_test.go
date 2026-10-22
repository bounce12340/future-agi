package server

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
	"time"
)

// Tests for the process-wide budget on decompressed gzip bytes (inflateBudget).
// The per-request 16 MiB cap bounds one request; these check that many
// concurrent or stalled gzip requests cannot add up past the budget.

func postGzipTo(s *Server, body []byte) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/traces", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/x-protobuf")
	req.Header.Set("Content-Encoding", "gzip")
	rec := httptest.NewRecorder()
	s.handleHTTPTraces(rec, req)
	return rec
}

// zeroGzipMembers is n concatenated gzip members of 1 MiB of zeros each: n MiB
// once inflated, about n KiB on the wire.
func zeroGzipMembers(t *testing.T, n int) []byte {
	t.Helper()
	member := gzipBytes(t, make([]byte, 1<<20))
	return bytes.Repeat(member, n)
}

// A valid gzip request that needs more budget than is free is a 503 with
// Retry-After, and the same request succeeds once the budget is back.
func TestOTLPHTTPGzipOverInflateBudgetIs503(t *testing.T) {
	traces := makeTraces("budget-span", "21212121-2121-4121-8121-212121212121")
	traces.ResourceSpans().At(0).ScopeSpans().At(0).Spans().At(0).
		Attributes().PutStr("raw_log", strings.Repeat("x", 2<<20))
	body := gzipBytes(t, protoBody(t, traces)) // ~2 MiB inflated, a few KiB on the wire

	s := New(Config{BatchMaxRows: 1 << 20, BatchMaxAge: time.Hour}, nil, nil, nil, nil)
	others := inflateLease{b: s.inflate} // in-flight requests holding all but 1 MiB
	if !others.grow(s.inflate.size - 1<<20) {
		t.Fatal("could not occupy the budget")
	}

	rec := postGzipTo(s, body)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503 while the budget is full (%s)", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("503 must carry Retry-After so exporters back off and retry")
	}

	others.release()
	if rec := postGzipTo(s, body); rec.Code != http.StatusOK {
		t.Fatalf("status %d after the budget freed up, want 200 (%s)", rec.Code, rec.Body.String())
	}
	if n := s.inflate.inUse.Load(); n != 0 {
		t.Errorf("%d budget bytes still leased after every request returned", n)
	}
}

// Concurrent gzip bombs share one budget: none of them can hold more than it,
// each is turned away once it would pass it, and every lease is returned.
func TestOTLPHTTPConcurrentGzipBombsStayWithinBudget(t *testing.T) {
	const budget = 4 << 20 // below the 16 MiB per-request cap, so no bomb reaches 413
	s := directServer()
	s.inflate = newInflateBudget(budget)
	bomb := zeroGzipMembers(t, 64) // 64 MiB inflated, ~64 KiB on the wire

	const n = 32
	codes := make([]int, n)
	var wg sync.WaitGroup
	for i := range codes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			codes[i] = postGzipTo(s, bomb).Code
		}()
	}
	wg.Wait()

	for i, c := range codes {
		if c != http.StatusServiceUnavailable {
			t.Errorf("request %d: status %d, want 503 once the shared budget is spent", i, c)
		}
	}
	if p := s.inflate.peak.Load(); p > budget {
		t.Errorf("peak leased %d bytes, over the %d byte budget", p, budget)
	}
	if u := s.inflate.inUse.Load(); u != 0 {
		t.Errorf("%d budget bytes still leased after every request returned", u)
	}
}

// liveHeap is the heap still reachable after a full GC.
func liveHeap() uint64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

// A gzip upload that stalls part way must not hold its inflated bytes: the
// body is inflated only once it has fully arrived, so a stalled request holds
// just the compressed bytes it sent. Otherwise a sender can pin about 1000x
// what it uploads, for as long as it keeps the connection open.
func TestOTLPHTTPStalledGzipUploadHoldsOnlyItsWireBytes(t *testing.T) {
	prefix := zeroGzipMembers(t, 12) // ~12 KiB that inflates to 12 MiB
	pr, pw := io.Pipe()
	req := httptest.NewRequest(http.MethodPost, "/v1/traces", pr)
	req.Header.Set("Content-Type", "application/x-protobuf")
	req.Header.Set("Content-Encoding", "gzip")
	rec := httptest.NewRecorder()
	s := directServer()

	before := liveHeap()
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.handleHTTPTraces(rec, req)
	}()
	// io.Pipe returns from Write only once the handler has read all of it.
	if _, err := pw.Write(prefix); err != nil {
		t.Fatalf("write prefix: %v", err)
	}

	var grew uint64
	for deadline := time.Now().Add(300 * time.Millisecond); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
		if h := liveHeap(); h > before {
			grew = max(grew, h-before)
		}
	}
	_ = pw.CloseWithError(io.ErrUnexpectedEOF) // the sender gives up
	<-done

	if grew > 4<<20 {
		t.Errorf("a stalled upload of %d compressed bytes held %d more live heap bytes; want only its wire bytes",
			len(prefix), grew)
	}
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status %d after the sender gave up, want 400", rec.Code)
	}
	if u := s.inflate.inUse.Load(); u != 0 {
		t.Errorf("%d budget bytes still leased after the request returned", u)
	}
}

func TestGzipInflightBytes(t *testing.T) {
	prev := debug.SetMemoryLimit(-1)
	t.Cleanup(func() { debug.SetMemoryLimit(prev) })

	debug.SetMemoryLimit(1 << 62) // effectively unset
	if got := gzipInflightBytes(0); got != defaultGzipInflightBytes {
		t.Errorf("unset: %d, want the %d default", got, defaultGzipInflightBytes)
	}
	if got := gzipInflightBytes(64); got != 64<<20 {
		t.Errorf("64 MiB configured: %d", got)
	}
	if got := gzipInflightBytes(1); got != maxOTLPHTTPBodyBytes {
		t.Errorf("1 MiB configured: %d, want the %d floor (one full request)", got, maxOTLPHTTPBodyBytes)
	}

	debug.SetMemoryLimit(384 << 20) // a GOMEMLIMIT of 384MiB
	if got := gzipInflightBytes(0); got != 48<<20 {
		t.Errorf("unset under GOMEMLIMIT=384MiB: %d, want an eighth (48 MiB)", got)
	}
	if got := gzipInflightBytes(200); got != 200<<20 {
		t.Errorf("an explicit setting wins over GOMEMLIMIT: %d", got)
	}
}
