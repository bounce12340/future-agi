package server

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.opentelemetry.io/collector/pdata/ptrace/ptraceotlp"
)

// Tests for decode amplification on OTLP/HTTP. A body that is small once
// decompressed can still decode into far more heap than its size: 16 MiB of
// empty ResourceSpans is 8.4M of them (over 1 GiB of heap), and an empty span
// becomes a ~4 KiB row. Gzip makes such a body cost ~16 KiB to send, so a gzip
// request leases what its decoded body will hold, counted before decoding.

// bestGzip compresses at the highest level, as an attacker would.
func bestGzip(t *testing.T, raw []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw, err := gzip.NewWriterLevel(&buf, gzip.BestCompression)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := zw.Write(raw); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func postBody(s *Server, contentType string, gzipped bool, header map[string]string, body []byte) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/traces", bytes.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	if gzipped {
		req.Header.Set("Content-Encoding", "gzip")
	}
	for k, v := range header {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	s.handleHTTPTraces(rec, req)
	return rec
}

// totalAllocDuring reports the bytes allocated (live or not) while fn ran.
func totalAllocDuring(fn func()) uint64 {
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)
	fn()
	runtime.ReadMemStats(&m1)
	return m1.TotalAlloc - m0.TotalAlloc
}

// acceptingServer enqueues accepted rows and never flushes them.
func acceptingServer(budget int64) *Server {
	s := New(Config{BatchMaxRows: 1 << 30, BatchMaxAge: time.Hour}, nil, nil, nil, nil)
	s.inflate = newInflateBudget(budget)
	return s
}

// A small gzip body whose decoded form would hold far more heap than the
// budget is refused with 413 before it is decoded: the handler allocates
// little more than the decompressed bytes, and quickly.
func TestOTLPHTTPGzipDecodeAmplificationIs413(t *testing.T) {
	cases := []struct {
		name string
		ct   string
		raw  []byte
	}{
		{"1M empty ResourceSpans", "application/x-protobuf", pbRepeat(1<<20, pbField(1, nil))},
		{"64K empty spans", "application/x-protobuf", pbField(1, pbField(2, pbRepeat(1<<16, pbField(2, nil))))},
		{"JSON 64K null spans", "application/json",
			[]byte(`{"resourceSpans":[{"scopeSpans":[{"spans":[` + jsonList(1<<16, `null`) + `]}]}]}`)},
		{"JSON 1M empty ResourceSpans", "application/json", []byte(`{"resourceSpans":[` + jsonList(1<<20, `{}`) + `]}`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := acceptingServer(64 << 20)
			body := bestGzip(t, tc.raw)
			var rec *httptest.ResponseRecorder
			alloc := totalAllocDuring(func() {
				rec = postBody(s, tc.ct, true, map[string]string{"X-Project-Name": "p"}, body)
			})
			if rec.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status %d, want 413 for a %d byte gzip body (%d inflated) (%s)",
					rec.Code, len(body), len(tc.raw), rec.Body.String())
			}
			if limit := uint64(3*len(tc.raw)) + 8<<20; alloc > limit {
				t.Errorf("handler allocated %d bytes for a %d byte gzip body; want under %d (no decoding)",
					alloc, len(body), limit)
			}
			if u := s.inflate.inUse.Load(); u != 0 {
				t.Errorf("%d budget bytes still leased after the request returned", u)
			}
		})
	}
}

// The reviewers' case at full size: ~16 KiB of gzip that inflates to 16 MiB of
// empty ResourceSpans (8.4M of them) under the default budget.
func TestOTLPHTTPGzipEmptyResourceSpansBombAtTheCapIs413(t *testing.T) {
	raw := pbRepeat(maxOTLPHTTPBodyBytes/2, pbField(1, nil))
	body := bestGzip(t, raw)
	s := directServer()
	var rec *httptest.ResponseRecorder
	start := time.Now()
	alloc := totalAllocDuring(func() {
		rec = postBody(s, "application/x-protobuf", true, map[string]string{"X-Project-Name": "p"}, body)
	})
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d, want 413 (%s)", rec.Code, rec.Body.String())
	}
	if alloc > 48<<20 {
		t.Errorf("handler allocated %d bytes for a %d byte gzip body; want only the inflated body", alloc, len(body))
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Errorf("took %v to refuse", d)
	}
}

// A valid, highly compressible request (many small spans) leases its decode
// cost, not just its decompressed size: refused with 413 when that cost can
// never fit in the budget, accepted when it fits, and the lease taken while
// it ran is at least the estimate.
func TestOTLPHTTPGzipCompressibleValidRequestLeasesItsDecodeCost(t *testing.T) {
	raw, err := ptraceotlp.NewExportRequestFromTraces(spansWithAttrs(5000, 8)).MarshalProto()
	if err != nil {
		t.Fatal(err)
	}
	body := bestGzip(t, raw)
	est, err := estimateDecodeCost(raw, false)
	if err != nil {
		t.Fatal(err)
	}

	small := acceptingServer(16 << 20)
	if rec := postBody(small, "application/x-protobuf", true, nil, body); rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("budget 16 MiB: status %d, want 413 (the decode cost alone is over it) (%s)", rec.Code, rec.Body.String())
	}

	big := acceptingServer(256 << 20)
	if rec := postBody(big, "application/x-protobuf", true, nil, body); rec.Code != http.StatusOK {
		t.Fatalf("budget 256 MiB: status %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	if p := big.inflate.peak.Load(); p < est || est < 16<<20 {
		t.Errorf("leased %d bytes for a request estimated at %d; want the estimate (over 16 MiB) leased", p, est)
	}
	if u := big.inflate.inUse.Load(); u != 0 {
		t.Errorf("%d budget bytes still leased after the request returned", u)
	}

	// The same body uncompressed is not budgeted (it costs the sender one
	// byte of upload per byte), as before gzip was accepted.
	if rec := postBody(small, "application/x-protobuf", false, nil, raw); rec.Code != http.StatusOK {
		t.Errorf("uncompressed: status %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
}

// Many concurrent compressible requests share one budget: the decode cost of
// the requests admitted at once never passes it, and the heap stays within a
// small multiple of it.
func TestOTLPHTTPConcurrentCompressibleGzipStaysWithinBudget(t *testing.T) {
	raw, err := ptraceotlp.NewExportRequestFromTraces(spansWithAttrs(8000, 8)).MarshalProto()
	if err != nil {
		t.Fatal(err)
	}
	body := bestGzip(t, raw)
	const budget = 64 << 20
	s := acceptingServer(budget)

	runtime.GC()
	var base runtime.MemStats
	runtime.ReadMemStats(&base)
	var peak atomic.Uint64
	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		var m runtime.MemStats
		for {
			select {
			case <-stop:
				return
			default:
			}
			runtime.ReadMemStats(&m)
			if m.HeapInuse > peak.Load() {
				peak.Store(m.HeapInuse)
			}
			time.Sleep(time.Millisecond)
		}
	}()

	const n = 12
	codes := make([]int, n)
	var wg sync.WaitGroup
	for i := range codes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			codes[i] = postBody(s, "application/x-protobuf", true, nil, body).Code
		}()
	}
	wg.Wait()
	close(stop)
	<-sampled

	ok := 0
	for i, c := range codes {
		switch c {
		case http.StatusOK:
			ok++
		case http.StatusServiceUnavailable:
		default:
			t.Errorf("request %d: status %d, want 200 or 503", i, c)
		}
	}
	if ok == 0 {
		t.Error("no request was admitted; one at a time fits the budget")
	}
	if p := s.inflate.peak.Load(); p > budget {
		t.Errorf("peak leased %d bytes, over the %d byte budget", p, budget)
	}
	if grew := peak.Load() - base.HeapInuse; grew > 4*budget {
		t.Errorf("heap grew by %d bytes, over 4x the %d byte budget", grew, budget)
	}
	if u := s.inflate.inUse.Load(); u != 0 {
		t.Errorf("%d budget bytes still leased after every request returned", u)
	}
}

// Protobuf nesting deeper than any real attribute is refused before decoding,
// compressed or not: the decoder recurses once per level, and a 16 MiB body
// nested ~2M levels deep overflows the goroutine stack, which kills the
// process.
func TestOTLPHTTPDeeplyNestedProtobufIs400(t *testing.T) {
	raw := pbDeepAttr(maxDecodeNesting)
	for _, gz := range []bool{false, true} {
		body := raw
		if gz {
			body = bestGzip(t, raw)
		}
		rec := postBody(directServer(), "application/x-protobuf", gz, nil, body)
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "nest") {
			t.Errorf("gzip=%v: status %d %q, want 400 naming the nesting", gz, rec.Code, rec.Body.String())
		}
	}
}

// A gzip body is inflated into one buffer of its exact size, leased before
// it is allocated: in one pass when its trailer is plausible, and after
// measuring it when the trailer declares a ratio no OTLP batch has.
func TestInflateGzipAllocatesTheExactSizeOnce(t *testing.T) {
	words := make([]string, 0, 300000)
	for i := 0; len(words) < cap(words); i++ {
		words = append(words, strconv.FormatUint(uint64(i)*2654435761%1000003, 36))
	}
	cases := map[string][]byte{
		"plausible ratio":         []byte(strings.Join(words, " ")),
		"ratio past the trusted":  bytes.Repeat([]byte("otlp "), 3<<20/5),
		"empty body":              nil,
		"smaller than min buffer": []byte("tiny"),
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			wire := gzipBytes(t, raw)
			lease := inflateLease{b: newInflateBudget(64 << 20)}
			defer lease.release()
			var out []byte
			var err error
			alloc := totalAllocDuring(func() { out, err = inflateGzip(wire, maxOTLPHTTPBodyBytes, &lease) })
			if err != nil || !bytes.Equal(out, raw) {
				t.Fatalf("inflate: err=%v, %d bytes out, want %d", err, len(out), len(raw))
			}
			if want := max(int64(len(raw)), minInflateBuf); lease.held != want {
				t.Errorf("leased %d bytes for a %d byte body, want %d", lease.held, len(raw), want)
			}
			if alloc > uint64(len(raw))+1<<20 {
				t.Errorf("allocated %d bytes to inflate a %d byte body (ratio %d)", alloc, len(raw), len(raw)/max(len(wire), 1))
			}
		})
	}
}

// A trailer can understate the size (a multi-member body declares only its
// last member). Such a body is measured and inflated again at its real size.
func TestInflateGzipBodyLongerThanItsTrailer(t *testing.T) {
	raw := bytes.Repeat([]byte("span "), 1<<20/5)
	wire := append(gzipBytes(t, raw), gzipBytes(t, []byte("x"))...) // last member: 1 byte
	if gzipSizeHint(wire) != 1 {
		t.Fatalf("hint %d, want the last member's size", gzipSizeHint(wire))
	}
	lease := inflateLease{b: newInflateBudget(64 << 20)}
	defer lease.release()
	out, err := inflateGzip(wire, maxOTLPHTTPBodyBytes, &lease)
	if err != nil || !bytes.Equal(out, append(raw, 'x')) {
		t.Fatalf("inflate: err=%v, %d bytes out, want %d", err, len(out), len(raw)+1)
	}
	if lease.held != int64(len(out)) {
		t.Errorf("leased %d bytes for a %d byte body", lease.held, len(out))
	}

	big := make([]byte, 24)
	binary.LittleEndian.PutUint32(big[20:], 1<<31)
	if h := gzipSizeHint(big); h != 1<<31 {
		t.Errorf("hint %d, want the trailer's value", h)
	}
}

// A decompression bomb is refused without allocating a buffer for it: one
// whose trailer is honest is refused from the trailer alone, one that hides
// its size is measured without keeping the bytes.
func TestOTLPHTTPGzipBombAllocatesNoInflateBuffer(t *testing.T) {
	zeros := gzipBytes(t, make([]byte, 64<<20))
	cases := map[string][]byte{
		"trailer declares 64 MiB":        zeros,
		"trailer declares 1 byte":        append(append([]byte(nil), zeros...), gzipBytes(t, []byte{0})...),
		"256 members of 1 MiB of zeros":  zeroGzipMembers(t, 256),
		"trailer declares exactly 16MiB": gzipBytes(t, make([]byte, maxOTLPHTTPBodyBytes+1)),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			s := directServer()
			var rec *httptest.ResponseRecorder
			alloc := totalAllocDuring(func() { rec = postBody(s, "application/x-protobuf", true, nil, body) })
			if rec.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status %d, want 413 (%s)", rec.Code, rec.Body.String())
			}
			if limit := uint64(len(body)) + 2<<20; alloc > limit {
				t.Errorf("allocated %d bytes for a %d byte bomb; want no inflate buffer (under %d)", alloc, len(body), limit)
			}
			if p := s.inflate.peak.Load(); p > 1<<20 {
				t.Errorf("leased %d bytes for a bomb", p)
			}
		})
	}
}

// Floods of rejected requests do not write one log line each: the lines are
// sampled to at most about one per second per reason.
func TestOTLPHTTPRejectionLogsAreSampled(t *testing.T) {
	var logs bytes.Buffer
	var mu sync.Mutex
	s := directServer()
	s.log = slog.New(slog.NewTextHandler(writerFunc(func(p []byte) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return logs.Write(p)
	}), nil))
	others := inflateLease{b: s.inflate} // in-flight requests holding the whole budget
	if !others.grow(s.inflate.size) {
		t.Fatal("could not occupy the budget")
	}
	defer others.release()
	body := gzipBytes(t, protoBody(t, makeTraces("busy", "31313131-3131-4131-8131-313131313131")))

	start := time.Now()
	for i := 0; i < 200; i++ {
		if rec := postBody(s, "application/x-protobuf", true, nil, body); rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status %d, want 503", rec.Code)
		}
	}
	elapsed := time.Since(start)
	mu.Lock()
	defer mu.Unlock()
	if n := strings.Count(logs.String(), "request rejected"); n == 0 || n > 1+int(elapsed/time.Second) {
		t.Errorf("%d log lines for 200 rejections in %v, want about one per second:\n%s", n, elapsed, logs.String())
	}
}
