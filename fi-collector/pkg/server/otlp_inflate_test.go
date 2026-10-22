package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// Tests for how a gzip body is inflated (otlp_gzip.go): into one buffer of
// its exact decompressed size, leased before it is allocated, and with no
// buffer at all for a body that inflates past the 16 MiB cap.

// totalAllocDuring reports the bytes allocated (live or not) while fn ran.
func totalAllocDuring(fn func()) uint64 {
	var m0, m1 runtime.MemStats
	runtime.ReadMemStats(&m0)
	fn()
	runtime.ReadMemStats(&m1)
	return m1.TotalAlloc - m0.TotalAlloc
}

// acceptingServer enqueues accepted rows and never flushes them.
func acceptingServer() *Server {
	return New(Config{BatchMaxRows: 1 << 30, BatchMaxAge: time.Hour}, nil, nil, nil, nil)
}

// A gzip body is inflated into one buffer of its exact size, leased before it
// is allocated: in one pass when its trailer is plausible, and after measuring
// it when the trailer declares a ratio no OTLP batch has.
func TestInflateGzipAllocatesTheExactSizeOnce(t *testing.T) {
	words := make([]string, 0, 300000)
	for i := 0; len(words) < cap(words); i++ {
		words = append(words, strconv.FormatUint(uint64(i)*2654435761%1000003, 36))
	}
	cases := map[string][]byte{
		"plausible ratio":        []byte(strings.Join(words, " ")),
		"ratio past the trusted": bytes.Repeat([]byte("otlp "), 3<<20/5),
		"empty body":             nil,
		"tiny body":              []byte("tiny"),
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			wire := gzipBytes(t, raw)
			b := newInflateBudget(64 << 20)
			var out []byte
			var release func()
			var err error
			alloc := totalAllocDuring(func() { out, release, err = inflateGzip(context.Background(), wire, maxOTLPHTTPBodyBytes, b) })
			if err != nil || !bytes.Equal(out, raw) {
				t.Fatalf("inflate: err=%v, %d bytes out, want %d", err, len(out), len(raw))
			}
			if held := b.inUse.Load(); held != int64(len(raw)) {
				t.Errorf("leased %d bytes for a %d byte body, want exactly its size", held, len(raw))
			}
			if alloc > uint64(len(raw))+1<<20 {
				t.Errorf("allocated %d bytes to inflate a %d byte body (ratio %d)", alloc, len(raw), len(raw)/max(len(wire), 1))
			}
			release()
			release() // a second call is a no-op
			if u := b.inUse.Load(); u != 0 {
				t.Errorf("%d bytes still leased after release", u)
			}
		})
	}
}

// A trailer can understate the size (a multi-member body declares only its
// last member). Such a body is measured and inflated again at its real size,
// and only that size is leased once it returns.
func TestInflateGzipBodyLongerThanItsTrailer(t *testing.T) {
	raw := bytes.Repeat([]byte("span "), 1<<20/5)
	wire := append(gzipBytes(t, raw), gzipBytes(t, []byte("x"))...) // last member: 1 byte
	if gzipSizeHint(wire) != 1 {
		t.Fatalf("hint %d, want the last member's size", gzipSizeHint(wire))
	}
	b := newInflateBudget(64 << 20)
	out, release, err := inflateGzip(context.Background(), wire, maxOTLPHTTPBodyBytes, b)
	if err != nil || !bytes.Equal(out, append(raw, 'x')) {
		t.Fatalf("inflate: err=%v, %d bytes out, want %d", err, len(out), len(raw)+1)
	}
	defer release()
	if held := b.inUse.Load(); held != int64(len(out)) {
		t.Errorf("leased %d bytes for a %d byte body", held, len(out))
	}

	big := make([]byte, 24)
	binary.LittleEndian.PutUint32(big[20:], 1<<31)
	if h := gzipSizeHint(big); h != 1<<31 {
		t.Errorf("hint %d, want the trailer's value", h)
	}
}

// A decompression bomb is refused without allocating a buffer for it: one
// whose trailer is honest is measured because the trailer is over the cap,
// one that hides its size behind a small last member is measured once the
// small buffer overflows.
func TestOTLPHTTPGzipBombAllocatesNoInflateBuffer(t *testing.T) {
	zeros := gzipBytes(t, make([]byte, 64<<20))
	// A member of 4 GiB + 1 byte declares 1 in its 32-bit trailer. Only its
	// first 16 MiB + 1 byte are ever inflated, so a 17 MiB member with that
	// trailer stands in for it (the CRC would only be checked at the end).
	wrapped := gzipBytes(t, make([]byte, 17<<20))
	binary.LittleEndian.PutUint32(wrapped[len(wrapped)-4:], 1)
	cases := map[string][]byte{
		"trailer declares 64 MiB":          zeros,
		"trailer declares 1 byte":          append(append([]byte(nil), zeros...), gzipBytes(t, []byte{0})...),
		"256 members of 1 MiB of zeros":    zeroGzipMembers(t, 256),
		"trailer declares 16 MiB + 1 byte": gzipBytes(t, make([]byte, maxOTLPHTTPBodyBytes+1)),
		"trailer wrapped at 4 GiB":         wrapped,
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
			if u := s.inflate.inUse.Load(); u != 0 {
				t.Errorf("%d budget bytes still leased after the request returned", u)
			}
		})
	}
}

// Floods of rejected requests do not write one log line each: the lines are
// sampled to at most about one per second per reason (over the cap, and
// budget busy).
func TestOTLPHTTPRejectionLogsAreSampled(t *testing.T) {
	var logs bytes.Buffer
	var mu sync.Mutex
	s := directServer()
	s.log = slog.New(slog.NewTextHandler(writerFunc(func(p []byte) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return logs.Write(p)
	}), nil))
	s.inflate.wait = time.Millisecond
	hold, err := s.inflate.acquire(context.Background(), s.inflate.size) // in-flight requests holding the whole budget
	if err != nil {
		t.Fatal(err)
	}
	defer hold()
	busy := gzipBytes(t, protoBody(t, makeTraces("busy", "31313131-3131-4131-8131-313131313131")))
	overCap := make([]byte, maxOTLPHTTPBodyBytes+1)

	start := time.Now()
	for i := 0; i < 100; i++ {
		if rec := postBody(s, "application/x-protobuf", true, nil, busy); rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status %d, want 503", rec.Code)
		}
	}
	for i := 0; i < 20; i++ {
		if rec := postBody(s, "application/x-protobuf", false, nil, overCap); rec.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("status %d, want 413", rec.Code)
		}
	}
	elapsed := time.Since(start)
	mu.Lock()
	defer mu.Unlock()
	perReason := 1 + int(elapsed/time.Second)
	for _, line := range []string{"inflate budget busy", "over size cap"} {
		if n := strings.Count(logs.String(), line); n == 0 || n > perReason {
			t.Errorf("%d %q lines for a burst of rejections in %v, want about one per second:\n%s", n, line, elapsed, logs.String())
		}
	}
}
