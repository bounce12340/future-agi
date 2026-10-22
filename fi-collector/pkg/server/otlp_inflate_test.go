package server

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// Tests for streaming gzip inflation (otlp_gzip.go). The 16 MiB reservation is
// taken before inflation, then shrunk to the exact decompressed body size; the
// reader applies the cap after decompression and does not cap or buffer the
// wire body.

// acceptingServer enqueues accepted rows and never flushes them.
func acceptingServer() *Server {
	return New(Config{BatchMaxRows: 1 << 30, BatchMaxAge: time.Hour}, nil, nil, nil, nil)
}

// A request reserves the 16 MiB maximum before streaming, then holds exactly
// its decompressed body size until the handler returns.
func TestInflateGzipReservesThenShrinksToDecompressedSize(t *testing.T) {
	cases := map[string][]byte{
		"random body":         bytes.Repeat([]byte("0123456789abcdef"), 1<<14),
		"highly compressible": bytes.Repeat([]byte("otlp "), 3<<20/5),
		"empty body":          nil,
		"tiny body":           []byte("tiny"),
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			wire := gzipBytes(t, raw)
			b := newInflateBudget(maxOTLPHTTPBodyBytes)
			var out []byte
			var release func()
			var err error
			out, release, err = inflateGzip(context.Background(), bytes.NewReader(wire), maxOTLPHTTPBodyBytes, b)
			if err != nil || !bytes.Equal(out, raw) {
				t.Fatalf("inflate: err=%v, %d bytes out, want %d", err, len(out), len(raw))
			}
			if p := b.peak.Load(); p != maxOTLPHTTPBodyBytes {
				t.Errorf("reserved %d bytes before inflate, want the maximum %d", p, maxOTLPHTTPBodyBytes)
			}
			if held := b.inUse.Load(); held != int64(len(raw)) {
				t.Errorf("leased %d bytes after inflate, want exact decompressed size %d", held, len(raw))
			}
			release()
			release() // a second call is a no-op
			if u := b.inUse.Load(); u != 0 {
				t.Errorf("%d bytes still leased after release", u)
			}
		})
	}
}

// A valid gzip body may contain multiple members. Stream all members and hold
// the full decompressed size, regardless of the last member's trailer size.
func TestInflateGzipReadsAllMembers(t *testing.T) {
	raw := bytes.Repeat([]byte("span "), 1<<20/5)
	wire := append(gzipBytes(t, raw), gzipBytes(t, []byte("x"))...)
	b := newInflateBudget(maxOTLPHTTPBodyBytes)
	out, release, err := inflateGzip(context.Background(), bytes.NewReader(wire), maxOTLPHTTPBodyBytes, b)
	if err != nil || !bytes.Equal(out, append(raw, 'x')) {
		t.Fatalf("inflate: err=%v, %d bytes out, want %d", err, len(out), len(raw)+1)
	}
	defer release()
	if held := b.inUse.Load(); held != int64(len(out)) {
		t.Errorf("leased %d bytes for a %d byte body", held, len(out))
	}
}

// A gzip bomb is stopped as soon as it expands past 16 MiB. This checks the
// decompressed stream limit, not the gzip trailer or compressed wire size.
func TestOTLPHTTPGzipBombIs413WithinDecompressedCap(t *testing.T) {
	zeros := gzipBytes(t, make([]byte, 64<<20))
	cases := map[string][]byte{
		"64 MiB member":                         zeros,
		"64 MiB followed by 1-byte last member": append(append([]byte(nil), zeros...), gzipBytes(t, []byte{0})...),
		"256 members of 1 MiB of zeros":         zeroGzipMembers(t, 256),
		"16 MiB + 1 byte":                       gzipBytes(t, make([]byte, maxOTLPHTTPBodyBytes+1)),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			s := directServer()
			rec := postBody(s, "application/x-protobuf", true, nil, body)
			if rec.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status %d, want 413 (%s)", rec.Code, rec.Body.String())
			}
			if p := s.inflate.peak.Load(); p != maxOTLPHTTPBodyBytes {
				t.Errorf("peak reservation %d, want the 16 MiB maximum", p)
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
