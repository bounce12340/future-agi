package server

import (
	"bytes"
	"compress/gzip"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Upstream confighttp applies max_request_body_size after decompression. A
// gzip body may therefore be over 16 MiB on the wire if it inflates to no more
// than 16 MiB. Keep the wire streaming; the decompressed size is the cap.
func TestReadOTLPHTTPGzipWireMayExceedLimitWhenInflatedSizeFits(t *testing.T) {
	const limit = maxOTLPHTTPBodyBytes
	raw := make([]byte, limit-1024)
	if _, err := rand.New(rand.NewSource(42)).Read(raw); err != nil {
		t.Fatal(err)
	}
	var wire bytes.Buffer
	zw := gzip.NewWriter(&wire)
	if _, err := zw.Write(raw); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if wire.Len() <= limit {
		t.Fatalf("test setup: compressed wire body is %d bytes, want more than %d", wire.Len(), limit)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/traces", bytes.NewReader(wire.Bytes()))
	budget := newInflateBudget(limit)
	body, release, err := readOTLPHTTPBody(httptest.NewRecorder(), req, true, limit, budget)
	defer release()
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if !bytes.Equal(body, raw) {
		t.Fatalf("got %d decompressed bytes, want %d", len(body), len(raw))
	}
	if got := budget.inUse.Load(); got != int64(len(raw)) {
		t.Errorf("leased %d bytes, want exact decompressed size %d", got, len(raw))
	}
}
