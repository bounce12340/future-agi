package server

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The compressed bytes on the wire are capped too. Empty deflate stored
// blocks inflate to nothing, so a stream of them is never caught by the
// decompressed-size cap; only the wire cap can stop it. A small limit keeps
// the stream short; the handler passes maxOTLPHTTPBodyBytes.
func TestReadOTLPHTTPBodyCapsCompressedWireBytes(t *testing.T) {
	const limit = 4 << 10
	var body bytes.Buffer
	body.Write([]byte{0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff}) // gzip header, deflate
	emptyStoredBlock := []byte{0x00, 0x00, 0x00, 0xff, 0xff}  // BFINAL=0 BTYPE=00 LEN=0 NLEN=0xffff
	for body.Len() <= 2*limit {
		body.Write(emptyStoredBlock)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/traces", bytes.NewReader(body.Bytes()))
	_, err := readOTLPHTTPBody(httptest.NewRecorder(), req, true, limit)
	var tooLarge *http.MaxBytesError
	if !errors.As(err, &tooLarge) {
		t.Fatalf("err = %v, want *http.MaxBytesError from the wire cap", err)
	}
}
