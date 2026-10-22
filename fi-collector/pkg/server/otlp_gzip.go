package server

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// otlpContentCoding reports whether the body is gzip-compressed, and false
// for ok when it carries a coding we do not decode. Tokens are
// case-insensitive (RFC 9110 section 8.4.1); `identity` is a no-op and
// `x-gzip` is gzip. Exactly one gzip layer is supported, which is all OTLP
// exporters send.
func otlpContentCoding(h http.Header) (gzipped, ok bool) {
	layers := 0
	for _, v := range h.Values("Content-Encoding") {
		for _, tok := range strings.Split(v, ",") {
			switch strings.ToLower(strings.TrimSpace(tok)) {
			case "", "identity":
			case "gzip", "x-gzip":
				layers++
			default:
				return false, false
			}
		}
	}
	if layers > 1 {
		return false, false
	}
	return layers == 1, true
}

// errInflatedBodyTooLarge reports a body whose decompressed size passes the
// 16 MiB OTLP/HTTP cap.
var errInflatedBodyTooLarge = errors.New("decompressed body over size cap")

// readOTLPHTTPBody reads at most limit bytes from an uncompressed body, or
// streams a gzip body through a decompressed-size limit. Gzip requests reserve
// limit bytes before inflation so the buffer's maximum output is accounted for
// while it grows; once the exact decompressed size is known, the reservation is
// shrunk to that size. The caller holds the returned lease until the handler
// returns. The gzip wire stream is not capped or retained in memory; the OTLP
// body limit applies after decompression, like the upstream Collector.
func readOTLPHTTPBody(w http.ResponseWriter, r *http.Request, gzipped bool, limit int64, budget *inflateBudget) (body []byte, release func(), err error) {
	if !gzipped {
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
		return body, noRelease, err
	}
	return inflateGzip(r.Context(), r.Body, limit, budget)
}

func noRelease() {}

// inflateGzip streams the gzip body through a limit of limit+1 decompressed
// bytes. It reserves the maximum body size before inflating, then shrinks the
// lease to the actual decompressed length. The lease is released on errors and
// returned to the caller on success.
//
// The request body is not buffered on the wire. A slow upload does not hold an
// inflated buffer; once the gzip reader consumes bytes, it writes no more than
// limit+1 decompressed bytes. A body with more than limit decompressed bytes
// returns errInflatedBodyTooLarge (413). Any gzip header, trailer, checksum,
// or stream error before that limit is reached is malformed (400).
func inflateGzip(ctx context.Context, wire io.Reader, limit int64, budget *inflateBudget) ([]byte, func(), error) {
	lease, err := budget.reserve(ctx, limit)
	if err != nil {
		return nil, noRelease, errInflateBudgetBusy
	}
	zr, err := gzip.NewReader(wire)
	if err != nil {
		lease.release()
		return nil, noRelease, gzipBodyError(err)
	}
	body, err := io.ReadAll(io.LimitReader(zr, limit+1))
	closeErr := zr.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		lease.release()
		return nil, noRelease, gzipBodyError(err)
	}
	if int64(len(body)) > limit {
		lease.release()
		return nil, noRelease, errInflatedBodyTooLarge
	}
	lease.shrink(int64(len(body)))
	return body, lease.release, nil
}

// gzipBodyError labels a corrupt or truncated gzip stream (a 400).
func gzipBodyError(err error) error {
	if errors.Is(err, io.EOF) {
		err = io.ErrUnexpectedEOF // an empty body is a truncated gzip stream
	}
	return fmt.Errorf("decode gzip: %w", err)
}
