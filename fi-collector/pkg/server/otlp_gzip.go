package server

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
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

// errInflatedBodyTooLarge reports a gzip body whose decompressed size passes
// maxOTLPHTTPBodyBytes.
var errInflatedBodyTooLarge = errors.New("decompressed body over size cap")

// maxTrustedGzipRatio is the most a gzip body may declare it inflates (per
// its trailer, against its wire size) for its trailer to be used as the size
// of its buffer. OTLP batches compress 3-20x, highly repetitive ones a few
// hundred times; a body declaring more is measured first (see inflateGzip).
const maxTrustedGzipRatio = 64

func noRelease() {}

// readOTLPHTTPBody reads at most limit bytes from the wire and, for a gzip
// body, at most limit bytes after decompression. A gzip body's decompressed
// size is leased from budget before its buffer is allocated; the caller
// defers release, which is never nil.
//
// A gzip body is read off the wire in full before any of it is inflated, so a
// slow or stalled upload only holds the compressed bytes it has sent (one
// byte held per byte uploaded, as for an uncompressed body). The wire limit
// also bounds gzip input that inflates to little or nothing (e.g. empty
// deflate blocks).
//
// Errors: *http.MaxBytesError (wire over the cap), errInflatedBodyTooLarge
// (decompressed over the cap), errInflateBudgetBusy (no lease in time);
// anything else is a malformed body.
func readOTLPHTTPBody(w http.ResponseWriter, r *http.Request, gzipped bool, limit int64, budget *inflateBudget) (body []byte, release func(), err error) {
	wire, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return nil, noRelease, err
		}
		return nil, noRelease, fmt.Errorf("read body: %w", err)
	}
	if !gzipped {
		return wire, noRelease, nil
	}
	return inflateGzip(r.Context(), wire, limit, budget)
}

// inflateGzip decompresses wire into one buffer of exactly its decompressed
// size, at most limit bytes, leasing that size from budget before allocating
// it. It holds at most one lease at a time, so waiting for budget never holds
// any.
//
// The size comes from the gzip trailer (ISIZE, gzipSizeHint) when that is
// plausible: at most limit and at most maxTrustedGzipRatio times the wire
// size. The trailer is only a hint, since a multi-member body declares just
// its last member and the field wraps at 4 GiB. So a body whose trailer is
// implausible, or that turns out longer than its trailer said, is first
// inflated without keeping the bytes, to learn its real size. A bomb is
// refused there having cost CPU but no buffer.
func inflateGzip(ctx context.Context, wire []byte, limit int64, budget *inflateBudget) ([]byte, func(), error) {
	size := gzipSizeHint(wire)
	measure := size > limit || size > maxTrustedGzipRatio*int64(len(wire))
	for {
		if measure {
			n, err := inflatedSize(wire, limit)
			if err != nil {
				return nil, noRelease, err
			}
			size = n
		}
		release, err := budget.acquire(ctx, size)
		if err != nil {
			return nil, noRelease, err
		}
		buf, complete, err := inflateInto(wire, size)
		if err == nil && complete {
			return buf, release, nil
		}
		release()
		switch {
		case err != nil:
			return nil, noRelease, err
		case measure: // measured, yet longer: the same bytes inflate the same way
			return nil, noRelease, gzipBodyError(io.ErrUnexpectedEOF)
		}
		measure = true // longer than its trailer said
	}
}

// inflateInto inflates wire into a new buffer of size bytes. complete reports
// that the body fitted, and is false when there is more of it than fits.
func inflateInto(wire []byte, size int64) (out []byte, complete bool, err error) {
	zr, err := gzip.NewReader(bytes.NewReader(wire)) // a bad header fails before the allocation
	if err != nil {
		return nil, false, gzipBodyError(err)
	}
	defer zr.Close()
	buf := make([]byte, size)
	// A valid body is never shorter than its trailer says: gzip.Reader checks
	// each member's size and CRC at its end, and a body is at least as long as
	// its last member. So running out early is a corrupt body.
	if _, err := io.ReadFull(zr, buf); err != nil {
		return nil, false, gzipBodyError(err)
	}
	// Full: the body fits only if nothing is left.
	var probe [1]byte
	n, err := io.ReadFull(zr, probe[:])
	switch {
	case n > 0:
		return nil, false, nil
	case errors.Is(err, io.EOF):
		return buf, true, nil
	default:
		return nil, false, gzipBodyError(err)
	}
}

// inflatedSize inflates wire without keeping the bytes and returns its
// decompressed size, or errInflatedBodyTooLarge once it passes limit.
func inflatedSize(wire []byte, limit int64) (int64, error) {
	zr, err := gzip.NewReader(bytes.NewReader(wire))
	if err != nil {
		return 0, gzipBodyError(err)
	}
	defer zr.Close()
	n, err := io.Copy(io.Discard, io.LimitReader(zr, limit+1))
	switch {
	case err != nil:
		return 0, gzipBodyError(err)
	case n > limit:
		return 0, errInflatedBodyTooLarge
	}
	return n, nil
}

// gzipBodyError labels a corrupt or truncated gzip stream (a 400).
func gzipBodyError(err error) error {
	if errors.Is(err, io.EOF) {
		err = io.ErrUnexpectedEOF // an empty body is a truncated gzip stream
	}
	return fmt.Errorf("decode gzip: %w", err)
}

// gzipSizeHint is the decompressed size a gzip body declares in its trailer
// (ISIZE, RFC 1952 section 2.3.1): the last member's size modulo 2^32.
func gzipSizeHint(wire []byte) int64 {
	if len(wire) < 18 { // a 10-byte header and an 8-byte trailer at least
		return 0
	}
	return int64(binary.LittleEndian.Uint32(wire[len(wire)-4:]))
}
