package main

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"google.golang.org/grpc/encoding"
)

// The collector must not register a gzip decompressor for gRPC: grpc-go
// inflates and decodes a message before any interceptor, auth included, runs,
// so a registered gzip codec would let an unauthenticated caller make the
// collector inflate up to the receive cap for a few KiB of upload. Importing
// google.golang.org/grpc/encoding/gzip anywhere in the binary (directly or
// through a dependency) registers it globally; this test binary links the
// same packages as cmd/fi-collector, so such an import fails here.
func TestGRPCGzipDecompressorIsNotRegistered(t *testing.T) {
	if c := encoding.GetCompressor("gzip"); c != nil {
		t.Fatalf("a gzip gRPC compressor is registered (%T); see the comment by grpc.NewServer in pkg/server", c)
	}
}

func TestApplyEnvOverridesGzipInflightMiB(t *testing.T) {
	cases := []struct {
		env     string
		want    int
		warning bool
	}{
		{"", 0, false},
		{"64", 64, false},
		{"0", 0, true},
		{"-8", 0, true},
		{"lots", 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.env, func(t *testing.T) {
			clearCollectorEnv(t)
			t.Setenv("FI_GZIP_INFLIGHT_MIB", tc.env)
			var logs bytes.Buffer
			cfg := rootConfig{}
			if err := applyEnvOverrides(slog.New(slog.NewTextHandler(&logs, nil)), &cfg); err != nil {
				t.Fatal(err)
			}
			if cfg.Server.GzipInflightMiB != tc.want {
				t.Errorf("GzipInflightMiB = %d, want %d", cfg.Server.GzipInflightMiB, tc.want)
			}
			if got := strings.Contains(logs.String(), "FI_GZIP_INFLIGHT_MIB"); got != tc.warning {
				t.Errorf("warning logged = %v, want %v (%q)", got, tc.warning, logs.String())
			}
		})
	}
}
