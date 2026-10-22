package server

import (
	"bytes"
	"context"
	"errors"
	"runtime"
	"strconv"
	"strings"
	"testing"

	chexp "github.com/future-agi/future-agi/fi-collector/exporter/clickhouse25exporter"
	"github.com/future-agi/future-agi/fi-collector/pkg/auth"
	"go.opentelemetry.io/collector/pdata/ptrace"
	"go.opentelemetry.io/collector/pdata/ptrace/ptraceotlp"
	"google.golang.org/protobuf/encoding/protowire"
)

// Tests for the decode-cost estimate (decode_cost.go): what it counts in a
// body before the body is decoded, and that the estimate covers the heap the
// handler really holds for that body once it is decoded and converted.

// pbField is one length-delimited field: tag, length, payload.
func pbField(field protowire.Number, payload []byte) []byte {
	b := protowire.AppendTag(nil, field, protowire.BytesType)
	return protowire.AppendBytes(b, payload)
}

func pbVarint(field protowire.Number, v uint64) []byte {
	b := protowire.AppendTag(nil, field, protowire.VarintType)
	return protowire.AppendVarint(b, v)
}

func pbCat(parts ...[]byte) []byte { return bytes.Join(parts, nil) }

func pbRepeat(n int, elem []byte) []byte { return bytes.Repeat(elem, n) }

// pbOneSpan wraps the fields of one span (with valid ids) in a request.
func pbOneSpan(spanFields []byte) []byte {
	ids := pbCat(pbField(1, bytes.Repeat([]byte{0xaa}, 16)), pbField(2, bytes.Repeat([]byte{0xbb}, 8)))
	return pbField(1, pbField(2, pbField(2, pbCat(ids, spanFields))))
}

// pbDeepArray is an AnyValue holding an array holding an AnyValue ... d levels
// deep, built inside-out in O(d).
func pbDeepArray(d int) []byte {
	anySz := make([]int, d+1)
	arrSz := make([]int, d+1)
	for i := 1; i <= d; i++ {
		arrSz[i] = 1 + protowire.SizeVarint(uint64(anySz[i-1])) + anySz[i-1]
		anySz[i] = 1 + protowire.SizeVarint(uint64(arrSz[i])) + arrSz[i]
	}
	out := make([]byte, 0, anySz[d])
	for i := d; i >= 1; i-- {
		out = append(out, 0x2a) // AnyValue.array_value
		out = protowire.AppendVarint(out, uint64(arrSz[i]))
		out = append(out, 0x0a) // ArrayValue.values
		out = protowire.AppendVarint(out, uint64(anySz[i-1]))
	}
	return out
}

func pbDeepAttr(d int) []byte {
	return pbOneSpan(pbField(9, pbCat(pbField(1, []byte("k")), pbField(2, pbDeepArray(d)))))
}

func jsonList(n int, elem string) string {
	return strings.TrimSuffix(strings.Repeat(elem+",", n), ",")
}

func jsonOneSpan(spanFields string) string {
	return `{"resourceSpans":[{"scopeSpans":[{"spans":[{"traceId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","spanId":"bbbbbbbbbbbbbbbb"` +
		spanFields + `}]}]}]}`
}

func spansWithAttrs(n, attrs int) ptrace.Traces {
	tr := ptrace.NewTraces()
	ss := tr.ResourceSpans().AppendEmpty().ScopeSpans().AppendEmpty().Spans()
	for i := 0; i < n; i++ {
		sp := ss.AppendEmpty()
		sp.SetName("s")
		sp.SetTraceID([16]byte{1, byte(i), byte(i >> 8), byte(i >> 16)})
		sp.SetSpanID([8]byte{2, byte(i), byte(i >> 8), byte(i >> 16)})
		sp.SetStartTimestamp(1)
		sp.SetEndTimestamp(2)
		for k := 0; k < attrs; k++ {
			sp.Attributes().PutInt("k"+strconv.Itoa(k), int64(k))
		}
	}
	return tr
}

func TestCountProtoElements(t *testing.T) {
	kv := pbField(9, pbCat(pbField(1, []byte("k")), pbField(2, pbVarint(3, 1))))
	cases := []struct {
		name string
		body []byte
		want decodeCounts
	}{
		{"empty request", nil, decodeCounts{}},
		{"empty ResourceSpans", pbRepeat(1000, pbField(1, nil)), decodeCounts{resourceSpans: 1000}},
		{"empty ScopeSpans and deprecated ScopeSpans", pbField(1, pbCat(pbRepeat(3, pbField(2, nil)), pbRepeat(4, pbField(1000, nil)))),
			decodeCounts{resourceSpans: 1, scopeSpans: 7}},
		{"empty spans", pbField(1, pbField(2, pbRepeat(500, pbField(2, nil)))), decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 500}},
		{"span attributes, events, links", pbOneSpan(pbCat(pbRepeat(8, kv), pbRepeat(3, pbField(11, pbField(3, nil))), pbRepeat(2, pbField(13, pbField(4, nil))))),
			decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 1, keyValues: 8 + 3 + 2, events: 3, links: 2}},
		{"resource and scope attributes", pbField(1, pbCat(pbField(1, pbRepeat(5, pbField(1, nil))), pbField(2, pbField(1, pbRepeat(6, pbField(3, nil)))))),
			decodeCounts{resourceSpans: 1, scopeSpans: 1, keyValues: 11}},
		{"array and kvlist values", pbOneSpan(pbCat(
			pbField(9, pbField(2, pbField(5, pbRepeat(7, pbField(1, nil))))),
			pbField(9, pbField(2, pbField(6, pbRepeat(4, pbField(1, nil))))))),
			decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 1, keyValues: 2 + 4, anyValues: 7, containers: 2}},
		{"strings and unknown fields are not counted", pbCat(pbField(1, pbCat(pbField(3, []byte("\x0a\x00\x0a\x00")), pbField(99, pbRepeat(10, pbField(2, nil))))), pbVarint(7, 1)),
			decodeCounts{resourceSpans: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := countProtoElements(tc.body)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if got != tc.want {
				t.Errorf("counts = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestCountProtoElementsRejectsMalformedAndTooDeep(t *testing.T) {
	if _, err := countProtoElements([]byte{0x0a, 0x05, 0x00}); err == nil {
		t.Error("a truncated length-delimited field must be an error")
	}
	if _, err := countProtoElements(make([]byte, 64)); err == nil {
		t.Error("field number 0 must be an error")
	}
	if _, err := countProtoElements(pbDeepAttr(maxDecodeNesting / 4)); err != nil {
		t.Errorf("nesting well under the limit: err = %v", err)
	}
	if _, err := countProtoElements(pbDeepAttr(maxDecodeNesting)); !errors.Is(err, errNestingTooDeep) {
		t.Errorf("nesting past the limit: err = %v, want errNestingTooDeep", err)
	}
}

func TestCountJSONElements(t *testing.T) {
	cases := []struct {
		name string
		body string
		want decodeCounts
	}{
		{"empty object", `{}`, decodeCounts{}},
		{"empty ResourceSpans", `{"resourceSpans":[` + jsonList(1000, `{}`) + `]}`, decodeCounts{resourceSpans: 1000}},
		{"snake_case keys", `{"resource_spans":[{"scope_spans":[{},{}]}]}`, decodeCounts{resourceSpans: 1, scopeSpans: 2}},
		// pdata appends an element before reading it, so null counts too.
		{"null spans", `{"resourceSpans":[{"scopeSpans":[{"spans":[` + jsonList(300, `null`) + `]}]}]}`,
			decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 300}},
		// jsoniter unescapes keys before pdata matches them.
		{"escaped keys", `{"resource\u005fspans":[{"scope\/Spans":[],"scopeSpans":[{"sp\u0061ns":[{},{},{}]}]}]}`,
			decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 3}},
		{"attributes, events, links", jsonOneSpan(`,"attributes":[{"key":"a","value":{"intValue":"1"}},{}],"events":[{"attributes":[{}]}],"links":[{},{"attributes":[{},{}]}]`),
			decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 1, keyValues: 2 + 1 + 2, events: 1, links: 2}},
		{"array and kvlist values", jsonOneSpan(`,"attributes":[{"key":"a","value":{"arrayValue":{"values":[{},{},{}]}}},{"key":"b","value":{"kvlist_value":{"values":[{},{}]}}}]`),
			decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 1, keyValues: 2 + 5, containers: 2}},
		{"brackets inside strings are not counted", jsonOneSpan(`,"name":"[{\"spans\":[{},{}]}]","attributes":[{"key":"spans","value":{"stringValue":"{[\\\"x\\\"]}"}}]`),
			decodeCounts{resourceSpans: 1, scopeSpans: 1, spans: 1, keyValues: 1}},
		{"unknown keys are not counted", `{"resourceSpans":[{"foo":[{},{},{}],"scopeSpans":[]}]}`, decodeCounts{resourceSpans: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := countJSONElements([]byte(tc.body))
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if got != tc.want {
				t.Errorf("counts = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestCountJSONElementsRejectsTooDeep(t *testing.T) {
	deep := strings.Repeat("[", maxDecodeNesting+1) + strings.Repeat("]", maxDecodeNesting+1)
	if _, err := countJSONElements([]byte(`{"x":` + deep + `}`)); !errors.Is(err, errNestingTooDeep) {
		t.Errorf("err = %v, want errNestingTooDeep", err)
	}
}

// handlerPipeline is what handleHTTPTraces does with a decoded body when auth
// is on and the project comes from the header: decode, stamp the header's
// project, stamp the two ids StampResourceAttrs adds, convert to rows. It
// returns everything the handler keeps alive until it returns.
func handlerPipeline(t *testing.T, body []byte, jsonBody bool) []any {
	t.Helper()
	req := ptraceotlp.NewExportRequest()
	var err error
	if jsonBody {
		err = req.UnmarshalJSON(body)
	} else {
		err = req.UnmarshalProto(body)
	}
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, err := auth.ApplyProjectNameHeader(req.Traces(), []string{"project-from-header"}); err != nil {
		t.Fatalf("header: %v", err)
	}
	rss := req.Traces().ResourceSpans()
	for i := 0; i < rss.Len(); i++ {
		attrs := rss.At(i).Resource().Attributes()
		attrs.PutStr("fi.project_id", "11111111-1111-4111-8111-111111111111")
		attrs.PutStr("fi.org_id", "22222222-2222-4222-8222-222222222222")
	}
	rows, ids, err := chexp.ConvertWithIdentities(context.Background(), req.Traces(), nil)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	return []any{req, rows, ids}
}

func liveHeapAfterGC() uint64 {
	runtime.GC()
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

// The estimate must cover the heap the handler really holds for a body, for
// every shape an attacker could choose: most elements per wire byte (empty
// messages), the deepest nesting allowed, and the biggest strings and bytes.
// If a converter change makes rows bigger, this fails and the per-element
// costs in decode_cost.go need raising.
func TestDecodeCostCoversHeldHeap(t *testing.T) {
	spans8, err := ptraceotlp.NewExportRequestFromTraces(spansWithAttrs(5000, 8)).MarshalProto()
	if err != nil {
		t.Fatal(err)
	}
	strKV := func(i int) []byte {
		return pbField(9, pbCat(pbField(1, []byte("key."+strconv.Itoa(i))), pbField(2, pbField(1, []byte("v")))))
	}
	var manyStrKVs [][]byte
	for i := 0; i < 50000; i++ {
		manyStrKVs = append(manyStrKVs, strKV(i))
	}
	big := bytes.Repeat([]byte("x"), 4<<20)
	cases := []struct {
		name     string
		body     []byte
		jsonBody bool
	}{
		{"empty ResourceSpans", pbRepeat(50000, pbField(1, nil)), false},
		{"empty ScopeSpans", pbField(1, pbRepeat(50000, pbField(2, nil))), false},
		{"empty spans", pbField(1, pbField(2, pbRepeat(10000, pbField(2, nil)))), false},
		{"spans with 8 int attributes", spans8, false},
		{"empty span attributes", pbOneSpan(pbRepeat(100000, pbField(9, nil))), false},
		{"distinct string attributes", pbOneSpan(pbCat(manyStrKVs...)), false},
		{"empty events", pbOneSpan(pbRepeat(50000, pbField(11, nil))), false},
		{"empty links", pbOneSpan(pbRepeat(50000, pbField(13, nil))), false},
		{"kvlist values", pbOneSpan(pbField(9, pbField(2, pbField(6, pbRepeat(100000, pbField(1, nil)))))), false},
		{"array values", pbOneSpan(pbField(9, pbField(2, pbField(5, pbRepeat(100000, pbField(1, nil)))))), false},
		{"deepest nesting allowed", pbDeepAttr(maxDecodeNesting/2 - 8), false},
		{"one big string", pbOneSpan(pbField(9, pbCat(pbField(1, []byte("k")), pbField(2, pbField(1, big))))), false},
		{"one big bytes value", pbOneSpan(pbField(9, pbCat(pbField(1, []byte("k")), pbField(2, pbField(7, big))))), false},
		{"JSON empty ResourceSpans", []byte(`{"resourceSpans":[` + jsonList(50000, `{}`) + `]}`), true},
		{"JSON null spans", []byte(`{"resourceSpans":[{"scopeSpans":[{"spans":[` + jsonList(10000, `null`) + `]}]}]}`), true},
		{"JSON empty attributes", []byte(jsonOneSpan(`,"attributes":[` + jsonList(100000, `{}`) + `]`)), true},
		{"JSON big string", []byte(jsonOneSpan(`,"name":"` + string(big) + `"`)), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			est, err := estimateDecodeCost(tc.body, tc.jsonBody)
			if err != nil {
				t.Fatalf("estimate: %v", err)
			}
			before := liveHeapAfterGC()
			held := handlerPipeline(t, tc.body, tc.jsonBody)
			after := liveHeapAfterGC()
			runtime.KeepAlive(held)
			runtime.KeepAlive(tc.body)
			var used int64
			if after > before {
				used = int64(after - before)
			}
			if est < used {
				t.Errorf("estimate %d bytes < %d bytes held for a %d byte body", est, used, len(tc.body))
			}
			t.Logf("estimate %d, held %d (%.2fx)", est, used, float64(est)/float64(max(used, 1)))
		})
	}
}
