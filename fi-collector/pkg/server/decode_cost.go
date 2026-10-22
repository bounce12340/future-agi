package server

import (
	"encoding/binary"
	"fmt"

	"google.golang.org/protobuf/encoding/protowire"
)

// Decode cost. Decoding an OTLP request allocates far more than its size when
// it is made of many small elements: an empty ResourceSpans is 2 bytes of
// protobuf and ~280 bytes of heap, an empty span is 2 bytes and a ~4 KiB row
// once converted. 16 MiB of empty spans would hold ~36 GiB. Uncompressed, the
// sender pays one byte of upload per body byte for that; with gzip, 16 MiB of
// them is ~16 KiB on the wire. So before a body is decoded, its elements are
// counted (one pass, no allocation), and a gzip request leases the estimate
// from the inflate budget along with its decompressed bytes.
//
// The pass also refuses nesting deeper than maxDecodeNesting for every body:
// pdata's protobuf decoder recurses once per level with no limit, and ~1.5M
// levels (a 14 MiB body) overflow the goroutine stack, which kills the
// process.

// Heap held per decoded element, in bytes, on the handler's full path (decode,
// project header, auth stamp, ConvertWithIdentities) with pdata v1.20.0,
// measured and rounded up. TestDecodeCostCoversHeldHeap fails if a body ever
// holds more than its estimate, e.g. after a converter change.
const (
	costResourceSpans = 320      // measured 282 (with the header and auth attributes)
	costScopeSpans    = 160      // 122
	costSpan          = 4608     // 4306 empty, 4450 with every scalar field set: the converted row
	costEvent         = 192      // 138
	costLink          = 128      // 89
	costKeyValue      = 128      // 36 empty, 113 with a distinct key and a string value
	costAnyValue      = 48       // 34 per array element
	costContainer     = 64       // an array or kvlist: 89 per nesting level with its AnyValue
	costPerBodyByte   = 2        // strings are copied once, bytes values twice
	costRequest       = 64 << 10 // per request: request and response objects, fixed overhead
)

// maxDecodeNesting is the deepest message (protobuf) or object/array (JSON)
// nesting accepted. It matches protobuf-go's default recursion limit and
// jsoniter's maxDepth, and is far beyond any real attribute value.
const maxDecodeNesting = 10000

var errNestingTooDeep = fmt.Errorf("nesting deeper than %d levels", maxDecodeNesting)

// decodeCounts is what decoding a body will allocate, element by element.
type decodeCounts struct {
	resourceSpans, scopeSpans, spans, events, links, keyValues, anyValues, containers int64
}

// cost is the estimated heap, in bytes, the handler holds for a body of
// bodyLen bytes with these elements, not counting the body itself.
func (c decodeCounts) cost(bodyLen int) int64 {
	return costRequest +
		c.resourceSpans*costResourceSpans +
		c.scopeSpans*costScopeSpans +
		c.spans*costSpan +
		c.events*costEvent +
		c.links*costLink +
		c.keyValues*costKeyValue +
		c.anyValues*costAnyValue +
		c.containers*costContainer +
		int64(bodyLen)*costPerBodyByte
}

// estimateDecodeCost counts the elements of an ExportTraceServiceRequest
// body and returns their estimated heap cost. An error means the body is
// nested too deep or is not valid protobuf; the decoder would refuse the
// latter too.
func estimateDecodeCost(body []byte, jsonBody bool) (int64, error) {
	var c decodeCounts
	var err error
	if jsonBody {
		c, err = countJSONElements(body)
	} else {
		c, err = countProtoElements(body)
	}
	if err != nil {
		return 0, err
	}
	return c.cost(len(body)), nil
}

// ── protobuf ───────────────────────────────────────────────────────────────

// pbKind is an OTLP trace message type the count descends into.
type pbKind uint8

const (
	pbRequest pbKind = iota // ExportTraceServiceRequest
	pbResourceSpans
	pbResource
	pbScopeSpans
	pbScope
	pbSpan
	pbEvent
	pbLink
	pbKeyValue
	pbAnyValue
	pbArrayValue
	pbKeyValueList
)

// pbChild is the message type of field num of a parent message, following the
// OTLP trace protos as pdata v1.20.0 decodes them. ok is false for scalar,
// string and bytes fields, for messages that hold nothing repeated (Status),
// and for fields pdata does not know (it skips those without allocating).
func pbChild(parent pbKind, num protowire.Number) (child pbKind, ok bool) {
	switch parent {
	case pbRequest:
		return pbResourceSpans, num == 1
	case pbResourceSpans:
		switch num {
		case 1:
			return pbResource, true
		case 2, 1000: // 1000: the deprecated InstrumentationLibrarySpans
			return pbScopeSpans, true
		}
	case pbResource:
		return pbKeyValue, num == 1
	case pbScopeSpans:
		switch num {
		case 1:
			return pbScope, true
		case 2:
			return pbSpan, true
		}
	case pbScope:
		return pbKeyValue, num == 3
	case pbSpan:
		switch num {
		case 9:
			return pbKeyValue, true
		case 11:
			return pbEvent, true
		case 13:
			return pbLink, true
		}
	case pbEvent:
		return pbKeyValue, num == 3
	case pbLink:
		return pbKeyValue, num == 4
	case pbKeyValue:
		return pbAnyValue, num == 2
	case pbAnyValue:
		switch num {
		case 5:
			return pbArrayValue, true
		case 6:
			return pbKeyValueList, true
		}
	case pbArrayValue:
		return pbAnyValue, num == 1
	case pbKeyValueList:
		return pbKeyValue, num == 1
	}
	return 0, false
}

func (c *decodeCounts) addProto(parent, child pbKind) {
	switch child {
	case pbResourceSpans:
		c.resourceSpans++
	case pbScopeSpans:
		c.scopeSpans++
	case pbSpan:
		c.spans++
	case pbEvent:
		c.events++
	case pbLink:
		c.links++
	case pbKeyValue:
		c.keyValues++
	case pbAnyValue:
		if parent == pbArrayValue { // a KeyValue's own value is part of it
			c.anyValues++
		}
	case pbArrayValue, pbKeyValueList:
		c.containers++
	}
}

// countProtoElements counts the elements of a protobuf
// ExportTraceServiceRequest without decoding it.
func countProtoElements(b []byte) (decodeCounts, error) {
	var c decodeCounts
	err := c.walkProto(b, pbRequest, 0)
	return c, err
}

func (c *decodeCounts) walkProto(b []byte, kind pbKind, depth int) error {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return protowire.ParseError(n)
		}
		b = b[n:]
		if typ == protowire.BytesType {
			if child, ok := pbChild(kind, num); ok {
				msg, m := protowire.ConsumeBytes(b)
				if m < 0 {
					return protowire.ParseError(m)
				}
				b = b[m:]
				if depth >= maxDecodeNesting {
					return errNestingTooDeep
				}
				c.addProto(kind, child)
				if err := c.walkProto(msg, child, depth+1); err != nil {
					return err
				}
				continue
			}
		}
		m := protowire.ConsumeFieldValue(num, typ, b)
		if m < 0 {
			return protowire.ParseError(m)
		}
		b = b[m:]
	}
	return nil
}

// ── JSON ───────────────────────────────────────────────────────────────────

// jsonKey is what an object key makes of the value that follows it.
type jsonKey uint8

const (
	keyOther jsonKey = iota
	keyResourceSpans
	keyScopeSpans
	keySpans
	keyEvents
	keyLinks
	keyKeyValues // attributes, and the values of an arrayValue or kvlistValue
	keyContainer // arrayValue, kvlistValue
)

// jsonKeyKind maps an object key to what pdata v1.20.0 decodes under it.
// pdata appends one element per entry of these arrays, whatever the entry is
// (null included), and matches keys exactly after unescaping them.
func jsonKeyKind(key []byte) jsonKey {
	switch string(key) {
	case "resourceSpans", "resource_spans":
		return keyResourceSpans
	case "scopeSpans", "scope_spans":
		return keyScopeSpans
	case "spans":
		return keySpans
	case "events":
		return keyEvents
	case "links":
		return keyLinks
	case "attributes", "values":
		return keyKeyValues
	case "arrayValue", "array_value", "kvlistValue", "kvlist_value":
		return keyContainer
	}
	return keyOther
}

func (c *decodeCounts) addJSON(k jsonKey) {
	switch k {
	case keyResourceSpans:
		c.resourceSpans++
	case keyScopeSpans:
		c.scopeSpans++
	case keySpans:
		c.spans++
	case keyEvents:
		c.events++
	case keyLinks:
		c.links++
	case keyKeyValues:
		c.keyValues++
	}
}

type jsonFrame struct {
	array bool
	elem  jsonKey // array: what its entries decode into
	// expect is, for an array, that the next token starts an entry; for an
	// object, that the next string is a key.
	expect bool
}

// countJSONElements counts the elements of an OTLP/JSON
// ExportTraceServiceRequest without decoding it. It only fails on nesting
// that is too deep; anything else malformed is left to the decoder, which
// refuses it.
func countJSONElements(b []byte) (decodeCounts, error) {
	var c decodeCounts
	stack := make([]jsonFrame, 0, 16)
	key := keyOther // the last key read, until its value starts
	for i := 0; i < len(b); i++ {
		ch := b[i]
		if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
			continue
		}
		top := len(stack) - 1
		if top >= 0 && stack[top].array && stack[top].expect && ch != ']' {
			stack[top].expect = false
			c.addJSON(stack[top].elem)
		}
		switch ch {
		case '"':
			end, escaped := jsonStringEnd(b, i+1)
			if end < 0 {
				return c, nil // unterminated: the decoder refuses it
			}
			if top >= 0 && !stack[top].array && stack[top].expect {
				stack[top].expect = false
				raw := b[i+1 : end]
				if escaped {
					var buf [32]byte
					raw = unescapeJSONKey(raw, buf[:0])
				}
				key = jsonKeyKind(raw)
			} else {
				key = keyOther
			}
			i = end
		case ':':
			// The key stays pending until its value starts.
		case ',':
			if top >= 0 {
				stack[top].expect = true
			}
			key = keyOther
		case '{', '[':
			if len(stack) >= maxDecodeNesting {
				return c, errNestingTooDeep
			}
			if ch == '{' {
				if key == keyContainer {
					c.containers++
				}
				stack = append(stack, jsonFrame{expect: true})
			} else {
				stack = append(stack, jsonFrame{array: true, elem: key, expect: true})
			}
			key = keyOther
		case '}', ']':
			if top >= 0 {
				stack = stack[:top]
			}
			key = keyOther
		default: // numbers, true, false, null
			key = keyOther
		}
	}
	return c, nil
}

// jsonStringEnd returns the index of the quote closing the string that starts
// at b[start], or -1, and whether the string holds any escape.
func jsonStringEnd(b []byte, start int) (end int, escaped bool) {
	for j := start; j < len(b); j++ {
		switch b[j] {
		case '\\':
			escaped = true
			j++
		case '"':
			return j, escaped
		}
	}
	return -1, escaped
}

// unescapeJSONKey decodes the escapes of a short key into buf. Keys that
// cannot be one pdata knows (too long, not ASCII, a bad escape) come back as
// a value that matches none.
func unescapeJSONKey(raw, buf []byte) []byte {
	none := []byte{0}
	for j := 0; j < len(raw); j++ {
		if len(buf) == cap(buf) {
			return none
		}
		ch := raw[j]
		if ch != '\\' {
			buf = append(buf, ch)
			continue
		}
		if j+1 >= len(raw) {
			return none
		}
		j++
		switch raw[j] {
		case '"', '\\', '/':
			buf = append(buf, raw[j])
		case 'b', 'f', 'n', 'r', 't':
			return none // no key pdata knows has a control character
		case 'u':
			if j+4 >= len(raw) {
				return none
			}
			var r rune
			for _, h := range raw[j+1 : j+5] {
				r <<= 4
				switch {
				case h >= '0' && h <= '9':
					r |= rune(h - '0')
				case h >= 'a' && h <= 'f':
					r |= rune(h-'a') + 10
				case h >= 'A' && h <= 'F':
					r |= rune(h-'A') + 10
				default:
					return none
				}
			}
			if r == 0 || r >= 0x80 {
				return none
			}
			buf = append(buf, byte(r))
			j += 4
		default:
			return none
		}
	}
	return buf
}

// ── gzip ───────────────────────────────────────────────────────────────────

// gzipSizeHint is the decompressed size a gzip body declares in its trailer
// (ISIZE, RFC 1952 section 2.3.1). It is only a hint: a sender can write any
// value, and a multi-member body declares just its last member.
func gzipSizeHint(wire []byte) int64 {
	if len(wire) < 18 { // a 10-byte header and an 8-byte trailer at least
		return 0
	}
	return int64(binary.LittleEndian.Uint32(wire[len(wire)-4:]))
}
