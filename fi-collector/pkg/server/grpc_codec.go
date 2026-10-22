package server

import (
	"google.golang.org/grpc/encoding"
	"google.golang.org/grpc/encoding/proto"
	"google.golang.org/grpc/mem"
	"google.golang.org/protobuf/protoadapt"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// traceExportRequestName is the message checkProtoNesting walks.
const traceExportRequestName protoreflect.FullName = "opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest"

// nestingCheckedCodec is the gRPC server's codec: grpc-go's proto codec, plus
// the nesting limit (checkProtoNesting) on an ExportTraceServiceRequest before
// it is decoded, as the OTLP/HTTP handler does. grpc-go decodes a request
// before any interceptor runs, auth included, so the codec is the only place
// that sees the bytes first. A refused request fails with INTERNAL ("error
// unmarshalling request"), which is what grpc-go returns for any request its
// codec cannot decode. Other message types are decoded unchecked.
type nestingCheckedCodec struct {
	encoding.CodecV2
}

func newNestingCheckedCodec() nestingCheckedCodec {
	return nestingCheckedCodec{encoding.GetCodecV2(proto.Name)}
}

func (c nestingCheckedCodec) Unmarshal(data mem.BufferSlice, v any) error {
	if !isTraceExportRequest(v) {
		return c.CodecV2.Unmarshal(data, v)
	}
	// Join the message's buffers once and hand the same buffer to the proto
	// codec, which then decodes it without another copy.
	buf := data.MaterializeToBuffer(mem.DefaultBufferPool())
	defer buf.Free()
	if err := checkProtoNesting(buf.ReadOnlyData()); err != nil {
		return err
	}
	return c.CodecV2.Unmarshal(mem.BufferSlice{buf}, v)
}

// isTraceExportRequest reports whether v is an ExportTraceServiceRequest.
// pdata's generated type is internal, so it is matched by its protobuf name.
func isTraceExportRequest(v any) bool {
	var m protoreflect.ProtoMessage
	switch v := v.(type) {
	case protoadapt.MessageV1:
		m = protoadapt.MessageV2Of(v)
	case protoadapt.MessageV2:
		m = v
	default:
		return false
	}
	return m.ProtoReflect().Descriptor().FullName() == traceExportRequestName
}
