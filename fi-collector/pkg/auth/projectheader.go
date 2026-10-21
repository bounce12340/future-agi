package auth

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"go.opentelemetry.io/collector/pdata/ptrace"
)

// ProjectNameHeader names the Observe project for exporters that can set
// request headers but not resource attributes (OpenRouter Broadcast, vendor
// telemetry drains, edge runtimes). On gRPC it is the metadata key
// ProjectNameMetadataKey, since gRPC metadata keys are lowercase.
//
// Precedence, per ResourceSpans: a non-empty string resource attribute
// `project_name` wins; otherwise the header value is written into
// `project_name`; otherwise the request behaves as before (rejected when auth
// is on, because every ResourceSpans needs a project). A name that came from
// the header is then resolved, workspace-scoped and auto-created exactly like
// one that came from the attribute, because StampResourceAttrs only ever reads
// the attribute.
const (
	ProjectNameHeader      = "X-Project-Name"
	ProjectNameMetadataKey = "x-project-name"
)

// resourceProjectNameKey is the resource attribute StampResourceAttrs reads.
const resourceProjectNameKey = "project_name"

// ProjectNameFromHeaderValues returns the project named by every value of the
// X-Project-Name header (or x-project-name metadata key), or "" when none
// names one. Values are trimmed and blank values ignored. Repeated values must
// agree, since there is no rule for picking one, and the name must be valid
// UTF-8 because it becomes an OTLP string attribute.
func ProjectNameFromHeaderValues(values []string) (string, error) {
	name := ""
	for _, v := range values {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		if !utf8.ValidString(v) {
			return "", fmt.Errorf("%s is not valid UTF-8", ProjectNameHeader)
		}
		if name != "" && v != name {
			return "", fmt.Errorf("conflicting %s values", ProjectNameHeader)
		}
		name = v
	}
	return name, nil
}

// ApplyProjectNameDefault writes name into the `project_name` resource
// attribute of every ResourceSpans that has no usable one: absent, empty, or
// not a string (the same cases StampResourceAttrs counts as missing). A
// ResourceSpans that already names its project keeps it. It returns how many
// ResourceSpans took the default; an empty name changes nothing.
func ApplyProjectNameDefault(traces ptrace.Traces, name string) int {
	if name == "" {
		return 0
	}
	applied := 0
	rss := traces.ResourceSpans()
	for i := 0; i < rss.Len(); i++ {
		attrs := rss.At(i).Resource().Attributes()
		if getStrAttr(attrs, resourceProjectNameKey) != "" {
			continue
		}
		attrs.PutStr(resourceProjectNameKey, name)
		applied++
	}
	return applied
}
