package auth

import (
	"fmt"
	"net/http"
	"strings"
	"unicode"
	"unicode/utf8"

	"go.opentelemetry.io/collector/pdata/ptrace"
	"google.golang.org/grpc/metadata"
)

// ProjectNameHeader names the Observe project for exporters that can set
// request headers but not resource attributes (OpenRouter Broadcast, vendor
// telemetry drains, edge runtimes). gRPC carries it as metadata under the same
// name; gRPC metadata keys are lowercase, so that is x-project-name.
//
// ProjectNameHeaderAlias is read too, because traceAI's Java and C# exporters
// already send FI-Project-Name (and traceAI's agno, beeai and strands
// instrumentors already send x-project-name). Both headers name one project:
// values that disagree are rejected.
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
	ProjectNameHeaderAlias = "FI-Project-Name"
)

// resourceProjectNameKey is the resource attribute StampResourceAttrs reads.
const resourceProjectNameKey = "project_name"

// maxProjectNameChars is the longest name the header may carry: the project
// table's name column holds 255 characters, so a longer name could never be
// matched or created, and its spans would be dropped without an error the
// sender sees.
const maxProjectNameChars = 255

// ProjectNameHTTPValues returns every value of the project headers on an
// OTLP/HTTP request, canonical header first.
func ProjectNameHTTPValues(h http.Header) []string {
	return append(append([]string(nil), h.Values(ProjectNameHeader)...), h.Values(ProjectNameHeaderAlias)...)
}

// ProjectNameMetadataValues returns every value of the project metadata keys
// on an OTLP/gRPC call, canonical key first.
func ProjectNameMetadataValues(md metadata.MD) []string {
	return append(append([]string(nil), md.Get(ProjectNameHeader)...), md.Get(ProjectNameHeaderAlias)...)
}

// ApplyProjectNameHeader writes the project named by the header values into
// the `project_name` resource attribute of every ResourceSpans that has no
// usable one: absent, empty, or not a string (the cases StampResourceAttrs
// counts as missing). A ResourceSpans that names its own project keeps it.
//
// The header is only read when some ResourceSpans needs it, so a request whose
// resources all name a project is unaffected by the header, whatever it holds.
// When it is needed, values are trimmed and blank ones ignored; values that
// disagree, are not valid UTF-8, are longer than 255 characters or contain a
// control character are an error. It returns how many ResourceSpans took the
// header's project.
func ApplyProjectNameHeader(traces ptrace.Traces, values []string) (int, error) {
	rss := traces.ResourceSpans()
	need := false
	for i := 0; i < rss.Len() && !need; i++ {
		need = getStrAttr(rss.At(i).Resource().Attributes(), resourceProjectNameKey) == ""
	}
	if !need || len(values) == 0 {
		return 0, nil
	}
	name, err := projectNameFromHeaderValues(values)
	if err != nil || name == "" {
		return 0, err
	}
	applied := 0
	for i := 0; i < rss.Len(); i++ {
		attrs := rss.At(i).Resource().Attributes()
		if getStrAttr(attrs, resourceProjectNameKey) != "" {
			continue
		}
		attrs.PutStr(resourceProjectNameKey, name)
		applied++
	}
	return applied, nil
}

func projectNameFromHeaderValues(values []string) (string, error) {
	name := ""
	for _, v := range values {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		if !utf8.ValidString(v) {
			return "", fmt.Errorf("%s is not valid UTF-8", ProjectNameHeader)
		}
		if utf8.RuneCountInString(v) > maxProjectNameChars {
			return "", fmt.Errorf("%s is longer than %d characters", ProjectNameHeader, maxProjectNameChars)
		}
		if strings.IndexFunc(v, unicode.IsControl) >= 0 {
			return "", fmt.Errorf("%s contains a control character", ProjectNameHeader)
		}
		if name != "" && v != name {
			return "", fmt.Errorf("conflicting project names in %s / %s", ProjectNameHeader, ProjectNameHeaderAlias)
		}
		name = v
	}
	return name, nil
}
