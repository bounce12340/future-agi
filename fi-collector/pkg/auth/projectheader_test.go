package auth

import (
	"context"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/collector/pdata/ptrace"
	"google.golang.org/grpc/metadata"
)

func TestProjectNameFromHeaderValues(t *testing.T) {
	cases := []struct {
		name    string
		values  []string
		want    string
		wantErr string
	}{
		{name: "absent", values: nil, want: ""},
		{name: "single", values: []string{"checkout-bot"}, want: "checkout-bot"},
		{name: "trimmed", values: []string{"  checkout-bot\t"}, want: "checkout-bot"},
		{name: "blank is absent", values: []string{"   "}, want: ""},
		{name: "repeats that agree", values: []string{"a", " a ", ""}, want: "a"},
		{name: "unicode", values: []string{"проект-ü"}, want: "проект-ü"},
		{name: "conflicting", values: []string{"a", "b"}, wantErr: "conflicting"},
		{name: "invalid utf-8", values: []string{"bad\xff"}, wantErr: "UTF-8"},
		// The project name column holds 255 characters (not bytes).
		{name: "255 characters", values: []string{strings.Repeat("é", 255)}, want: strings.Repeat("é", 255)},
		{name: "256 characters", values: []string{strings.Repeat("é", 256)}, wantErr: "255 characters"},
		{name: "tab inside", values: []string{"a\tb"}, wantErr: "control character"},
		{name: "newline inside", values: []string{"a\nb"}, wantErr: "control character"},
		{name: "DEL inside", values: []string{"a\x7fb"}, wantErr: "control character"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := projectNameFromHeaderValues(tc.values)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want it to mention %q", err, tc.wantErr)
				}
				if !strings.Contains(err.Error(), ProjectNameHeader) {
					t.Errorf("err %q should name the header", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestProjectNameValuesReadCanonicalAndAliasHeaders(t *testing.T) {
	h := http.Header{}
	h.Add("x-project-name", "a")
	h.Add("Fi-Project-Name", "b")
	if got := ProjectNameHTTPValues(h); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("HTTP values = %v, want [a b] (canonical first)", got)
	}
	md := metadata.Pairs("fi-project-name", "b", "x-project-name", "a")
	if got := ProjectNameMetadataValues(md); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("metadata values = %v, want [a b] (canonical first)", got)
	}
}

func TestApplyProjectNameHeaderFillsOnlyMissingNames(t *testing.T) {
	traces := ptrace.NewTraces()
	named := traces.ResourceSpans().AppendEmpty()
	named.Resource().Attributes().PutStr("project_name", "from-attr")
	absent := traces.ResourceSpans().AppendEmpty()
	absent.Resource().Attributes().PutStr("service.name", "svc")
	empty := traces.ResourceSpans().AppendEmpty()
	empty.Resource().Attributes().PutStr("project_name", "")
	wrongType := traces.ResourceSpans().AppendEmpty()
	wrongType.Resource().Attributes().PutInt("project_name", 7)

	n, err := ApplyProjectNameHeader(traces, []string{" from-header "})
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("applied = %d, want 3 (absent, empty, non-string)", n)
	}
	want := []string{"from-attr", "from-header", "from-header", "from-header"}
	for i, w := range want {
		if got := getStrAttr(traces.ResourceSpans().At(i).Resource().Attributes(), "project_name"); got != w {
			t.Errorf("ResourceSpans[%d] project_name = %q, want %q", i, got, w)
		}
	}
	if got := getStrAttr(absent.Resource().Attributes(), "service.name"); got != "svc" {
		t.Errorf("other resource attributes must be kept, service.name = %q", got)
	}
}

func TestApplyProjectNameHeaderNoValuesIsNoOp(t *testing.T) {
	for _, values := range [][]string{nil, {""}, {"  "}} {
		traces := ptrace.NewTraces()
		traces.ResourceSpans().AppendEmpty()
		n, err := ApplyProjectNameHeader(traces, values)
		if err != nil || n != 0 {
			t.Fatalf("values %q: applied = %d, err = %v; want 0, nil", values, n, err)
		}
		if _, ok := traces.ResourceSpans().At(0).Resource().Attributes().Get("project_name"); ok {
			t.Errorf("values %q must not add project_name", values)
		}
	}
}

// A bad header is only an error when the request needs it: requests whose
// resources all name their project behave exactly as before.
func TestApplyProjectNameHeaderIgnoresBadHeaderWhenNotNeeded(t *testing.T) {
	traces := ptrace.NewTraces()
	traces.ResourceSpans().AppendEmpty().Resource().Attributes().PutStr("project_name", "from-attr")
	if n, err := ApplyProjectNameHeader(traces, []string{"a", "b", "bad\xff"}); err != nil || n != 0 {
		t.Fatalf("applied = %d, err = %v; want 0, nil when every resource names a project", n, err)
	}

	traces.ResourceSpans().AppendEmpty()
	if _, err := ApplyProjectNameHeader(traces, []string{"a", "b"}); err == nil {
		t.Fatal("conflicting values must be an error once a resource needs the header")
	}
}

// recordingProjectResolver knows some projects and auto-creates the rest,
// recording what it was asked to create.
type recordingProjectResolver struct {
	known   map[string]string
	created []string
}

func (r *recordingProjectResolver) ResolveProjects(
	_ context.Context, _, _ string, names []string,
) (map[string]string, error) {
	out := make(map[string]string)
	for _, n := range names {
		if id, ok := r.known[n]; ok {
			out[n] = id
		}
	}
	return out, nil
}

func (r *recordingProjectResolver) GetOrCreateProject(
	_ context.Context, orgID, workspaceID, name, traceType string,
) (string, error) {
	r.created = append(r.created, orgID+"/"+workspaceID+"/"+name+"/"+traceType)
	return "created-" + name, nil
}

// A header-named project goes through the same workspace-scoped lookup and
// observe auto-create as an attribute-named one, and a ResourceSpans that
// names its own project in the same request keeps it.
func TestStampResolvesHeaderProjectLikeAttributeProject(t *testing.T) {
	resolver := &recordingProjectResolver{known: map[string]string{"attr-project": "id-attr"}}
	a := &Authenticator{cache: newCache(5*time.Minute, time.Hour), projects: resolver, log: slog.Default()}
	result := &ResolveResult{OrgID: "org-h", WorkspaceID: "ws-h", Projects: map[string]string{}}
	ck := "header-project-key"
	a.cache.putPositive(ck, result)

	traces := ptrace.NewTraces()
	fromAttr := traces.ResourceSpans().AppendEmpty()
	fromAttr.Resource().Attributes().PutStr("project_name", "attr-project")
	fromAttr.ScopeSpans().AppendEmpty().Spans().AppendEmpty().SetName("a")
	fromHeader := traces.ResourceSpans().AppendEmpty()
	fromHeader.ScopeSpans().AppendEmpty().Spans().AppendEmpty().SetName("b")

	// Without the header the request is rejected, as before.
	if _, err := StampResourceAttrs(context.Background(), a, ck, cloneTraces(traces), result); err == nil {
		t.Fatal("a ResourceSpans with no project_name and no header must still be rejected")
	}

	if n, err := ApplyProjectNameHeader(traces, []string{"header-project"}); err != nil || n != 1 {
		t.Fatalf("applied = %d, err = %v; want 1, nil", n, err)
	}
	dropped, err := StampResourceAttrs(context.Background(), a, ck, traces, result)
	if err != nil {
		t.Fatalf("stamp: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("dropped = %d, want 0", dropped)
	}
	got := []string{
		getStrAttr(traces.ResourceSpans().At(0).Resource().Attributes(), "fi.project_id"),
		getStrAttr(traces.ResourceSpans().At(1).Resource().Attributes(), "fi.project_id"),
	}
	if got[0] != "id-attr" || got[1] != "created-header-project" {
		t.Errorf("fi.project_id = %v, want [id-attr created-header-project]", got)
	}
	if len(resolver.created) != 1 || resolver.created[0] != "org-h/ws-h/header-project/observe" {
		t.Errorf("auto-create calls = %v, want one observe project in the key's workspace", resolver.created)
	}
}

func cloneTraces(t ptrace.Traces) ptrace.Traces {
	out := ptrace.NewTraces()
	t.CopyTo(out)
	return out
}
