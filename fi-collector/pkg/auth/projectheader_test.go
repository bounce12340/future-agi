package auth

import (
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/collector/pdata/ptrace"
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
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ProjectNameFromHeaderValues(tc.values)
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

func TestApplyProjectNameDefaultFillsOnlyMissingNames(t *testing.T) {
	traces := ptrace.NewTraces()
	named := traces.ResourceSpans().AppendEmpty()
	named.Resource().Attributes().PutStr("project_name", "from-attr")
	absent := traces.ResourceSpans().AppendEmpty()
	absent.Resource().Attributes().PutStr("service.name", "svc")
	empty := traces.ResourceSpans().AppendEmpty()
	empty.Resource().Attributes().PutStr("project_name", "")
	wrongType := traces.ResourceSpans().AppendEmpty()
	wrongType.Resource().Attributes().PutInt("project_name", 7)

	if n := ApplyProjectNameDefault(traces, "from-header"); n != 3 {
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

func TestApplyProjectNameDefaultEmptyNameIsNoOp(t *testing.T) {
	traces := ptrace.NewTraces()
	traces.ResourceSpans().AppendEmpty()
	if n := ApplyProjectNameDefault(traces, ""); n != 0 {
		t.Fatalf("applied = %d, want 0", n)
	}
	if _, ok := traces.ResourceSpans().At(0).Resource().Attributes().Get("project_name"); ok {
		t.Error("an empty default must not add project_name")
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

	if n := ApplyProjectNameDefault(traces, "header-project"); n != 1 {
		t.Fatalf("applied = %d, want 1", n)
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
