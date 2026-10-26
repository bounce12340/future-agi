import { readFile } from "node:fs/promises";
import type { Route } from "@playwright/test";
import { test, expect } from "../../lib/fixtures";
import { flowAnnotation } from "../../lib/flow-meta";

// Pinned from frontend/src/pages/dashboard/error-feed/exportTracesCsv.js (COLUMNS).
const CSV_HEADER =
  "trace_id,timestamp,model,score,latency_ms,turns,input_tokens," +
  "output_tokens,input,output,judge_reason,failure_evidence";
const UI_READY = 60_000;
const TIMESTAMP = "2026-10-20T10:00:00Z";

// The browser calls the API cross-origin, so a stubbed body needs the CORS
// headers the real API would send.
const fulfillJson = (route: Route, json: unknown) =>
  route.fulfill({
    json,
    headers: {
      "access-control-allow-origin": route.request().headers()["origin"] ?? "*",
      "access-control-allow-credentials": "true",
    },
  });

// The E2E issue seeder creates issues without trace memberships, so the real
// overview would return no trace cards. Stub the entitlement and the issue's
// redirect, detail and overview responses; the CSV the browser builds from
// them is the subject. Bodies follow FeedDetailCoreSerializer and
// OverviewResponseSerializer (futureagi/tracer/serializers/feed.py).
test(
  "ERR-E2E-004: an issue's traces download as a dataset-ready CSV",
  {
    tag: ["@flow"],
    annotation: flowAnnotation({
      id: "ERR-E2E-004",
      area: "error-feed",
      userGoal:
        "A developer downloads an issue's traces from the issue page as a dataset-ready CSV (overview response stubbed; dataset upload not exercised)",
      steps: ["open the issue page", "click Export", "save the downloaded CSV"],
      browserChecks: [
        "the export requests the issue overview once, with rep_limit=200",
        "the file is named error-feed-<issue>-traces.csv and its header is exactly the 12 export columns",
        "the file has exactly one row per stubbed trace, with quoted cells intact and no formula prefix",
        "the snackbar reports exactly the exported trace count",
      ],
    }),
  },
  async ({ page }, testInfo) => {
    // Three UI_READY waits (180 s) plus navigation pass the 120 s default.
    test.setTimeout(240_000);

    const suffix = `${testInfo.workerIndex}-${Date.now().toString(36)}`;
    const issueId = `e2e-err4-${suffix}`;
    const traceA = `e2e-trace-a-${suffix}`;
    const traceB = `e2e-trace-b-${suffix}`;
    const inputA = `- book a table, "for two"\nthen confirm ${suffix}`;
    const outputA = `+1 555 0100 ${suffix}`;
    await testInfo.attach("stubbed-issue", {
      body: JSON.stringify({ issueId, traceIds: [traceA, traceB] }),
      contentType: "application/json",
    });

    const card = (
      id: string,
      input: string,
      output: string,
      finding: string,
    ) => ({
      id,
      status: "fail",
      timestamp: TIMESTAMP,
      summary: {
        eval_score: 0.2,
        latency_ms: 1200,
        turns: 3,
        model: "gpt-4o",
        input_tokens: 500,
        output_tokens: 80,
      },
      evidence: {
        input,
        output,
        fail_reel: [
          {
            label: "FINDING",
            text: finding,
            span: null,
            status: "neutral",
            isFailure: false,
            raw: null,
            meta: null,
          },
        ],
        pass_reel: [],
        judge_reason: `judge ${suffix}`,
        score: 0.2,
      },
      agent_flow: { nodes: [], edges: [] },
      root_causes: [],
      recommendations: [],
      what_changed: null,
    });
    const overview = {
      events_over_time: [],
      pattern_summary: { insights: [], key_moments: [] },
      representative_traces: [
        card(traceA, inputA, outputA, `finding a ${suffix}`),
        card(
          traceB,
          `input b ${suffix}`,
          `output b ${suffix}`,
          `finding b ${suffix}`,
        ),
      ],
      representative_total: 2,
    };
    const detail = {
      row: {
        cluster_id: issueId,
        source: "scanner",
        modality: "text",
        error: { name: `E2E export ${suffix}`, type: "E2EExport" },
        status: "for_review",
        severity: "high",
        occurrences: 2,
        trace_count: 2,
        fix_layer: null,
        users_affected: 0,
        sessions: 0,
        first_seen: TIMESTAMP,
        last_seen: TIMESTAMP,
        trends: [],
        assignees: [],
        model: null,
        model_version: null,
        project: null,
        project_id: null,
        environment: null,
        eval_score: null,
        trace_id: traceA,
        external_issue_url: null,
        external_issue_id: null,
      },
      description: null,
      success_trace: null,
      representative_trace: null,
      rca: null,
    };

    const exportQueries: string[] = [];
    await page.route("**/api/capabilities/", (route) =>
      fulfillJson(route, { features: { error_feed: { allowed: true } } }),
    );
    const issuePath = `/tracer/feed/issues/${issueId}/`;
    await page.route(
      (url) => url.pathname.includes(issuePath),
      (route) => {
        const url = new URL(route.request().url());
        const rest = url.pathname.split(issuePath)[1];
        if (rest === "") {
          return fulfillJson(route, { status: true, result: detail });
        }
        if (rest === "redirect/") {
          return fulfillJson(route, {
            result: {
              requested_cluster_id: issueId,
              resolved_cluster_id: issueId,
            },
          });
        }
        if (rest === "overview/") {
          // The Overview tab reads without rep_limit; only the export sets it.
          if (url.searchParams.has("rep_limit")) exportQueries.push(url.search);
          return fulfillJson(route, { status: true, result: overview });
        }
        return route.fallback();
      },
    );

    await page.goto(`/dashboard/error-feed/${issueId}`);
    const exportButton = page.getByRole("button", {
      name: "Export",
      exact: true,
    });
    await expect(exportButton).toBeEnabled({ timeout: UI_READY });

    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: UI_READY }),
      exportButton.click(),
    ]);
    expect(exportQueries).toEqual(["?rep_limit=200"]);

    expect(download.suggestedFilename()).toBe(
      `error-feed-${issueId}-traces.csv`,
    );
    const csv = await readFile(await download.path(), "utf8");
    expect(csv.split("\r\n")[0]).toBe(CSV_HEADER);
    expect(csv).toBe(
      [
        CSV_HEADER,
        `${traceA},${TIMESTAMP},gpt-4o,0.2,1200,3,500,80,` +
          `"- book a table, ""for two""\nthen confirm ${suffix}",` +
          `+1 555 0100 ${suffix},judge ${suffix},FINDING: finding a ${suffix}`,
        `${traceB},${TIMESTAMP},gpt-4o,0.2,1200,3,500,80,` +
          `input b ${suffix},output b ${suffix},judge ${suffix},` +
          `FINDING: finding b ${suffix}`,
      ].join("\r\n"),
    );

    await expect(
      page.getByText("Exported 2 traces", { exact: true }),
    ).toBeVisible({ timeout: UI_READY });
  },
);
