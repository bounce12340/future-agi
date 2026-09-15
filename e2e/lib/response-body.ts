import type { Response } from '@playwright/test';

/**
 * Read a response's JSON body within `ms`, or reject with `body-timeout`.
 *
 * A flow's capture harness reads the body of every catalog/query response the
 * page receives and awaits those reads before it judges the receipts. A body
 * that never arrives — a response cut by the navigation that reopened the page
 * (CI run 34912956841: one `Response.body` started at +31.6 s and was still
 * open at the 660 s test ceiling) — must record an error the flow can attach,
 * not hold the flow until its ceiling with no attribution.
 */
// Defaults to `any`, exactly like Response.json(), so it is a drop-in replacement.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function readJsonWithin<T = any>(response: Response, ms: number): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(
      `body-timeout after ${ms} ms: ${response.request().method()} ${new URL(response.url()).pathname}`)), ms);
  });
  try {
    return await Promise.race([response.json() as Promise<T>, deadline]);
  } finally {
    clearTimeout(timer);
  }
}
