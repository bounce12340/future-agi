/* eslint-disable react/prop-types */
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, userEvent, waitFor } from "src/utils/test-utils";
import SetupOrganization from "../setup-org";

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  enqueueSnackbar: vi.fn(),
}));

vi.mock("src/utils/axios", () => ({
  default: { get: mocks.get, post: mocks.post },
  endpoints: {
    auth: {
      create_org: "/accounts/team/users/",
      user_onboarding_info: "/accounts/user-onboarding/",
    },
    settings: { teams: {
      getMemberList: "/accounts/team/users/",
      inviteMember: "/accounts/team/users/",
    } },
  },
}));
vi.mock("src/components/snackbar", () => ({
  useSnackbar: () => ({ enqueueSnackbar: mocks.enqueueSnackbar }),
}));
vi.mock("src/auth/hooks", () => ({
  useAuthContext: () => ({
    user: { email: "owner@example.com", organization_role: "Owner" },
  }),
}));
vi.mock("src/hooks/useDeploymentMode", () => ({
  useDeploymentMode: () => ({ isOSS: false, isSuccess: true }),
}));
vi.mock("src/utils/Mixpanel", () => ({
  Events: {}, PropertyName: {}, trackEvent: vi.fn(),
}));
vi.mock("../RightSectionAuth", () => ({ default: () => null }));
vi.mock("src/components/svg-color", () => ({ default: () => null }));
vi.mock("src/components/RadioField/RadioField", () => ({ default: () => null }));
vi.mock("src/components/FormCheckboxField", () => ({ FormCheckboxField: () => null }));
// Keep form validation and mutation behavior real, without unrelated field widgets.
vi.mock("src/components/FormTextField/FormTextFieldV2", async () => {
  const { Controller } = await import("react-hook-form");
  return { default: ({ control, fieldName, label }) => (
    <Controller control={control} name={fieldName} render={({ field }) => (
      <label>{label}<input {...field} value={field.value ?? ""} /></label>
    )} />
  ) };
});
vi.mock("src/components/FromSearchSelectField", () => ({
  FormSearchSelectFieldState: ({ label, value, onChange, options, disabled }) => (
    <label>{label}
      <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
  ),
}));

const email = "newhire@example.com";
const warning = "Member added, but the invitation email could not be sent. Open Members to resend the invite or copy an invite link if available.";
const createdMember = { id: "member-1", email, name: "Newhire" };

function renderSetup(getStarted = true) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <SetupOrganization getStarted={getStarted} />
    </QueryClientProvider>,
  );
}

async function addMember(user, address = email, getStarted = true) {
  await screen.findByDisplayValue("owner@example.com");
  await user.click(screen.getByText(getStarted ? "Add members" : "Add", { exact: true }));
  const input = screen.getAllByRole("textbox", { name: "Email" }).find((field) => !field.disabled && !field.value);
  await user.type(input, address);
}

async function submit(user) {
  const button = screen.getByRole("button", { name: "Continue", exact: true });
  await waitFor(() => expect(button).toBeEnabled());
  await user.click(button);
}

function respondWith(members) {
  mocks.post.mockResolvedValue({
    status: 201,
    data: { status: true, result: { created_members: members } },
  });
}

describe("organization setup invitation delivery", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    window.history.replaceState({}, "", "/auth/jwt/setup-org?step=2");
    mocks.get.mockImplementation(async (url, config) => ({
      data: { result: url === "/accounts/team/users/" ? {
        org_name: "Test Organization",
        results: config?.params ? [] : [{
          id: "owner", email: "owner@example.com", name: "Owner",
          organization_role: "Owner",
        }],
      } : {} },
    }));
  });

  it("reports a healthy invite as sent", async () => {
    respondWith([createdMember]);
    const user = userEvent.setup();
    renderSetup();
    await addMember(user);
    await submit(user);

    await waitFor(() => expect(mocks.enqueueSnackbar).toHaveBeenCalledWith(
      "Invite sent successfully.", { variant: "success" },
    ));
    expect(mocks.post).toHaveBeenCalledWith("/accounts/team/users/", {
      org_name: "Test Organization",
      members: [{ index: 1, email, name: "Newhire", organization_role: "Member" }],
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows recovery guidance, not sent-success, and acknowledges without inviting twice", async () => {
    respondWith([{ ...createdMember, invite_email_warning: warning }]);
    const user = userEvent.setup();
    renderSetup();
    await addMember(user);
    await submit(user);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(`${email}: ${warning}`);
    expect(screen.getByText("Members added", { exact: true })).toBeInTheDocument();
    expect(mocks.enqueueSnackbar).not.toHaveBeenCalledWith(
      "Invite sent successfully.", expect.anything(),
    );
    await user.click(screen.getByRole("button", { name: "Done", exact: true }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(mocks.post).toHaveBeenCalledTimes(1);
  });

  it("keeps onboarding warnings visible until the user continues to the dashboard", async () => {
    respondWith([{ ...createdMember, invite_email_warning: warning }]);
    const user = userEvent.setup();
    renderSetup(false);
    await addMember(user, email, false);
    await submit(user);

    expect(await screen.findByRole("alert")).toHaveTextContent(warning);
    expect(screen.getByRole("button", { name: "Continue to dashboard" })).toBeEnabled();
    expect(window.location.pathname).toBe("/auth/jwt/setup-org");
    expect(screen.queryByRole("button", { name: "Continue", exact: true })).not.toBeInTheDocument();
    expect(mocks.post).toHaveBeenCalledTimes(1);
  });

  it("identifies only the failed delivery in a mixed bulk result", async () => {
    respondWith([
      createdMember,
      { id: "member-2", email: "failed@example.com", invite_email_warning: warning },
    ]);
    const user = userEvent.setup();
    renderSetup();
    await addMember(user);
    await addMember(user, "failed@example.com");
    await submit(user);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(`failed@example.com: ${warning}`);
    expect(alert).not.toHaveTextContent(email);
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(mocks.enqueueSnackbar).not.toHaveBeenCalledWith(
      "Invite sent successfully.", expect.anything(),
    );
  });

  it("keeps the organization-only success message when no member was invited", async () => {
    respondWith([]);
    const user = userEvent.setup();
    renderSetup();
    await screen.findByDisplayValue("owner@example.com");
    await submit(user);

    await waitFor(() => expect(mocks.enqueueSnackbar).toHaveBeenCalledWith(
      "Organization Setup Successfully.", { variant: "success" },
    ));
    expect(mocks.post.mock.calls[0][1].members).toEqual([]);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("still reports member-creation failures as errors", async () => {
    mocks.post.mockRejectedValue({ result: "Error in managing users" });
    const user = userEvent.setup();
    renderSetup();
    await addMember(user);
    await submit(user);

    await waitFor(() => expect(mocks.enqueueSnackbar).toHaveBeenCalledWith(
      "Error in managing users", { variant: "error" },
    ));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(mocks.enqueueSnackbar).not.toHaveBeenCalledWith(
      "Invite sent successfully.", expect.anything(),
    );
  });
});
