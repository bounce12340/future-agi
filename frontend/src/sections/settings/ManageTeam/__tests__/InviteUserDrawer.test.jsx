/* eslint-disable react/prop-types */
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, userEvent, waitFor } from "src/utils/test-utils";
import InviteUserDrawer from "../InviteUserDrawer";

const mocks = vi.hoisted(() => ({ post: vi.fn(), enqueueSnackbar: vi.fn() }));
vi.mock("src/utils/axios", () => ({
  default: { post: mocks.post },
  endpoints: { settings: { teams: { inviteMember: "/accounts/team/users/" } } },
}));
vi.mock("src/components/snackbar", () => ({
  useSnackbar: () => ({ enqueueSnackbar: mocks.enqueueSnackbar }),
}));
vi.mock("src/components/iconify", () => ({ default: () => null }));
vi.mock("src/components/FormTextField/FormTextFieldV2", async () => {
  const { Controller } = await import("react-hook-form");
  return { default: ({ control, fieldName, label }) => (
    <Controller control={control} name={fieldName} render={({ field }) => (
      <label>{label}<input {...field} value={field.value ?? ""} /></label>
    )} />
  ) };
});
vi.mock("src/components/FormSelectField", async () => {
  const { Controller } = await import("react-hook-form");
  return { FormSelectField: ({ control, fieldName, label, options }) => (
    <Controller control={control} name={fieldName} render={({ field }) => (
      <label>{label}
        <select {...field}>
          <option value="">Select a role</option>
          {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </label>
    )} />
  ) };
});

const email = "newhire@example.com";
const member = { id: "member-1", email };
const warning = "Member added, but the invitation email could not be sent. Open Members to resend the invite or copy an invite link if available.";

async function submitInvite() {
  const onClose = vi.fn();
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  const invalidate = vi.spyOn(client, "invalidateQueries");
  render(
    <QueryClientProvider client={client}>
      <InviteUserDrawer open onClose={onClose} />
    </QueryClientProvider>,
  );
  const user = userEvent.setup();
  await user.type(screen.getByRole("textbox", { name: "User Email" }), email);
  await user.selectOptions(screen.getByRole("combobox", { name: "User Role" }), "Member");
  await user.click(screen.getByRole("button", { name: "Invite User", exact: true }));
  return { onClose, invalidate };
}

describe("Manage Team invite delivery", () => {
  beforeEach(() => vi.clearAllMocks());

  it("posts the current member payload and reports healthy delivery", async () => {
    mocks.post.mockResolvedValue({ data: { result: { created_members: [member] } } });
    const { onClose, invalidate } = await submitInvite();

    await waitFor(() => expect(mocks.enqueueSnackbar).toHaveBeenCalledWith(
      "Successfully invited 1 user(s)", { variant: "success" },
    ));
    expect(mocks.post).toHaveBeenCalledWith("/accounts/team/users/", {
      members: [{ email, name: "Newhire", organization_role: "Member" }],
    });
    expect(onClose).toHaveBeenCalledOnce();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["member-list"], type: "all" });
  });

  it("warns instead of claiming delivery and still refreshes persisted members", async () => {
    mocks.post.mockResolvedValue({ data: { result: {
      created_members: [{ ...member, invite_email_warning: warning }],
    } } });
    const { onClose, invalidate } = await submitInvite();

    await waitFor(() => expect(mocks.enqueueSnackbar).toHaveBeenCalledWith(
      `${email}: ${warning}`, { variant: "warning", persist: true },
    ));
    expect(mocks.enqueueSnackbar).not.toHaveBeenCalledWith(
      expect.anything(), { variant: "success" },
    );
    expect(onClose).toHaveBeenCalledOnce();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["member-list"], type: "all" });
  });

  it("does not close or claim success when member creation fails", async () => {
    mocks.post.mockRejectedValue({ message: "Error in managing users" });
    const { onClose, invalidate } = await submitInvite();

    await waitFor(() => expect(mocks.enqueueSnackbar).toHaveBeenCalledWith(
      "Error in managing users", { variant: "error" },
    ));
    expect(onClose).not.toHaveBeenCalled();
    expect(invalidate).not.toHaveBeenCalled();
  });
});
