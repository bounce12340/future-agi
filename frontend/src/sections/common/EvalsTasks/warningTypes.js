// Warning types an eval run can attach to EvalLogger.output_metadata.warnings.
// Keep in sync with tracer/views/eval_task.py.

export const PARTIAL_INPUT_WARNING_TYPE = "partial_input";
export const GROUND_TRUTH_NOT_APPLIED_WARNING_TYPE = "ground_truth_not_applied";

export const WARNING_TYPE_LABELS = {
  [PARTIAL_INPUT_WARNING_TYPE]: "Partial inputs",
  [GROUND_TRUTH_NOT_APPLIED_WARNING_TYPE]: "Ground Truth not applied",
};

export const WARNING_TYPE_FALLBACK_MESSAGES = {
  [PARTIAL_INPUT_WARNING_TYPE]:
    "Eval ran with some inputs empty. Result may be less reliable. Ignore if this is intentional.",
};

export const warningTypeLabel = (type) =>
  WARNING_TYPE_LABELS[type] || "Warning";

export const warningMessage = (warning) =>
  warning?.message || WARNING_TYPE_FALLBACK_MESSAGES[warning?.type] || "";
