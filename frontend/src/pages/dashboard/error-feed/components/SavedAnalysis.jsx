import React from "react";
import PropTypes from "prop-types";
import { Box, Stack, Typography } from "@mui/material";
import { CONFIDENCE } from "../constants";

const CONFIDENCE_LABEL = {
  [CONFIDENCE.HIGH]: "High confidence",
  [CONFIDENCE.MEDIUM]: "Medium confidence",
  [CONFIDENCE.LOW]: "Low confidence",
};

// The last saved analysis of an issue, read-only. A shared issue shows this in
// place of the live Fix tab: re-runs and follow-ups would start new runs.
export default function SavedAnalysis({ rca }) {
  if (!rca?.synthesis) {
    return (
      <Stack alignItems="center" justifyContent="center" sx={{ py: 8 }}>
        <Typography fontSize="13px" color="text.disabled">
          No analysis saved for this issue.
        </Typography>
      </Stack>
    );
  }
  return (
    <Stack gap={1.5} sx={{ p: 2, maxWidth: 820 }}>
      <Typography fontSize="11px" fontWeight={600} color="text.disabled">
        {CONFIDENCE_LABEL[rca.confidence] ?? "Saved analysis"}
      </Typography>
      <Typography
        fontSize="14px"
        fontWeight={500}
        color="text.primary"
        sx={{ lineHeight: 1.55 }}
      >
        {rca.synthesis}
      </Typography>
      {rca.fix && (
        <Box>
          <Typography fontSize="11px" fontWeight={600} color="text.disabled">
            Suggested fix
          </Typography>
          <Typography
            fontSize="13px"
            color="text.secondary"
            sx={{ lineHeight: 1.6 }}
          >
            {rca.fix}
          </Typography>
        </Box>
      )}
    </Stack>
  );
}

SavedAnalysis.propTypes = {
  rca: PropTypes.shape({
    synthesis: PropTypes.string,
    fix: PropTypes.string,
    confidence: PropTypes.string,
  }),
};
