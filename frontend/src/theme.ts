// Entity-type colors: dark-mode categorical steps in fixed order, validated against the
// graph surface (#0e0f1a) with the dataviz palette validator. Identity is never color-only:
// the legend, tooltips and labels always name the type.
export const TYPE_COLORS: Record<string, string> = {
  PERSON: "#3987e5",
  ORGANIZATION: "#d95926",
  LOCATION: "#199e70",
  EVENT: "#c98500",
  WORK: "#d55181",
  CONCEPT: "#9085e9",
  UNKNOWN: "#898781",
};

export const TYPE_LABELS: Record<string, string> = {
  PERSON: "Person",
  ORGANIZATION: "Organization",
  LOCATION: "Location",
  EVENT: "Event",
  WORK: "Work",
  CONCEPT: "Concept",
  UNKNOWN: "Other",
};

export const typeColor = (type: string) => TYPE_COLORS[type] ?? TYPE_COLORS.UNKNOWN;

export const MODE_LABELS: Record<string, string> = {
  qasa: "QASA",
  uniform: "Uniform",
  naive: "Vector",
  overview: "Overview",
  none: "No retrieval",
};

export const MODE_HINTS: Record<string, string> = {
  qasa: "Query-aware spreading activation: flow is weighted by similarity to the question",
  uniform: "Same traversal, but activation flows equally to every neighbour (ablation)",
  naive: "Plain vector search over entity descriptions, no graph traversal (baseline)",
};
