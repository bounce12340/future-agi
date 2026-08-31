# Ground Truth, and how to supply `expected_value`

Ground Truth and `expected_value` are two different things. Reading the Ground
Truth setup screen and concluding that enabling it will fill in a missing
`expected_value` is a reasonable mistake, and a common one. It will not.

This page states what Ground Truth does, what it does not do, and how you
actually give an eval an expected value when it runs from Observe.

---

## What Ground Truth is

Ground Truth is **few-shot calibration**. You upload labelled rows, the rows are
embedded, and at eval time the most similar rows are retrieved and attached to
the judge prompt as reference examples — this is how a graded example teaches
the judge what a good answer looks like for your domain.

Concretely, `GroundTruthService.inject_context`
(`model_hub/services/ground_truth_service.py`) puts the retrieved rows on
`mapped["ground_truth_blocks"]`, and the evaluator renders them into the prompt.

That is the whole mechanism. Ground Truth:

- **does** steer the judge with retrieved, similar, human-labelled examples
- **does not** populate any required eval input
- **does not** supply `expected_value`, ever

So an eval that declares `expected_value` as a required key still needs
`expected_value` mapped. If it is not mapped, the run fails with:

```
Missing required input(s) for eval: expected_value.
Required keys: ['generated_value', 'expected_value']. Optional keys: [].
```

That error is correct. It is telling you a required input is unmapped, not that
Ground Truth is broken.

### Where Ground Truth is applied

| Surface | Ground Truth applied |
| --- | --- |
| Datasets | Yes |
| Prompt playground | Yes |
| SDK evaluations | Yes |
| Observe eval tasks (spans / traces / sessions) | **No** |

Nothing on the Observe eval path calls `inject_context`. An eval task that uses
a template with Ground Truth switched on runs **uncalibrated**.

This used to be entirely silent. It is not any more: such a run now attaches a
`ground_truth_not_applied` warning to the eval result, which surfaces on the
task's logs view alongside partial-input warnings. The run still succeeds —
Ground Truth never blocks an eval — but the state is now visible instead of
invisible.

---

## Supplying `expected_value` on the Observe path

You supply it the same way you supply any other eval input: **emit it as a span
attribute, then map `expected_value` to that attribute name.**

There is no special-casing to work around. `_process_mapping`
(`tracer/utils/eval.py`) walks every key in the mapping and resolves each one
out of the span's attributes; `expected_value` resolves exactly like `input` or
`output` does.

### Worked example

**1. Emit the expected answer as a span attribute.**

```python
with tracer.start_as_current_span("answer_question") as span:
    answer = my_agent(question)
    span.set_attribute("input.value", question)
    span.set_attribute("output.value", answer)
    span.set_attribute("expected.answer", golden_answer)
```

The attribute name is yours to choose. `expected.answer` is used here to keep
it distinct from the eval input key.

**2. Map the eval input to that attribute.**

In the eval task's configuration:

| Eval input | Span attribute |
| --- | --- |
| `generated_value` | `output.value` |
| `expected_value` | `expected.answer` |

**3. Scope the task with attribute filters.**

Filter the task down to the spans that actually carry the attribute — for
example an attribute filter on `expected.answer` being present, or on whatever
marks your evaluation runs.

> **Do not scope the task with `trace_id` or `span_id`.**
> `parsing_evaltask_filters` (`tracer/utils/eval_tasks.py`) handles
> `span_attributes_filters`, `observation_type`, `session_id`, `date_range`,
> `created_at` and `project_id`. `trace_id` and `span_id` are accepted by the
> API and stored on the task, then fall through the branch chain and are
> ignored. A task scoped that way silently evaluates a much wider set of spans
> than you asked for, which looks exactly like the mapping being broken.

### What this approach cannot do

You can only emit `expected_value` where you already know the correct answer at
emit time — a test or regression environment. In production you do not know it,
and retrieval against a labelled dataset is the only mechanism that fits that
case. Ground Truth retrieval on the Observe path is not implemented today.

---

## Summary

- Ground Truth is few-shot calibration attached to `mapped["ground_truth_blocks"]`.
- Ground Truth never supplies `expected_value`.
- Ground Truth is applied on datasets, playground and SDK; **not** on Observe.
- On Observe, supply `expected_value` as a span attribute and map to it.
- Scope such a task with attribute filters, never `trace_id` / `span_id`.
