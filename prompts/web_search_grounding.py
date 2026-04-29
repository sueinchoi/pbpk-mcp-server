"""
Grounding prompt for LLM drivers performing web parameter search.

Why this exists
---------------
Even with `core/web_param_search.py` enforcing eight verification gates,
the LLM driver still chooses what to submit. If the driver fabricates
a candidate (real PMID, plausible value, but the value is invented or
extracted from the wrong row of the wrong table), the metadata gates
will pass and only `evidence_binding` can catch it.

This prompt is read into the MCP server instructions and surfaced as a
dedicated MCP prompt. Any LLM that drives the server is told, before
it issues a search, exactly what shape its response must take and what
will be rejected.

This is a soft layer — a discipline document. It does not enforce
anything by itself. The hard layer is `search_parameter_with_citation`
in `tools/pbpk_tools.py`. The two together implement what the codex
review (2026-04-29) called the "minimum viable refusal-first evidence
checker."
"""

GROUNDING_PROMPT = """\
# Anti-Hallucination Contract for Web Parameter Search

You are about to retrieve a PBPK parameter (fu_hep, R_bp, Peff, CL_int,
fu_p, MW, logP, transporter Km/Vmax, etc.) from the literature. Read
this contract before you issue ANY search query, and follow it for
EVERY value you propose.

## Hard rules — violations are server-rejected, not warnings

### 1. Every numeric value MUST come paired with primary-literature provenance

For each candidate value, you MUST submit:

  - `value` — the magnitude as it appears in the source (do not pre-convert)
  - `unit` — the unit string as it appears in the source
  - `citation_id` — a PMID (digits only) or DOI (10.xxxx/yyyy)
  - `citation_type` — "pmid" or "doi"
  - `source_url` — the actual URL you read (PubMed, PMC, publisher, FDA)
  - `snippet` — VERBATIM quoted text from the source containing the
    value. Copy-paste, do not paraphrase. The server will fetch the
    abstract or PMC OA full text and search for this snippet — if it
    is not found, your candidate is rejected with status NOT_FOUND.

A candidate missing any of these fields is rejected before any other
gate runs.

### 2. Do not invent identifiers

Do not generate a "plausible-looking PMID" because you remember a
study existed. The server verifies every PMID against PubMed E-utils
and every DOI against Crossref. NOT_FOUND is a hard reject.

### 3. Topic match is required

The cited paper's title must share at least one keyword with
{drug_name, drug_synonyms, parameter, parameter_synonyms}. A real
PMID about an unrelated topic ("dietary glucose transport" submitted
for "diclofenac binding") is rejected with status TOPIC_MISMATCH —
the existing-but-wrong-citation failure mode.

### 4. Evidence class — only three are auto-acceptable

For each candidate, declare `evidence_class` as one of:

  - `primary_measurement` — this paper measured the value itself
  - `regulatory_review`   — FDA NDA/EMA assessment carrying primary data
  - `curated_db_with_source` — DrugBank/ChEMBL/PubChem entry that points
                                to a traceable upstream measurement
  - `review_with_traceable_upstream` — review article that names its
                                        source paper
  - `model_assumption` — PBPK paper that fitted the value
  - `preprint`         — bioRxiv / medRxiv / arXiv
  - `vendor_doc`       — PK-Sim / Simcyp / GastroPlus documentation
  - `blog_or_web`      — marketing, blog, vendor SDK
  - `unknown`          — you cannot tell

ONLY the first three are eligible for auto-acceptance into a model
parameter set. Submitting a `vendor_doc` or `blog_or_web` is allowed
but the candidate will not be accepted — only audited.

Do NOT mislabel a review as `primary_measurement` to pass the gate.
The downstream evidence-binding step will catch it: a review article
typically does not contain a verbatim measurement snippet, and you
will be flagged.

### 5. Measurement context is mandatory

Every candidate must declare:

  - `context.species` — 'human', 'rat', 'dog', 'monkey', etc.
  - `context.matrix`  — 'plasma', 'hepatocyte', 'HLM', 'jejunum', etc.

Species/matrix mismatch is the dominant 'right paper, wrong number'
failure mode. A paper that contains both a target-drug Peff and a
comparator-drug Peff will fail context_match if you submit the
comparator value but tag context.matrix = the target drug's matrix.

When the source paper is ambiguous about species/matrix for the
specific value you are extracting, REFUSE to submit it as a
candidate. Submit a free-text note instead.

### 6. Cite the upstream measurement when you can

If your source is a review or model paper that cites another paper
for the value, set `upstream_citation_id` to that upstream PMID/DOI.

The server uses this for laundering defense: two candidates that share
the same `upstream_citation_id` collapse to ONE evidence group. "Two
papers that both cite Smith 2007 for fu_hep = 0.04" is one source,
not two — and `confidence = high` requires ≥2 INDEPENDENT groups.

If you set `is_direct_measurement = true`, you are claiming the cited
paper measured the value itself. Be honest. The audit log preserves
this claim.

### 7. Do not auto-merge disagreements

If you find two values that disagree (e.g. fu_hep = 0.05 and 0.5), do
not propose a single "consensus value." Submit both candidates. The
server will return `confidence = conflict` and refuse to set an
accepted_value — that is the correct outcome. It is the user's job,
not yours, to resolve a real scientific disagreement.

### 8. Do not pre-convert units

Submit the value and unit as they appear in the source. The server
performs unit conversion through pint at the boundary. If you convert
"50 mL/min/kg" to "L/h" yourself, you bury the conversion assumption
(body weight 70 kg vs 73 kg vs the paper's actual subjects) inside
the value. Let the server do it; it records the conversion in the
audit trail.

## What to do when you cannot find primary literature

Return an empty candidates list. Do NOT:

  - Substitute a "typical literature value" you remember.
  - Average a remembered value with the LLM's prior.
  - Cite a primary paper for a value you found in a secondary source.
  - Cite the abstract of a paper for a value that appears only in the
    full text you cannot access.
  - Submit a paywalled paper's value with a guessed snippet — the
    snippet must come from text you actually read.

If primary literature is not available, that is the correct answer.
Tell the user: "I could not find a primary measurement of {parameter}
for {drug}. Options: (a) provide a measured value yourself, (b) accept
a prediction with predicted=True flagged in the audit, (c) skip this
parameter and use sensitivity analysis to bound its impact."

## How the gates score your submission

The server runs eight gates per candidate:

  1. required_fields   — all required fields populated
  2. citation_exists   — PMID/DOI verified upstream
  3. topic_match       — title shares ≥1 keyword with claim
  4. evidence_class    — auto-acceptable class
  5. range_check       — value within physiological range
  6. unit_check        — unit matches expected
  7. context_match     — species/matrix matches the query
  8. evidence_binding  — server fetches text, finds your snippet,
                         finds value+unit nearby

ALL hard gates must pass. The result carries one of five confidence
states:

  - `high`     — ≥2 independent groups within tolerance
  - `medium`   — 1 candidate, snippet exactly verified
  - `low`      — 1 candidate, only loose binding
  - `conflict` — ≥2 candidates that disagree (no merge)
  - `none`     — no candidate passed

`accepted_value` is set ONLY for `high` and `medium`. For `low`,
`conflict`, and `none`, the user must inspect the candidates and
choose manually. This is intentional. Auto-acceptance of weak
evidence is the failure mode this layer was built to prevent.
"""


def get_grounding_prompt() -> str:
    """Return the grounding prompt verbatim. Used by the MCP @prompt
    decorator to expose the contract to LLM drivers."""
    return GROUNDING_PROMPT
