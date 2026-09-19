# Agent evaluation

**Final evaluation.** These customers were never used to tune prompts (customers whose drafts were inspected during development are excluded).

Model: `granite4:micro` (temperature 0 for first drafts). 48 held-out customers who pass the expected-value gate, stratified to over-represent edge cases (no marketing consent, contact limit reached, new customers, high risk, high value). Every configuration sees the same customers.

**Scenario mix** (a customer can carry several tags): contact_limit 4, high_risk_m2m 16, high_value 9, loyalty_preferred 2, m2m_dsl 9, new_customer 16, one_year 7, opt_out 7, phone_only 4

*Compliant* = no policy violation from the deterministic verifier, ignoring citation rules. Compliance of a verifier-loop configuration's **final** draft is high by construction, which is why first-draft compliance (the effect of the prompt and the retrieved policy) is reported separately. The mandatory opt-out line is appended by the system when the model omits it (mandatory disclosures should not depend on a model). 'Offer matches playbook' compares the final offer with the first eligible entry in the segment's preferred order. CI = 95% Wilson interval.

## Results
| config | first draft compliant | final compliant | escalated | offer matches playbook | prompt tokens | LLM calls | LLM time / run |
|---|---|---|---|---|---|---|---|
| none | 1/48 (2%; 0-11) | 1/48 (2%; 0-11) | 0 | 23% | 420 | 1.00 | 2.8s |
| full | 26/48 (54%; 40-67) | 26/48 (54%; 40-67) | 0 | 23% | 2810 | 1.00 | 3.6s |
| rag | 27/48 (56%; 42-69) | 27/48 (56%; 42-69) | 0 | 42% | 1582 | 1.00 | 3.3s |
| rag+verify | 28/48 (58%; 44-71) | 48/48 (100%; 93-100) | 3 | 65% | 3290 | 1.94 | 5.7s |

## Paired comparisons
| contrast (paired, same scenarios) | only first succeeds | only second succeeds | exact McNemar p |
|---|---|---|---|
| first draft: full policy vs no policy | 25 | 0 | < 0.0001 |
| first draft: RAG vs no policy | 26 | 0 | < 0.0001 |
| first draft: RAG vs full policy | 6 | 5 | 1.0000 |
| offer matches playbook: full policy vs no policy | 10 | 10 | 1.0000 |
| offer matches playbook: RAG vs no policy | 17 | 8 | 0.1078 |
| offer matches playbook: RAG vs full policy | 14 | 5 | 0.0636 |
| offer matches playbook: rag+verify vs rag | 11 | 0 | 0.0010 |
| verifier loop on rag: final vs first draft | 20 | 0 | < 0.0001 |

## Which rules the first drafts break
| first-draft violation (share of scenarios) | none | full | rag | rag+verify |
|---|---|---|---|---|
| consent | 15% | 4% | 0% | 0% |
| contact_limit | 8% | 8% | 8% | 8% |
| discount_cap | 15% | 0% | 0% | 0% |
| eligibility | 31% | 23% | 21% | 21% |
| no_action | 0% | 0% | 4% | 2% |
| numbers | 71% | 0% | 12% | 12% |
| terms | 77% | 17% | 6% | 6% |

## Grounding and retrieval
- full: valid citations on 0/48 (0%; 0-7) of first drafts
- rag: valid citations on 20/48 (42%; 29-56) of first drafts
- rag+verify: valid citations on 20/48 (42%; 29-56) of first drafts
- rag: mean context recall 1.00
- rag+verify: mean context recall 1.00

![compliance](figures/agent_eval_granite4-micro.png)

## Example repairs (first draft -> verifier feedback -> final)
**6178-KFNHS** (rag+verify)
- first-draft violations: [contact_limit] customer already had 3 contacts in 90 days (limit 3); the action must be NO_ACTION: offer_id NONE, discount_pct 0, duration_months 0 and an empty message
- final message: 

**5609-CEBID** (rag+verify)
- first-draft violations: [contact_limit] customer already had 3 contacts in 90 days (limit 3); the action must be NO_ACTION: offer_id NONE, discount_pct 0, duration_months 0 and an empty message; [citations] cite the catalog section for the chosen offer: offer_catalog#loyalty-discount
- final message: 

**7486-KSRVI** (rag+verify)
- first-draft violations: [citations] cite at least one policy section id (for example offer_catalog#loyalty-discount); [citations] cite the catalog section for the chosen offer: offer_catalog#service-check-in
- final message: Hi there! We're here to check in and see how your service is going. Let us know if you have any questions or need assistance.
