# Model comparison (final evaluation, 48 held-out customers)

Same customers, prompts and policy for every model. CI = 95% Wilson interval in parentheses.

| model | config | first draft compliant | final compliant | offer matches playbook | prompt tokens | LLM calls | LLM time / run |
|---|---|---|---|---|---|---|---|
| granite4-micro | none | 1/48 (2%; 0-11) | 1/48 (2%; 0-11) | 23% | 420 | 1.00 | 2.8s |
| granite4-micro | full | 26/48 (54%; 40-67) | 26/48 (54%; 40-67) | 23% | 2810 | 1.00 | 3.6s |
| granite4-micro | rag | 27/48 (56%; 42-69) | 27/48 (56%; 42-69) | 42% | 1582 | 1.00 | 3.3s |
| granite4-micro | rag+verify | 28/48 (58%; 44-71) | 48/48 (100%; 93-100) | 65% | 3290 | 1.94 | 5.7s |
| granite4-small-h | none | 20/48 (42%; 29-56) | 20/48 (42%; 29-56) | 4% | 420 | 1.00 | 6.9s |
| granite4-small-h | full | 34/48 (71%; 57-82) | 34/48 (71%; 57-82) | 65% | 2810 | 1.00 | 9.2s |
| granite4-small-h | rag | 33/48 (69%; 55-80) | 33/48 (69%; 55-80) | 71% | 1582 | 1.00 | 9.8s |
| granite4-small-h | rag+verify | 30/48 (62%; 48-75) | 47/48 (98%; 89-100) | 96% | 2609 | 1.54 | 13.5s |
