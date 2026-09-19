# Offer catalog

Maple Telecom uses four retention actions. Every drafted action is reviewed by a person before anything is sent.

## Loyalty discount
offer_id: LOYALTY_DISCOUNT. A percentage discount on monthly charges for a limited number of months.
- Eligible: month-to-month or one-year contract, at least 4 months of tenure, marketing consent on file, and fewer than 3 outreach contacts in the last 90 days.
- Standard terms: 10% off for up to 6 months. Limits by contract and risk tier are in the discount caps document.
- Use it when no service-fit offer applies; the segment playbooks give the preferred order.
- Not available on two-year contracts or to customers in their first 3 months.

## Term upgrade
offer_id: TERM_UPGRADE. A discount in exchange for moving from a month-to-month contract to a 12-month term.
- Eligible: month-to-month contract, at least 4 months of tenure, marketing consent on file.
- Fixed terms: 10% off for 12 months (discount_pct 10, duration_months 12). The message must say the offer applies when the customer moves to a 12-month term.
- Not available on one-year or two-year contracts.

## Tech support trial
offer_id: TECH_SUPPORT_TRIAL. The Tech Support add-on at no charge for 3 months.
- Eligible: the customer has an internet service (DSL or Fiber optic), does not currently have Tech Support, and has given marketing consent.
- Fixed terms: discount_pct 0, duration_months 3. The word "free" may be used for this offer only.
- Not available to phone-only customers or to customers who already have Tech Support.

## Service check-in
offer_id: SERVICE_CHECKIN. A friendly, non-promotional invitation to talk to a service adviser about the customer's plan.
- Eligible: any customer while the contact limit has not been reached. It does not need marketing consent because it is a service message.
- Fixed terms: discount_pct 0, duration_months 0. The message contains no numbers and no promotional language.
- Use it when no other offer is eligible, or when the segment playbook lists it as the best fit.
