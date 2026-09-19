# Consent and contact rules

## Marketing consent
Promotional messages (any offer with a price reduction or a trial) may only be sent to customers with marketing_opt_in = 1. If marketing_opt_in = 0, the only permitted action is the service check-in.

## Contact frequency
A customer may receive at most 3 outreach contacts in any 90 days. If contacts_last_90d is 3 or more, take no action: the action must be NO_ACTION with offer_id NONE and an empty message.

## Required opt-out line
Every promotional message must contain this exact line: "Reply STOP to opt out".

## Channel limits
SMS messages may be at most 320 characters. Email messages may be at most 900 characters. Count every character, including spaces and the opt-out line.

## Service messages
The service check-in is a service message, not a promotion. It may be sent without marketing consent, it does not need the opt-out line, and it must not use promotional words such as discount, offer, save, deal or free.
