# Validation Rules

## Categories

- `syntactic`: field-local format checks such as CPF/CNPJ, date parsing, percentage parsing, service code shape, email, phone, UF, and monetary parsing.
- `semantic`: plausibility checks on otherwise parseable values, such as future issue dates, very old dates, and negative monetary values.
- `relational`: cross-field checks, mainly net amount consistency against gross amount, discounts, and retentions, plus ISS amount consistency against taxable amount and ISS rate.
- `resolution-state`: unresolved required fields, explicit `missing` fields, and `conflict` outputs coming from the resolver.

## Critical Fields

Current critical fields are:

- `nfse_number`
- `verification_code`
- `issue_date`
- `provider_name`
- `provider_document`
- `recipient_name`
- `service_description`
- `gross_amount`
- `net_amount`

These are expected to be present and safely resolved before downstream comparison or acceptance.

## Blocking Issues

Treat these as blocking by default:

- missing required field
- unresolved required field
- `conflict` on a required field
- invalid required CPF/CNPJ
- invalid required date
- invalid required monetary value
- negative required monetary value
- inconsistent `net_amount` relative to gross amount, discounts, and retentions

## Warning-Level Issues

Keep these as warnings unless a stricter policy is introduced:

- invalid optional CPF/CNPJ
- invalid email
- invalid phone
- invalid UF
- invalid or implausible percentage
- invalid service code
- future issue date
- suspiciously old issue date
- inconsistent ISS amount relative to taxable amount and ISS rate
- provider and recipient documents being equal
- `missing` or `conflict` on non-critical optional fields

## Assumptions And Limits

- Validation runs on `ResolvedField` outputs, not raw OCR tokens.
- No value is inferred during validation; the validator only checks supplied values.
- Monetary consistency uses a tolerance of `0.02` to avoid false failures from rounding.
- Date parsing currently accepts `DD/MM/YYYY`, `YYYY-MM-DD`, and `MM/YYYY`.
- Email, phone, service code, and UF validation are intentionally lightweight and implementation-oriented, not exhaustive regulatory validation.
- Validation severity is driven by field criticality and rule type, not by source engine.
