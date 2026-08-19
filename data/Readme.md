# Data

Node names may use an ordered alias list. Store the English name first, then
add names in other languages as needed:

```json
{
  "id": "person:example",
  "name": ["English Name", "中文名"]
}
```

Explicit localized objects are also supported:

```json
{
  "id": "person:example",
  "name": {"en": "English Name", "zh": "中文名"}
}
```

Person records may optionally specify how household agents should address the
person. `address_as` is a presentation preference, not another name or alias:

```json
{
  "id": "person:example",
  "name": {"en": "Jian Kuang", "zh": "匡健"},
  "address_as": {"en": "Mr. Kuang", "zh": "先生"}
}
```

Never infer `address_as` from age, gender, or relationships. Store it only when
the household has explicitly chosen the preference.

Person records may store `dob` as an ISO date. That field is the date of
birth. Do not invent a birthday when `dob` is absent.

Household status is contextual and belongs on the edge connecting a Person to
a household. For a resident, add `household_role` to `lives_in`:

```json
{
  "from": "person:example",
  "to": "location:example_home",
  "residence_type": "primary",
  "household_role": "owner"
}
```

The V1 reception roles are `owner`, `minor_dependent`, `adult_dependent`, and
`guest`. A guest should use a suitable relationship edge rather than
`is_guest` or `person_type: guest` on the Person node. Missing, conflicting, or
unrecognized roles resolve to the neutral `unknown` reception policy.

Relationship files under `edges/` are named for a registered relationship in
`schemas/edge/`. Schema files define endpoint types, direction, symmetry, and
whether temporal fields are allowed. Data files contain facts only.

Optional `start` and `end` are the interval for a temporal relation only:

- `spouse_of.start` is the marriage date (结婚纪念日).
- `lives_in.start` is when the person began living at that location.

`end: null` means the relationship is current. Do not reuse a `start` date
from one relation as a fact about another.

Store a symmetric `spouse_of` fact once; traversal works from either spouse.
Store only canonical `parent_of` facts. `child_of` is a derived inverse query
name and must not have its own data file.

Do not use separate records for translations of the same person's or
location's name. Record IDs remain language-neutral and stable.
