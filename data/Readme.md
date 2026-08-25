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

Recurring date semantics live in `schemas/memorable_dates.yaml`. The registry
maps a localized concept such as `birthday` or `wedding_anniversary` to its
authoritative node/edge field and recurrence rule. Keep the actual date only in
that source field: do not copy `person.dob` or `spouse_of.start` into a second
"memorable dates" data file. To introduce another recurring date, add its
aliases, localized label, recurrence, and source mapping to the registry.

Household status is contextual and belongs on the edge connecting a Person to
a household. For a resident, add `household_role` to `lives_in`:

```json
{
  "from": "person:example",
  "to": "address:example_home",
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
Every registered canonical relationship must have a matching JSON file. Use
`[]` when that relationship currently has no facts; deleting the file is an
invalid source configuration.

Optional `start` and `end` are the interval for a temporal relation only:

- `spouse_of.start` is the marriage date (结婚纪念日).
- `lives_in.start` is when the person began living at that address.

`end: null` means the relationship is current. Do not reuse a `start` date
from one relation as a fact about another.

Store a symmetric `spouse_of` fact once; traversal works from either spouse.
Store only canonical `parent_of` facts. `child_of` is a derived inverse query
name and must not have its own data file.

An addressable home site is an `address`; the physical house on that site is a
tracked `item`. Place the house Item at the Address with `located_in`:

```json
{
  "from": "item:example_house",
  "to": "address:example_home"
}
```

Named rooms and outdoor areas belonging to the house are `space` nodes hosted
by that house Item:

```json
{
  "from": "space:kitchen",
  "to": "item:example_house"
}
```

`space_type` is `room` or `storage`. Structural membership is modeled solely
through `hosted_by`. Every modeled Space has one explicit edge to the physical
Item that provides it.

An Item is a physical entity tracked as an independent identity unit. Its
current position uses `located_in` (`item` → `address` or `space`). A Space
provided or defined by an Item uses `hosted_by` (`space` → `item`). This applies
equally to house rooms and to container regions such as a refrigerator
interior. `hosts_space` is derived by reverse traversal and must not have its
own JSON file. Hosted spaces are always explicit; never generate them from
`item_type`. Items hosting zero, one, or many spaces are all valid.

Record keys may contain non-empty colon-delimited segments, so a nested space
may use `space:home:kitchen:fridge_01:interior`. Keep the table name before the
first colon and use only letters, digits, `_`, or `-` within each segment.

People live at an `address`. Do not attach `lives_in` to a space.

Do not use separate records for translations of the same person's, address's,
space's, or item's name. Record IDs remain language-neutral and stable.
