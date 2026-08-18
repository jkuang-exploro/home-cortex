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

Do not use separate records for translations of the same person's or
location's name. Record IDs remain language-neutral and stable.
