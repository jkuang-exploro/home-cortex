# Data

Node names are ordered lists of aliases for the same entity. Store the English
name first, then add names in other languages as needed:

```json
{
  "id": "person:example",
  "name": ["English Name", "中文名"]
}
```

Do not use separate records for translations of the same person's or
location's name. Record IDs remain language-neutral and stable.
