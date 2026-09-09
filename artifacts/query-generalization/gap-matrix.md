# G1 gap matrix

Categories: **existing data**, **missing declaration**, **missing binding**, **missing data**, **unexpressible**.
No guessed containment edges were added. Counts are metadata from `/app/data` on the serving host.

| Capability | Physical schema | Semantic V1 (active) | Semantic V2 file (inactive) | Production data (aggregates) | Gap |
|---|---|---|---|---|---|
| Household members | `lives_in` person→address, inverse `has_resident`; temporal; `household_role`, `residence_type`, `start`/`end` | `member` / `residence` | same | 5/8 persons have `lives_in`; roles owner=2, minor_dependent=2, adult_dependent=1 | **existing data**. 3 persons have no current residence edge (**missing data** for member-set completeness). |
| Kinship parent/child | `parent_of` person→person, inverse `child_of`; extra physical field `type` | `parent`/`child` + gendered concepts | same | 8 edges | **existing data**. Physical `type` is advertised as relation property `type` with **no ontology declaration** beyond catalog inference (V1 open vocabulary). |
| Spouse | `spouse_of` symmetric temporal | `spouse`/`wife`/`husband` | same | 1 edge | **existing data** |
| Adult / minor | age fallback on `dob`; optional `household_role` match | predicates advertised; conjunction **accepted** | `disjoint_with` rejects conjunction | 8/8 persons have `dob`; 5 residents have roles | **existing data** for age fallback. Serving V1 does **not** reject `adult AND minor`. |
| Gender filter | person.gender | property `gender`, open string on V1 | closed male/female domain | 4 male / 4 female | **existing data**. V1 cannot reject non-domain literals. |
| Person names | `name`, `first_name`, `last_name`, `address_as` | display/given/family/form_of_address | typed unions | 8 unique names | **existing data** |
| Address / home site | address node; `address`, `address_type` | `full_address`, `display_name`; `residence`→address | same | 1 home address | **existing data** |
| Item location | `located_in` item→address\|space; `unique_from` | **no base relation** | **no base relation** | 7/10 items have a location (1 at address, 6 at space) | **missing declaration** + **missing binding**. 3 items have **missing data** (no `located_in`). Inverse not declared. |
| Space hosted by item | `hosted_by` space→item; inverse `hosts_space`; `unique_from` | **no base relation** | **no base relation** | 13/13 spaces have a host; 2 host items | **missing declaration** + **missing binding**. Physical inverse exists; not advertised. |
| Room-to-home containment | not a single edge. Documented composition: space `hosted_by` item(house) and that item `located_in` address | unexpressible: no `hosted_by`/`located_in` relations, no multi-hop room concept | same | 10/13 spaces sit on a host item that is located at an address; 3 spaces sit on a host item that is **not** address-located (storage interiors) | **existing data** for a two-hop path. **unexpressible** in current IR. Do not invent space→address. |
| Room / product category | `space_type` (room, outdoor_space, storage); `item_type` (house, refrigerator, food, …) | catalog exposes `space_type`/`item_type` as V1 semantic properties of space/item | V2 only if declared+bound; V2 candidate has **no** space/item location relations and **no** room concept | types populated | Category **fields exist**. There is **no** collection concept “rooms of this home” or “food items”. Filtering `item_type=food` still requires a traversal that starts at a legal root (**missing declaration** of item/space collections). |
| Named item in household scope | alias scan is table-wide; household_id only scopes appellations | `named_entity` | same | 0 within-table name collisions in current data | **missing binding** of household scope to alias lookup. Collision test requires synthetic same-named items (G2), not extra private records. |
| Collections >25 | retrieval `limit` / `MAX_TOOL_RECORDS=25` | count/select have no incompleteness status | same | current household member count is 5 (<25) | Truncation is **unexpressible as incomplete**; would look like an exact total. Current production size does not exhibit it. G2 uses a synthetic 30-member graph. |

## Unexpressible operations (current IR + active V1)

1. “Where is this item?” as a declared location hop.
2. “Which rooms are in this home?” as home→room containment.
3. “How many rooms?” without misusing `member`.
4. “Food/milk as a category collection” as a typed set rooted at the household.
5. Completeness-aware count when the resolver truncates at 25.
6. Rejecting `adult AND minor` on the **serving** process (V2 file does this only if loaded).

Codex owns whether to add `located_in` / `hosted_by` as base relations, a room concept, scoped named lookup, and completeness status. This matrix does not invent those declarations.
