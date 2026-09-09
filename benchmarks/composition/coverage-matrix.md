# Compositional evaluation — coverage matrix and proposed size

Ticket 4. Pair with `annotation-guide.md`. This document is the **size and
coverage proposal**. Do not generate household JSON or case YAML until Codex
approves this inventory. The listed utterances **are** the intended set, not a
sample to auto-expand.

Clock for every gold execution: `2026-09-03T12:00:00-07:00`.
Adulthood: completed years `gte 18` / minor `lt 18` (on the birthday the person
is an adult).

## 1. Proposed size

| Slice | Cases | What they are |
|---|---:|---|
| Development standalone | 42 | One canonical wording family per cell (`家里` / `我的` / `请列出`) |
| Frozen acceptance standalone | 36 | Held-out families (`本户` / `咱家` / English / paraphrase) |
| Development sequences | 8 | Multi-turn leak/restart; score last turn only |
| Frozen sequences | 8 | Same compositions, held-out wording |
| **Scored last turns** | **94** | 78 standalone + 16 sequence last turns |
| Households | 3 | alpha, beta, gamma |
| Distinct composition cells | 28 | Rows in §4 |
| Expression families | 2 | Canonical vs held-out; **not** a random split |

Codex review (`codex-approval.md`): **APPROVE WITH CONDITIONS**. Size is
**42 / 36 / 8 / 8**. E3 is merged into E8-1. Do not pad with `dev-beta-F7-1`.
This is the complete Ticket 4 set. No paraphrase generator, no extra rows
after approval without a new coverage review.

Rationale:

- Large enough to hold the required contrasts on more than one graph.
- Small enough that every gold plan can be executed and every critical
  forbidden plan can be shown non-equivalent.
- Sequences are extra **context conditions**, not extra compositions.
- Frozen is smaller than development because household transfer of a cell
  belongs in development; frozen spends budget on unseen wording.

Out of scope for this set: GPU accuracy, prompt edits, ontology edits,
executor edits, `SCORING_REVISION`, `DEFAULT_EVAL_PATH`.

## 2. File layout after approval

```
benchmarks/composition/
  annotation-guide.md
  coverage-matrix.md
  households/{alpha,beta,gamma}/{nodes,edges}/*.json
  development/{alpha,beta,gamma,sequences}.yaml
  frozen/{alpha,beta,gamma,sequences}.yaml
  frozen/MANIFEST.yaml
  fingerprints.json
tests/test_composition_eval.py
```

Standalone YAML reuses the existing **probe** shape (`version: 1`, `plans`,
`probe.cases` with `expected_status` / `expected_entity_ids` / `expected_value`).
Sequences are a sibling file with `history` on the scored turn. A **new**
loader in the composition tests reads `history`; `load_probe_dataset` and
`load_semantic_eval_cases` stay unchanged. Extra keys on probe cases remain
ignored by the existing loader.

## 3. Households (invented; not `data/`)

Wrong paths must change **person or count**. Count collisions are allowed only
when the case asserts `expected_entity_ids` and a paired count cell still
differs. Names are unique across the three graphs.

### 3.1 alpha — male speaker, FIL ≠ father

Address `address:alpha`. Speaker `person:alpha_self`.

| id | name | gender | dob | in household | notes |
|---|---|---|---|---|---|
| person:alpha_self | 安乔 | male | 1980-01-15 | yes | speaker, age 46 |
| person:alpha_wife | 安梅 | female | 1982-04-20 | yes | spouse |
| person:alpha_son_adult | 安晨 | male | 2006-03-10 | yes | child, age 20 |
| person:alpha_son_minor | 安野 | male | 2014-07-22 | yes | child, age 12 |
| person:alpha_daughter | 安秋 | female | 2012-11-05 | yes | child, age 13 |
| person:alpha_father | 安石 | male | 1952-08-08 | yes | speaker's father |
| person:alpha_mother | 安兰 | female | 1954-09-09 | yes | speaker's mother |
| person:alpha_fil | 梅山 | male | 1950-02-14 | yes | wife's father |
| person:alpha_mil | 梅竹 | female | 1953-06-18 | yes | wife's mother |
| person:alpha_niece | 安禾 | female | 2015-05-30 | yes | sister's daughter; sister not resident |
| person:alpha_sister | 安芳 | female | 1985-08-08 | **no** | parent of niece |

Marriage `spouse_of` self↔wife from `2004-06-18`. Parents of the three
children: both self and wife. Parent of niece: sister only.

| Query meaning | Gold set / value | Forbidden that must differ |
|---|---|---|
| Household members | 10 people | `self → member` unsupported |
| Female | 5 (wife, daughter, mother, mil, niece) | adult=7, minor=3 |
| Male | 5 | adult=7 |
| Adult | 7 | female=5 |
| Minor | 3 (son_minor, daughter, niece) | my children=3 **different ids** |
| Adult female | 3 (wife, mother, mil) | female=5, adult=7 |
| Minor female | 2 (daughter, niece) | my daughters=1 |
| My children | 3 (adult son, minor son, daughter) | household minors ids |
| My daughters | 1 (安秋) | minor female=2 |
| My sons | 2 | household minor males=1 |
| Father-in-law / wife's father | 梅山 | father 安石; spouse then FIL 安石 |
| Spouse's father-in-law | 安石 | FIL 梅山 |
| Speaker's father | 安石 | FIL 梅山 |
| Adult daughters of self | count 0 | my daughters=1 |
| Wife birth_date | 1982-04-20 | annual_occurrence days |

### 3.2 beta — female speaker, inverted ages/genders

Address `address:beta`. Speaker `person:beta_self`. FIL id and adult-female
count both differ from alpha.

| id | name | gender | dob | in household | notes |
|---|---|---|---|---|---|
| person:beta_self | 林夏 | female | 1983-02-11 | yes | speaker, age 43 |
| person:beta_husband | 林海 | male | 1981-07-19 | yes | spouse |
| person:beta_dau_adult | 林晴 | female | 2005-12-01 | yes | child, age 20 |
| person:beta_dau_minor | 林露 | female | 2013-08-16 | yes | child, age 13 |
| person:beta_son_minor | 林川 | male | 2016-02-02 | yes | child, age 10 |
| person:beta_father | 夏松 | male | 1955-03-03 | yes | speaker's father |
| person:beta_mother | 夏荷 | female | 1958-11-11 | yes | speaker's mother |
| person:beta_fil | 海峰 | male | 1949-12-25 | yes | husband's father |
| person:beta_mil | 海萍 | female | 1951-04-04 | yes | husband's mother |
| person:beta_nephew | 林岩 | male | 2018-06-06 | yes | brother's son; brother not resident |
| person:beta_brother | 夏柏 | male | 1986-01-01 | **no** | parent of nephew |

Marriage from `2003-11-02`.

| Query meaning | Gold | vs alpha |
|---|---|---|
| Members | 10 | same count, different ids |
| Female | 5 | same count, different ids |
| Adult female | **4** (self, adult daughter, mother, mil) | alpha **3** |
| Minor female | **1** (minor daughter) | alpha **2** |
| Adult male | **3** | alpha **4** |
| Minor male | **2** (son, nephew) | alpha **1** |
| My daughters | **2** | alpha **1** |
| Father-in-law / husband's father | 海峰 | ≠ 梅山, ≠ 夏松 |
| Spouse's father-in-law | 夏松 (speaker's father) | ≠ 海峰 |

### 3.3 gamma — empty in-laws, ambiguity, missing field, date boundary, speaker change

Address `address:gamma`. Default speaker `person:gamma_self`. No spouse edges.

| id | name | gender | dob | in household | notes |
|---|---|---|---|---|---|
| person:gamma_self | 顾北 | male | 1979-10-10 | yes | speaker, no spouse |
| person:gamma_dau_a | 顾棠 | female | 2011-03-03 | yes | self's daughter |
| person:gamma_dau_b | 顾薇 | female | 2014-09-09 | yes | self's daughter |
| person:gamma_just_adult | 顾成 | male | 2008-09-03 | yes | self's son; **18 today** |
| person:gamma_still_minor | 顾未 | male | 2008-09-04 | yes | self's son; **17** (one day younger) |
| person:gamma_roommate | 顾南 | female | 1985-05-05 | yes | second speaker |
| person:gamma_roommate_child | 顾苗 | female | 2019-01-20 | yes | roommate's only daughter |
| person:gamma_missing | 顾隐 | male | *absent* | **no** | named person, no `dob` |

Adults: 顾北, 顾成, 顾南 = 3.
Minors: 顾棠, 顾薇, 顾未, 顾苗 = 4.
If 顾成 used the still-minor dob, adult count would be 2. If 顾未 used the
just-adult dob, minor count would be 3. The one-day pair is the date-boundary
cell.

`我女儿是谁` as 顾北 → `ambiguous` (棠, 薇).
Same utterance as 顾南 → 顾苗 only.
`我岳父是谁` as either speaker → `relationship_not_found`.
`顾隐哪天出生` → `property_unavailable`.
Household members born in 1999 → count `found` / `0`.

Missing `dob` is **not** a resident, so household `adult`/`minor` counts remain
executable. A dedicated named-entity cell covers the missing field.

## 4. Composition cells

Each cell is a meaning, not a wording. Development uses family D
(canonical). Frozen uses family F (held-out). `+` marks a household-transfer
row that keeps family D and is development-only.

Forbidden plans are executed on the same household; they must not match gold
entity ids or count.

| Cell | Meaning | Gold IR (expanded) | Primary | Transfer | Dev | Frozen |
|---|---|---|---|---|---:|---:|
| S1 | List household members | `select` `current_household`→`member` property null | alpha | beta+ | 2 | 1 |
| S2 | Count household members | `count` same subject | alpha | gamma+ | 2 | 1 |
| S3 | “My household” still members | same as S1; **not** `self`→`member` | alpha | — | 1 | 1 |
| F1 | Count females | `count` members + `gender=female` | alpha | beta+ | 2 | 1 |
| F2 | Count adults | members + `adult` | alpha | — | 1 | 1 |
| F3 | Count minors | members + `minor` | alpha | gamma+ | 2 | 1 |
| F4 | Count adult females | `adult` **and** `gender=female` | alpha | beta+ | 2 | 1 |
| F5 | Count minor females | `minor` **and** `gender=female` | alpha | beta+ | 2 | 1 |
| F6 | List females | `select` members + `gender=female` | alpha | — | 1 | 1 |
| F7 | Count males | members + `gender=male` | alpha | — | 1 | 1 |
| C1 | My children (list) | `self`→`child` | alpha | beta+ | 2 | 1 |
| C2 | Household minors (list) | members + `minor` | alpha | — | 1 | 1 |
| C3 | My daughters | `self`→`daughter` | alpha | beta+ | 2 | 1 |
| C4 | My sons (count) | `count` `self`→`son` | alpha | — | 1 | 1 |
| K1 | Father-in-law | `self`→`father_in_law` | alpha | beta+ | 2 | 1 |
| K2 | Wife's father | `self`→`wife` then `father` | alpha | — | 1 | 1 |
| K3 | Spouse's father-in-law | `self`→`spouse` then `father_in_law` | alpha | beta+ | 2 | 1 |
| K4 | Speaker's father | `self`→`father` | alpha | — | 1 | 1 |
| K5 | Husband's father | `self`→`husband` then `father` | beta | — | 1 | 1 |
| K6 | Empty father-in-law | `father_in_law`; status `relationship_not_found` | gamma | — | 1 | 1 |
| O1 | Wife stored birth date | `select(birth_date)` `wife` | alpha | — | 1 | 1 |
| O2 | Wife next-birthday days | `annual_occurrence` `wife` `mode=days` | alpha | — | 1 | 1 |
| O3 | Wife age years | `date_difference` `wife` `mode=years` | alpha | — | 1 | 1 |
| O4 | Household oldest | `argmin` members `birth_date` | alpha | — | 1 | 1 |
| E1 | Empty adult daughters | `count` `self`→`daughter` + `adult` → 0 | alpha | — | 1 | 1 |
| E2 | Missing birth date | `select(birth_date)` named 顾隐 → `property_unavailable` | gamma | — | 1 | 1 |
| E4 | Date-boundary adult count | members + `adult` includes 顾成, excludes 顾未 | gamma | — | 1 | 1 |
| E5 | Date-boundary minor count | members + `minor` includes 顾未, excludes 顾成 | gamma | — | 1 | 1 |
| E6 | Zero-match year | members + `birth_date` `date_range` 1999 → count 0 | gamma | — | 1 | 1 |
| E7 | Bare “他们” | `kind=unresolved` (see §6) | alpha | — | 1 | 1 |
| E8 | Speaker change, “我女儿” / “我闺女” | same wording; 顾北 `ambiguous` vs 顾南 = 顾苗. E3 is this cell. | gamma | — | 2 | 2 |
| **Standalone total** | | | | | **42** | **36** |

S3's forbidden plan `self`→`member` is not a gold case. Tests execute it and
assert `semantic_plan_unsupported`.

K1 vs K2 vs K3 are three distinct expanded paths. K2 gold is `wife` then
`father` only. K1 gold is `father_in_law` only. K3 gold is `spouse` then
`father_in_law` only. Count-collision cells (F1/F7, C1 vs C2 lists) also
assert `expected_entity_ids`.

## 5. Utterance inventory

Frozen utterances must be absent from `_semantic_planner_examples`,
`_PLANNER_INSTRUCTIONS`, `semantic_planner_eval.yaml`, held-out, synthetic,
age-filter, and date-interval datasets. Development may resemble eval
canonicals because the **household and gold ids** differ; they still must not
be copied into prompt examples.

### 5.1 Development (family D)

| id | cell | hh | utterance | expected_status | gold highlight |
|---|---|---|---|---|---|
| dev-alpha-S1-1 | S1 | alpha | 请列出家里的人 | found | 10 member ids |
| dev-beta-S1-1 | S1+ | beta | 请列出家里的人 | found | 10 beta ids |
| dev-alpha-S2-1 | S2 | alpha | 家里有几个人 | found | 10 |
| dev-gamma-S2-1 | S2+ | gamma | 家里有几个人 | found | 7 |
| dev-alpha-S3-1 | S3 | alpha | 我家里都有谁 | found | same ids as S1; not `self→member` |
| dev-alpha-F1-1 | F1 | alpha | 家里有几个女的 | found | 5 |
| dev-beta-F1-1 | F1+ | beta | 家里有几个女的 | found | 5 (different ids) |
| dev-alpha-F2-1 | F2 | alpha | 家里有几个成年人 | found | 7 |
| dev-alpha-F3-1 | F3 | alpha | 家里有几个未成年人 | found | 3 |
| dev-gamma-F3-1 | F3+ | gamma | 家里有几个未成年人 | found | 4 |
| dev-alpha-F4-1 | F4 | alpha | 家里有几个成年女性 | found | 3 |
| dev-beta-F4-1 | F4+ | beta | 家里有几个成年女性 | found | 4 |
| dev-alpha-F5-1 | F5 | alpha | 家里有几个未成年女性 | found | 2 |
| dev-beta-F5-1 | F5+ | beta | 家里有几个未成年女性 | found | 1 |
| dev-alpha-F6-1 | F6 | alpha | 请列出家里的女性 | found | 5 female ids |
| dev-alpha-F7-1 | F7 | alpha | 家里有几个男的 | found | 5 |
| dev-alpha-C1-1 | C1 | alpha | 请列出我的孩子 | found | 安晨, 安野, 安秋 |
| dev-beta-C1-1 | C1+ | beta | 请列出我的孩子 | found | 林晴, 林露, 林川 |
| dev-alpha-C2-1 | C2 | alpha | 请列出家里的未成年人 | found | 安野, 安秋, 安禾 |
| dev-alpha-C3-1 | C3 | alpha | 请列出我的女儿 | found | 安秋 |
| dev-beta-C3-1 | C3+ | beta | 请列出我的女儿 | found | 林晴, 林露 |
| dev-alpha-C4-1 | C4 | alpha | 我有几个儿子 | found | 2 |
| dev-alpha-K1-1 | K1 | alpha | 我的岳父是哪位 | found | 梅山 |
| dev-beta-K1-1 | K1+ | beta | 我的岳父是哪位 | found | 海峰 |
| dev-alpha-K2-1 | K2 | alpha | 我妻子的父亲是哪位 | found | 梅山; not `father_in_law` |
| dev-alpha-K3-1 | K3 | alpha | 我配偶的岳父是哪位 | found | 安石 |
| dev-beta-K3-1 | K3+ | beta | 我配偶的岳父是哪位 | found | 夏松 |
| dev-alpha-K4-1 | K4 | alpha | 我的父亲是哪位 | found | 安石 |
| dev-beta-K5-1 | K5 | beta | 我丈夫的父亲是哪位 | found | 海峰 |
| dev-gamma-K6-1 | K6 | gamma | 我的岳父是哪位 | relationship_not_found | empty in-laws |
| dev-alpha-O1-1 | O1 | alpha | 我妻子的出生日期是哪天 | found | 1982-04-20 |
| dev-alpha-O2-1 | O2 | alpha | 我妻子下个生日还有多少天 | found | days from clock to 04-20 |
| dev-alpha-O3-1 | O3 | alpha | 我妻子现在多少岁 | found | 44 years |
| dev-alpha-O4-1 | O4 | alpha | 家里谁最年长 | found | 梅山 |
| dev-alpha-E1-1 | E1 | alpha | 我有几个成年女儿 | found | 0 |
| dev-gamma-E2-1 | E2 | gamma | 顾隐哪天出生 | property_unavailable | missing dob |
| dev-gamma-E4-1 | E4 | gamma | 家里有几个成年人 | found | 3 (includes 顾成) |
| dev-gamma-E5-1 | E5 | gamma | 家里未成年的有几个 | found | 4 (includes 顾未) |
| dev-gamma-E6-1 | E6 | gamma | 家里1999年出生的有几个人 | found | 0 |
| dev-alpha-E7-1 | E7 | alpha | 他们是谁 | ambiguous | unresolved; see §6 |
| dev-gamma-E8-1 | E8 | gamma | 我女儿是谁 | ambiguous | speaker 顾北 |
| dev-gamma-E8-2 | E8 | gamma | 我女儿是谁 | found | speaker 顾南 → 顾苗 |

E3 is `dev-gamma-E8-1` (ambiguous 我女儿 as 顾北). Do not list a duplicate.
Development standalone **42**. Do not add `dev-beta-F7-1` as padding.

Eval contains `Who is my father-in-law?`, so frozen K1/K6/G2/G3 must not use
that sentence. Beta uses 公公; gamma uses `Name the person who is my father in law.`
Count cells F1 and F7 also record `expected_entity_ids` because both counts
are 5 on alpha.

### 5.2 Frozen (family F) — held-out wording

None of these strings may appear in interpreter examples or the existing
eval/heldout/synthetic/age/date YAML.

| id | cell | hh | utterance |
|---|---|---|---|
| frz-alpha-S1-1 | S1 | alpha | Who are the people in this household? |
| frz-alpha-S2-1 | S2 | alpha | 本户现在一共几位成员？ |
| frz-alpha-S3-1 | S3 | alpha | 咱家这户人都有哪些？ |
| frz-beta-F1-1 | F1 | beta | How many females live in this household? |
| frz-alpha-F2-1 | F2 | alpha | 本户已满十八岁的成员有几位？ |
| frz-gamma-F3-1 | F3 | gamma | 咱家还没成年的有几位？ |
| frz-beta-F4-1 | F4 | beta | How many adult women are in this household? |
| frz-alpha-F5-1 | F5 | alpha | 本户未成年的女孩子有几个？ |
| frz-alpha-F6-1 | F6 | alpha | List the women who live here. |
| frz-beta-F7-1 | F7 | beta | 本户男性成员有几位？ |
| frz-beta-C1-1 | C1 | beta | 把我自己的子女都列出来 |
| frz-alpha-C2-1 | C2 | alpha | Who in this household is still a minor? |
| frz-beta-C3-1 | C3 | beta | 我自己的千金都有谁 |
| frz-alpha-C4-1 | C4 | alpha | 我名下的儿子一共几个？ |
| frz-beta-K1-1 | K1 | beta | 请指出我的公公是哪一位？ |
| frz-alpha-K2-1 | K2 | alpha | 我老婆她爸是哪一位？ |
| frz-alpha-K3-1 | K3 | alpha | 我爱人的岳父是谁？ |
| frz-alpha-K4-1 | K4 | alpha | 我自己的爸爸是哪一位？ |
| frz-beta-K5-1 | K5 | beta | 我先生的父亲叫什么身份？ |
| frz-gamma-K6-1 | K6 | gamma | Name the person who is my father in law. |
| frz-alpha-O1-1 | O1 | alpha | What is my wife's date of birth on record? |
| frz-alpha-O2-1 | O2 | alpha | 我太太距离下次过生日还有几天？ |
| frz-alpha-O3-1 | O3 | alpha | 我太太今年满几周岁了？ |
| frz-alpha-O4-1 | O4 | alpha | 本户出生最早的是哪一位？ |
| frz-alpha-E1-1 | E1 | alpha | How many of my daughters are adults? |
| frz-gamma-E2-1 | E2 | gamma | 顾隐的出生日期能查到吗？ |
| frz-gamma-E4-1 | E4 | gamma | 本户已成年的一共几人？ |
| frz-gamma-E5-1 | E5 | gamma | How many household members are still minors? |
| frz-gamma-E6-1 | E6 | gamma | 本户在一九九九年出生的有几位？ |
| frz-alpha-E7-1 | E7 | alpha | Who are they? |
| frz-gamma-E8-1 | E8 | gamma | 我闺女是哪位？ (speaker 顾北, ambiguous) |
| frz-gamma-E8-2 | E8 | gamma | 我闺女是哪位？ (speaker 顾南, 顾苗) |
| frz-beta-K2-1 | K2 transfer | beta | Which person is my wife's father? — female speaker, no wife. `relationship_not_found`. Distinguishes K2 from K1 on beta (K1 公公 finds 海峰). |
| frz-alpha-O1-2 | O1 | alpha | 我妻子出生那天是哪一号？ — `select(birth_date)`, not countdown |
| frz-beta-C2-1 | C2 transfer | beta | 本户的小孩子都有谁 |
| frz-alpha-F4-2 | F4 | alpha | 成年的女性成员有几位？ |

That's 36 frozen ids. E3 is covered by `frz-gamma-E8-1` (ambiguous 闺女).

`frz-beta-K2-1` is a required contrast: on a female speaker, “wife's father”
must not silently become father-in-law.

## 6. Ambiguity and acceptable plans

| Case | Rule |
|---|---|
| K2 male speaker | `wife` then `father` only. Not `father_in_law`. |
| K1 | `father_in_law` only; not `wife`+`father` and not `spouse`+`father_in_law` |
| K5 | `husband` then `father` only |
| E7 standalone 他们 / Who are they? | Gold `kind=unresolved` (cardinality stays `single`) → `ambiguous`. Acceptable: `kind=discourse`, `turn_offset=1`, `entity_type=person` → `discourse_context_missing`. Accept **either** status. A named-entity or household guess is wrong. |
| E8 顾北 我女儿/我闺女 | `ambiguous`; candidates 顾棠, 顾薇. Not roommate's daughter |
| G7 / G8 last turns | `kind=discourse` only. No restart-as-wife/husband/daughter/household-member |

## 7. Sequences (score last turn only)

History is prior **user** strings. The planner inserts the existing assistant
boundary. Preceding turns are not jointly scored.

### 7.1 Development

| id | hh | history → last utterance | last gold | leak that must fail |
|---|---|---|---|---|
| dev-seq-G1 | alpha | 家里有几个成年人 → **家里有几个男的** | F7 male count 5 | keep `adult`, drop gender (lucky 7 or 4) |
| dev-seq-G2 | alpha | 我的父亲是哪位 → **我的岳父是哪位** | K1 梅山 | `father` then `spouse` |
| dev-seq-G3 | alpha | 我的岳父是哪位 → **我妻子下个生日还有多少天** | O2 wife countdown | leftover in-law hops; `select(birth_date)` |
| dev-seq-G4 | alpha | 请列出家里的人 → **我妻子的出生日期是哪天** | same as standalone O1 | must match O1 exactly |
| dev-seq-G5 | alpha | 请列出我的孩子 → **我家里都有谁** | S3 household list | `self→member` or leftover `child` |
| dev-seq-G6 | alpha | 家里有几个女的 → **家里有几个成年女性** | F4 3 | drop gender or drop adult |
| dev-seq-G7 | alpha | 请列出家里的人 → **他们谁最年长** | `argmin` `discourse` collection `birth_date` | named guess; `self` |
| dev-seq-G8 | alpha | 我妻子的出生日期是哪天 → **她下个生日还有几天** | `annual_occurrence` `discourse` `turn_offset=1` single | restart-as-`wife` or daughter countdown |

### 7.2 Frozen

Same eight compositions, held-out wording, mostly beta so leaked adult/gender
counts cannot reuse alpha numbers.

| id | hh | history → last |
|---|---|---|
| frz-seq-G1 | beta | 本户已满十八岁的成员有几位？ → **本户男性成员有几位？** |
| frz-seq-G2 | beta | 我自己的爸爸是哪一位？ → **请指出我的公公是哪一位？** |
| frz-seq-G3 | beta | 请指出我的公公是哪一位？ → **我先生下次生日还有几天？** gold `husband` + `annual_occurrence`. Same leftover-hop composition as male-speaker wife countdown. |
| frz-seq-G4 | beta | Who are the people in this household? → **What is my husband's date of birth on record?** |
| frz-seq-G5 | beta | 把我自己的子女都列出来 → **咱家这户人都有哪些？** |
| frz-seq-G6 | beta | How many females live in this household? → **How many adult women are in this household?** |
| frz-seq-G7 | beta | Who are the people in this household? → **Which of them was born first?** |
| frz-seq-G8 | gamma | 我闺女是哪位？ (顾南, unique) → **她下个生日还有几天？** |

G3 frozen cannot copy the male-speaker wife countdown onto beta. The revised
husband countdown is the same **composition** (complete first-person kinship
after an in-law turn).

## 8. Deterministic checks (Ticket 4 tests, no LLM)

1. Every gold plan `validation_code == VALID`, except none of the golds are
   the unsupported `self→member` plan.
2. Execute golds on the named household/clock; status, ids, value, unit match.
3. Execute each cell's forbidden plan; ids or count differ (or status is
   unsupported).
4. Frozen utterances ∉ examples, instructions, eval, heldout, synthetic,
   age-filter, date-interval.
5. `load_semantic_eval_cases()` still loads `benchmarks/semantic_planner_eval.yaml`
   and `len(cases) >= 100`.
6. `SCORING_REVISION` string unchanged.
7. Development ∩ frozen utterance strings = ∅ (speaker variants of the same
   frozen string are the same split).
8. Fingerprints: sha256 of composition YAML, household trees, ontology,
   annotation-guide, this matrix, and `tests/test_composition_eval.py`.

Critical combinations for Codex to inspect after generation (not before):
K1/K2/K3/K4 on alpha; K1/K5/K2-empty on beta; F4 vs F1 vs F2; C1 vs C2;
E4 vs E5; G1, G2, G3 sequences; E8 speaker pair.

## 9. Explicit non-goals

- Do not add these utterances to `_semantic_planner_examples`.
- Do not change scoring to treat `found` or a colliding count as a pass.
- Do not implement Ticket 2 contracts.
- Do not run GPU accuracy (Ticket 6).

## 10. Codex answers (locked)

See `codex-approval.md`. Locked before generation:

1. Size **42 / 36 / 8 / 8**. No F7 padding.
2. Households approved with entity-id discipline on count collisions.
3. G7/G8 gold is `discourse` only.
4. E7 accepts both clarification statuses; not a guessed person.
5. Frozen exact-strings disjoint from examples/eval. `frz-beta-K2-1` uses
   held-out English.
6. K2/K5 do **not** accept `father_in_law`.
7. Required ticket contrasts are covered.
