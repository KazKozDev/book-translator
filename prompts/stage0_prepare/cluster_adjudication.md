These groups of expressions come from one $source_name document. A clustering step guessed that the forms inside each group name the same entity, or could not decide. Rule on each group.

GROUPS:
$listing

Respond with ONLY a JSON array, no prose, no code fence. One element per numbered group:
{"group": <number>, "same_entity": true|false, "canonical": "<the form to use as the entry, copied exactly from the group>"}

Rules:
- same_entity is true only when every form in the group refers to one and the same entity.
- A singular name and its plural or family form are DIFFERENT entities: "Dursley" and "Dursleys" must stay separate, and so must a person and the place named after them.
- A name with a title and the bare name are the same entity ("Mrs. Fenwick" and "Fenwick"); prefer the bare name as the canonical form.
- "canonical" must be one of the forms shown in that group, copied character for character. When same_entity is false, give the form that is most usable on its own.
- Rule on every group. Do not add groups.
