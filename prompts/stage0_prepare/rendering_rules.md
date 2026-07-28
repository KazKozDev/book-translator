Respond with ONLY a JSON array, no prose, no code fence. Each element:
{"source": "<expression exactly as written in the $source_name text>", "target": "<the $target_name rendering>", "kind": "person|place|organisation|work|term|other", "mode": "exact|inflectable|preferred"}

Rules:
- One element per distinct proper noun. Do not repeat.
- "source" must be copied character for character from the document. You may add a proper noun you can see in the context quotes that is missing from the candidate list, as long as you copy it exactly; never write a form the document does not contain.
- "target" is the base form only. Do not add grammatical endings, articles, or explanations.
- Leave titles and honorifics out of "source": write "Fenwick", never "Mrs. Fenwick" or "Mr. and Mrs. Fenwick".
- Distinct source forms need distinct renderings: a singular name and its plural or family form must not both become the same target string.
- Names of people and places are normally transcribed, not translated by meaning, unless the document's tradition clearly demands otherwise.
Choose "mode" per term:
- "exact": the target must appear letter for letter every time — codes, invented brands, titles of works, anything a reader would notice being altered.
- "inflectable": the lexical choice is fixed but the target language may inflect it — the normal choice for names of people and places.
- "preferred": use this rendering where it fits, and allow a freer translation where it does not — descriptive names and common-noun terms.
- If there are no proper nouns at all, respond with [].

## genre_line

The document is: $genre.
