## Ship It

Build a reusable enrichment prep script that takes an API endpoint and an output filename, fetches JSON, extracts specific fields, and writes a CSV ready for Clay import. This mirrors the data engineering step that precedes a Clay waterfall: you clean and structure raw API output into the column format Clay expects.

```bash
cat << 'SCRIPT' > /tmp/enrichment_prep.sh
#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <json_file> <output_csv>"
    echo "Example: $0 apollo_response.json apollo_leads.csv"
    exit 1
fi

JSON_FILE="$1"
OUTPUT_CSV="$2"

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: '$JSON_FILE' not found"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Install with: brew install jq"
    exit 1
fi

echo "company_name,domain,linkedin_url" > "$OUTPUT_CSV"

jq -r '.people[] | "\(.organization.name),\(.organization.website_url // "N/A"),\(.linkedin_url // "N/A")"' "$JSON_FILE" >> "$OUTPUT_CSV" 2>errors.log

ROW_COUNT=$(wc -l < "$OUTPUT_CSV")
DATA_ROWS=$((ROW_COUNT - 1))

echo "Wrote $DATA_ROWS prospects to $OUTPUT_CSV"
echo "Preview:"
head -n 4 "$OUTPUT_CSV"

if [ -s errors.log ]; then
    echo "Warnings logged in errors.log"
fi
SCRIPT

chmod +x /tmp/enrichment_prep.sh
```

Test it with a realistic Apollo-style JSON payload:

```bash
cat << 'JSON' > /tmp/apollo_sample.json
{
  "people": [
    {
      "first_name": "Jane",
      "organization": { "name": "Acme Corp", "website_url": "acme.com" },
      "linkedin_url": "https://linkedin.com/in/jane-doe"
    },
    {
      "first_name": "John",
      "organization": { "name": "Globex", "website_url": "globex.com" },
      "linkedin_url": null
    },
    {
      "first_name": "Alice",
      "organization": { "name": "Initech", "website_url": "initech.com" },
      "linkedin_url": "https://linkedin.com/in/alice-cooper"
    }
  ]
}
JSON

/tmp/enrichment_prep.sh /tmp/apollo_sample.json /tmp/apollo_leads.csv
```

Expected output:

```
Wrote 3 prospects to /tmp/apollo_leads.csv
Preview:
company_name,domain,linkedin_url
Acme Corp,acme.com,https://linkedin.com/in/jane-doe
Globex,globex.com,N/A
```

The CSV this produces — `company_name`, `domain`, `linkedin_url` — is the column structure Clay expects when you upload a source table. The shell script did the extraction and transformation that a data pipeline in Clay's enrichment waterfall assumes has already happened upstream. When you later configure a waterfall in Clay (e.g., try Apollo first, fall back to Hunter, validate with NeverBounce), the data flowing through that waterfall was shaped by exactly this kind of terminal-based preparation.