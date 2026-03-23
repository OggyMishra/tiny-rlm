# Extraction Schemas

This directory contains YAML schema definitions for the Universal Extraction Service (UES).

## Schema Structure

Each schema is a standalone YAML file that defines:

1. **Metadata**: Name, version, description
2. **LLM Configuration**: Model, temperature, max tokens
3. **Prompt Configuration**: Instructions, system prompt, template
4. **Schema**: Field definitions with types and descriptions
5. **Optional**: Field metadata (citations, scoring)
6. **Optional**: Field type mapping (for specialized handling)
7. **Validation Config**: Validation rules
8. **Quality Config**: Quality thresholds

## Available Schemas

### 1. M&A Transaction Schema (`mna-transaction-schema.yaml`)
- **Purpose**: Extract comprehensive M&A transaction data
- **Fields**: 86 fields covering deal dates, parties, consideration, financing, advisors
- **Features**:
  - ✅ Citation tracking enabled
  - ✅ Confidence scoring enabled
  - ✅ Field type mapping for specialized handling
  - ✅ Supports DateField, NumericField, PercentageField, EntityField, etc.

### 2. Invoice Schema (`invoice-schema.yaml`)
- **Purpose**: Simple invoice data extraction
- **Fields**: 12 basic fields (invoice number, dates, amounts, parties)
- **Features**:
  - ❌ No citation tracking (simple extraction)
  - ❌ No field type mapping (uses default string handling)
  - ✅ Clean, minimal schema for teams that don't need advanced features

## Schema Examples

### Simple Schema (WITHOUT Citations)

```yaml
# Invoice Schema - Minimal configuration
name: "invoice"
version: "v1"
description: "Extract invoice data"

llm:
  service_model: "universal-extraction-service-primary"
  temperature: 0.1

prompt:
  instructions: "Extract invoice information..."
  system_prompt: "You are a precise invoice extractor..."
  template: "{{ text }}"

schema:
  invoice_number:
    type: string
    description: "Invoice number"
    required: true

  total_amount:
    type: string
    description: "Total amount"
    required: false

# No field_metadata or field_type_mapping needed!
```

**Extraction Output** (simple):
```json
{
  "invoice_number": "INV-12345",
  "total_amount": "1000 USD"
}
```

---

### Advanced Schema (WITH Citations)

```yaml
# M&A Schema - Advanced configuration
name: "mna_transaction"
version: "v2"

llm:
  service_model: "universal-extraction-service-primary"

# OPTIONAL: Enable citation tracking
field_metadata:
  enable_citations: true
  enable_confidence_scoring: true
  enable_validation_scoring: true
  citation_properties:
    source_statement_ids:
      type: array
      items:
        type: integer
    confidence_score:
      type: enum
      enum: ["high", "medium", "low"]
    validation_score:
      type: number

# OPTIONAL: Map fields to specialized types
field_type_mapping:
  date_fields:
    - announcement_date
    - agreement_date
  monetary_fields:
    - consideration_size
  percentage_fields:
    - cash_consideration_percentage

schema:
  announcement_date:
    type: string
    format: date
    description: "Announcement date"

  consideration_size:
    type: string
    description: "Deal value with currency"
```

**Extraction Output** (with citations):
```json
{
  "announcement_date": {
    "value": "06/30/2025",
    "parsed_date": "2025-06-30",
    "date_format": "MM/DD/YYYY",
    "source_statement_ids": [1, 3],
    "confidence_score": "high",
    "validation_score": 0.9
  },
  "consideration_size": {
    "value": "100 million USD",
    "numeric_value": 100000000.0,
    "unit": "USD",
    "scale": "million",
    "source_statement_ids": [5],
    "confidence_score": "medium",
    "validation_score": 0.8
  }
}
```

## Creating Your Own Schema

### Step 1: Create a YAML file

Create a new file: `schemas/your-schema-name.yaml`

### Step 2: Define basic configuration

```yaml
name: "your_schema"
version: "v1"
description: "Extract your custom data"

llm:
  service_model: "universal-extraction-service-primary"
  temperature: 0.1
  max_tokens: 2000

prompt:
  instructions: |
    Extract information from the document.
    Be precise and factual.

  system_prompt: |
    You are a data extraction specialist.

  template: |
    Document: {{ text }}

schema:
  field_name:
    type: string
    description: "Description of what to extract"
    required: false
```

### Step 3: (Optional) Enable citations

Only add this if you need citation tracking:

```yaml
field_metadata:
  enable_citations: true
  enable_confidence_scoring: true
  citation_properties:
    source_statement_ids:
      type: array
      items:
        type: integer
    confidence_score:
      type: enum
      enum: ["high", "medium", "low"]
```

### Step 4: (Optional) Map field types

Only add this if you want specialized field handling:

```yaml
field_type_mapping:
  date_fields:
    - your_date_field
  monetary_fields:
    - your_amount_field
  percentage_fields:
    - your_percentage_field
```

## Field Types

Supported field types in `schema` section:

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text field | `"John Smith"` |
| `number` | Numeric value | `100.5` |
| `integer` | Integer value | `42` |
| `boolean` | True/false | `true` |
| `enum` | Fixed set of values | `enum: ["A", "B", "C"]` |
| `array` | List of items | `items: {type: string}` |
| `object` | Nested structure | `properties: {...}` |

## Specialized Field Types (via field_type_mapping)

When you map fields using `field_type_mapping`, they get specialized handling:

| Category | Pydantic Class | Features |
|----------|----------------|----------|
| `date_fields` | `DateField` | Automatic date parsing, format detection |
| `numeric_fields` | `NumericField` | Number parsing, scale detection (million, billion) |
| `monetary_fields` | `NumericField` | Currency extraction, scale handling |
| `percentage_fields` | `PercentageField` | Percentage normalization |
| `entity_fields` | `EntityField` | Entity name normalization |
| `categorical_fields` | `CategoricalField` | Category validation |
| `text_fields` | `TextField` | Text cleaning and normalization |

## Best Practices

### ✅ DO:
- Use clear, descriptive field names
- Provide detailed field descriptions
- Set `required: true` only for critical fields
- Use enums for categorical fields with known values
- Add detailed extraction instructions in prompts
- Start simple without citations, add them later if needed

### ❌ DON'T:
- Don't enable citations if you don't need them
- Don't use field_type_mapping unless you need specialized handling
- Don't make too many fields required
- Don't use overly complex nested structures
- Don't duplicate field definitions

## Testing Your Schema

Load and test your schema:

```python
from services.schema_loader import SchemaLoader

loader = SchemaLoader()
schema = loader.load_schema("your-schema-name")
print(schema.name, schema.version)
```

## Schema Versioning

Schemas are version-controlled via Git:

- Each schema has a `version` field (e.g., `v1`, `v2`)
- Load schemas at specific Git commit SHA for reproducibility
- Update version when making breaking changes

## Questions?

- Check existing schemas for examples
- See `src/models/schema_config.py` for Pydantic model definitions
- Review `src/services/schema_converter.py` for conversion logic
